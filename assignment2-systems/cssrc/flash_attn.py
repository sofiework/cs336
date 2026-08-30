import torch, einops
import math
import triton
import triton.language as tl




class Flash_attn_torch(torch.autograd.Function):
    Br = 16
    Bc = 16

    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal=False) -> torch.Tensor:
       
        N, d = Q.shape[-2:]
        Br, Bc = Flash_attn_torch.Br, Flash_attn_torch.Bc

        # blocks
        Tr = math.ceil(N / Br)
        Tc = math.ceil(N / Bc)

        scale = 1.0 / math.sqrt(d) # QKT / d**0.5

        # initiate O, L
        O = torch.empty_like(Q)
        L = torch.empty_like(Q[..., 0]) # = torch.empty(Q.shape[:-1])

        # loop query blocks
        for i in range(Tr):
            # load Qi, initiate Oi
            Qi = Q[..., Br * i : Br * (i + 1), : ] # [..., Br, d]
            Oi = torch.zeros_like(Qi)

            # m, l
            mi = torch.full(Qi.shape[:-1], float("-inf"), device=Q.device, dtype=Q.dtype) # [..., Br]
            li = torch.zeros_like(mi)  # [..., Br]

            # loop kv blocks
            for j in range(Tc):
                Kj = K[..., Bc * j : Bc * (j + 1), :] # [..., Bc, d]
                Vj = V[..., Bc * j : Bc * (j + 1), :] # [..., Bc, d]

                # QKT
                S = torch.einsum("... r d, ... c d -> ... r c", Qi, Kj) * scale # [..., Br, Bc]

                # softmax
                m_new = torch.maximum(mi, torch.amax(S, dim=-1)) # [..., Br]
                P = torch.exp(S - m_new.unsqueeze(-1))           # [..., Br, Bc]

                # l
                delta = torch.exp(mi - m_new)      # [..., Br]
                li = delta * li + P.sum(dim=-1) # [..., Br]

                # @V
                Oi = delta.unsqueeze(-1) * Oi + P @ Vj # [..., Br, d]

                # wb
                mi = m_new
 
            O[..., Br * i : Br * (i + 1), :] = Oi / li.unsqueeze(-1) # [..., Br, d]
            L[..., Br * i : Br * (i + 1)] = mi + torch.log(li)    # [..., Br]

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O
            

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError




# one FA kernel output [Q_TILE_SIZE, D] elements
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr
):
    # args: pointer, stride, tensor_size / tile_size, is_causal 
    # stride_args: batch_stride, row_stride, dim_stride
    
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # Vj.shape = Kj.shape
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # Oi.shape = Qi.shape
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )


    # fp32 only inside kernel, input/output in fp16
    # load q l o
    qi = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero") # boundary_check's dim order is by shape=[N_QUERIES, D]
    # zero initiate li oi instead of loading (its torch.empty() initiated garbage)
    li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    oi = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    
    # initiate mi: [Q_TILE_SIZE,] one element per-row
    mi = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32) 

    # out of bound mask
    query_out = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

    # causal optimization - only loop tile with at least one causal==1
    if IS_CAUSAL:
        loop_bound = tl.cdiv((query_tile_index + 1) * Q_TILE_SIZE, K_TILE_SIZE)
        # if N_KEYS < N_QUERIES
        loop_bound = min(loop_bound, tl.cdiv(N_KEYS, K_TILE_SIZE))
    else:
        loop_bound = tl.cdiv(N_KEYS, K_TILE_SIZE)


    # loop j over dim blocks
    for j in range(loop_bound): 
        kj = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
        vj = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
        
        # qkT / un-normal softmax
        # s: [Q_TILE_SIZE, K_TILE_SIZE]
        s = tl.dot(qi, tl.trans(kj)) * scale

        # tail mask
        # query_out = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        key_out = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE) # j to global idx
        mask = key_out[None, :] < N_KEYS

        # tail & causal mask
        # [Q_TILE, 1] & [1, K_TILE] -> [Q_TILE, K_TILE] causal_mask
        if IS_CAUSAL:
            causal_mask = query_out[:, None] >= key_out[None, :] # query row idx >= key row idx
            mask = mask & causal_mask
        s = tl.where(mask, s, -1e6) # exp(-inf - m) = 0 contibutes 0 to sum_exp


        m_new = tl.maximum(mi, tl.max(s, axis=-1)) # [Q_TILE_SIZE,]
        p = tl.exp(s - m_new[:, None])

        # update l: [Q_TILE_SIZE,]
        delta = tl.exp(mi - m_new)
        li = li * delta + tl.sum(p, axis=-1)

        # o: [Q_TILE_SIZE, D]
        # cast p.dtype to vj.dtype (fp32 -> input dtype)
        oi = delta[:, None] * oi + tl.dot(p.to(vj.dtype), vj)

        # wb
        mi = m_new

        # advance ptr along K V rows
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))


    # once normalize oi, compute Li
    oi = (oi / li[:, None]).to(qi.dtype)
    Li = mi + tl.log(li)

    # wb
    tl.store(O_block_ptr, oi, boundary_check=(0,))
    tl.store(L_block_ptr, Li, boundary_check=(0,))


class Flash_attn_triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal=False) -> torch.Tensor:
        assert Q.is_cuda

        # reshape QKV to flatten B dim
        q_shape = Q.shape
    
        Q = einops.rearrange(Q, "... N D -> (...) N D").contiguous()
        K = einops.rearrange(K, "... N D -> (...) N D").contiguous()
        V = einops.rearrange(V, "... N D -> (...) N D").contiguous()

        # read shape
        B, N_QUERIES, D = Q.shape
        N_KEYS = K.shape[-2]
        Q_TILE_SIZE, K_TILE_SIZE = 16, 16
        scale = 1 / D ** 0.5

        # initiate O L
        O = torch.empty_like(Q)
        L = torch.empty((B, N_QUERIES), device=Q.device, dtype=torch.float32)

        
        # call kernel
        # launch grid (num_query_block, batch_size)
        grid = (triton.cdiv(N_QUERIES, Q_TILE_SIZE), B)
        flash_fwd_kernel[grid](
            Q, K, V, O, L, 
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES=N_QUERIES, N_KEYS=N_KEYS,
            scale=scale, D=D, 
            Q_TILE_SIZE=Q_TILE_SIZE, K_TILE_SIZE=K_TILE_SIZE,
            IS_CAUSAL=is_causal
        )

        ctx.q_shape = q_shape
        ctx.save_for_backward(Q, K, V, O, L) # save reshaped QKV
        ctx.is_causal = is_causal
        return O.reshape(q_shape)
        
    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError
        