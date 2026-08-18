import torch.nn as nn
import torch
import einops

"""
# initiate W
torch.empty() # 
nn.init.trucn_normal_()

"""

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        std = (2 / (out_features + in_features)) ** 0.5
        mean = 0
        W = torch.empty(out_features, in_features, device=device, dtype=dtype)
        W = (nn.init.trunc_normal_(W, mean=mean, std=std, a=-3 * std, b=3 * std))
        self.W = nn.Parameter(W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("... i, o i -> ... o", x, self.W)



class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings # vocab_size
        self.embedding_dim = embedding_dim   # d_model

        m = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        std = 1
        mean = 0
        m = nn.init.trunc_normal_(m, mean=mean, std=std, a=-3 * std, b=3 * std)
        self.embedding_matrix = nn.Parameter(m)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_matrix[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        self.eps = eps
        self.d_model = d_model
        # init gi, [d_model]
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model] @ W.T -> [batch, seq_len, d_model]

        # upcast to fp32
        in_type = x.dtype
        x = x.to(torch.float32)

        pow2 = x.pow(2)                           # [batch, seq_len, d_model]
        pow2_sum = pow2.sum(-1, keepdim = True)   # [batch, seq_len, 1]
        pow2_sum_avg = pow2_sum / self.d_model    # [batch, seq_len, 1]
        rms = torch.sqrt(pow2_sum_avg + self.eps) # [batch, seq_len, 1]

        # [batch, seq_len, d_model] / [batch, seq_len, 1]  * [1, 1, d_model]
        rms_norm = x / rms * self.g               # [batch, seq_len, d_model]

        return rms_norm.to(in_type)


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int=None, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        # d_up % 64 == 0
        self.d_ff = round(d_model * 8 / 3 / 64) * 64 if d_ff is None else d_ff

        self.w1 = Linear(d_model, self.d_ff) # up proj
        self.w2 = Linear(self.d_ff, d_model) # down proj
        self.w3 = Linear(d_model, self.d_ff) # same as up proj
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU
        x_up = self.w1(x)
        silu = x_up * torch.sigmoid(x_up) # [..., seq_len, d_up]
        # GLU
        glu = silu * self.w3(x) # SiLU(x) * W3x

        # down_proj
        return self.w2(glu)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k # d_head
        self.max_seq_len = max_seq_len

        # theata_i,k
        k = torch.arange(1, d_k // 2 + 1, device=device).float() # [1,... d_k / 2]
        inv_theta = 1 / theta ** ((2 * k - 2) / d_k)
        pos = torch.arange(max_seq_len, device=device).float() # [max_seq_len]

        # pos: [max_seq_len] * inv_theta: [d_k / 2] -> [max_seq_len, d_k / 2]
        # outer product
        angles = torch.einsum("i, k -> i k", pos, inv_theta)

        # pre-compute lookup table 
        self.register_buffer("cos", angles.cos(), persistent=False) # [max_seq_len, d_k / 2]
        self.register_buffer("sin", angles.sin(), persistent=False)

     
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: [..., seq_len, d_k] -> out: [..., seq_len, d_k]
        
        # token_positions: [..., seq_len]
        cos = self.cos[token_positions] # [..., seq_len, d_k / 2]
        sin = self.sin[token_positions]

        # kth pair in x
        x1 = x[..., 0::2] # even [..., seq_len, d_k / 2]
        x2 = x[..., 1::2] # odd

        # Ri[row] * x, row vector dot product, only 2 non-zero each row
        o1 = x1 * cos - x2 * sin # [..., seq_len, d_k / 2]
        o2 = x1 * sin + x2 * cos

        return torch.stack((o1, o2), dim=-1).flatten(-2)

def softmax(x: torch.Tensor, dim_i: int) -> torch.Tensor:
    max_i = x.amax(dim=dim_i, keepdim=True)
    e = torch.exp(x - max_i)
    return e / e.sum(dim=dim_i, keepdim=True)


# keys:    [batch, ..., seq_len, d_k]
# queries: [batch, ..., seq_len, d_k]
# values:  [batch, ..., seq_len, d_v]
# mask:    [            seq_len, seq_len]
def scaled_dot_product_attn(queries: torch.Tensor, keys: torch.Tensor,
                            values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logits = torch.einsum("... n d, ... m d -> ... n m", queries, keys)
    d_k = queries.shape[-1]
    logits = logits / (d_k ** 0.5) # [..., seq_len, seq_len]

    # replace mask == False with -inf
    logits = logits.masked_fill(~mask, float('-inf'))
    probs = softmax(logits, -1)
    return torch.einsum("... n m, ... m d -> ... n d", probs, values)


class Causal_multihead_attn(nn.Module):
    """
    Causal MHA Attention:
    
    x:                    [..., S, in_features]
    self.w (wq/wk/wv/wo): [out_features, in_features]

    out:                  [..., S, in_features]

    """
    def __init__(self, d_model: int, num_head: int, theta: float=None, max_seq_len: int=None, 
                 device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.d_head = d_model // num_head

        # qkv projection matrix
        self.wq = Linear(d_model, d_model, device, dtype)
        self.wk = Linear(d_model, d_model, device, dtype)
        self.wv = Linear(d_model, d_model, device, dtype)
        self.wo = Linear(d_model, d_model, device, dtype)

        # positional embedding matrix
        self.theta = theta
        self.max_seq_len = max_seq_len
        if self.theta is not None and self.max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(self.theta, self.d_head, self.max_seq_len, device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor=None) -> torch.Tensor:
        # causal mask
        seq_len = x.shape[-2]
        no_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).triu(diagonal=1)

        # qkv reshape d_model = h * head_dim
        # x: [..., seq_len, d_model]
        # q: [..., d_model, d_head] -> [..., num_head, seq_len, d_head]
        queries = einops.rearrange(self.wq(x), "... n (h d) -> ... h n d", h=self.num_head)
        keys = einops.rearrange(self.wk(x), "... n (h d) -> ... h n d", h=self.num_head)
        values = einops.rearrange(self.wv(x), "... n (h d) -> ... h n d", h=self.num_head)

        # RoPE
        if self.theta is not None and self.max_seq_len is not None:
            token_positions = torch.arange(seq_len, device=x.device) if token_positions is None else token_positions

            queries = self.rope(queries, token_positions)
            keys = self.rope(keys, token_positions)

        attn_out = scaled_dot_product_attn(queries, keys, values, ~no_mask)
        attn_out = einops.rearrange(attn_out, "... h n d -> ... n (h d)")
        attn_out = self.wo(attn_out)
        return attn_out


class Transformer(nn.Module):
    """
    One Transformer Block:
    
    x:   [(B, S), in_features]
    out: [(B, S), in_features]

    """
    def __init__(self, d_model: int, num_head: int, d_ff: int, theta: float, max_seq_len: int,
                 device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.d_ff = d_ff

        self.theta = theta
        self.max_seq_len = max_seq_len

        # modules
        self.norm1 = RMSNorm(self.d_model, device=device) # each has own learnable g
        self.norm2 = RMSNorm(self.d_model, device=device)
        self.causal_runner = Causal_multihead_attn(self.d_model, self.num_head, self.theta, self.max_seq_len, device)
        self.ffn = FFN(self.d_model, self.d_ff, device=device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pre-RMSNorm
        # qkv - softmax - o
        y = self.causal_runner(self.norm1(x))

        # residual
        y = y + x

        # pre-RMSNorm
        # FFN
        out = self.ffn(self.norm2(y))
        out = out + y

        return out


class Transformer_LM(nn.Module):
    def __init__(self, d_model: int, num_head: int, d_ff: int, 
                 vocab_size: int, num_layers: int, theta: float=None, max_seq_len: int=None,
                 device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.d_ff = d_ff

        self.theta = theta
        self.max_seq_len = max_seq_len

        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # modules
        self.token_embedding = Embedding(vocab_size, d_model, device, dtype)

        self.norm_final = RMSNorm(self.d_model, device=device) 
        self.lm_head = Linear(d_model, vocab_size, device, dtype)
        # num_layers Transformer blocks
        self.layers = nn.ModuleList([Transformer(d_model, num_head, d_ff, theta, max_seq_len, device, dtype)
                       for _ in range(num_layers)])

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(token_ids)

        for layer in self.layers:
            x = layer(x)

        # final-RMSNorm
        # LM [..., d_model] -> [..., vocab_size]
        out = self.lm_head(self.norm_final(x))

        return out
