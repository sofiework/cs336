import torch
import torch.nn as nn
import einops, math

from cs336_basics.model import softmax


# pytorch
# q: [S, num_head, d_head], S = num_seq_in_batch (one query per seq)
# caches: [num_blocks, B, num_head_kv, head_dim] -> pool = num_blocks * B slots
# block_tables: [S, max_num_blocks]
# ctx_lens: 

def paged_attention_torch(q, k_cache, v_cache, block_tables, ctx_lens):
    # args
    S, H, d_head = q.shape
    num_blocks, B, KVH, D = k_cache.shape
    group = H // KVH # GQA
    

    # flatten pool: [(num_blocks * B), KVH, D]
    k_flat = k_cache.view((num_blocks * B), KVH, D)
    v_flat = v_cache.view((num_blocks * B), KVH, D)

    # T = (max_num_blocks * B)

    # slots: [S, T], offset in flattent pool
    # block idx -> block start offset -> B of slot offset
    offset = torch.arange(B, device=q.device)                    # [B,]
    slots = block_tables[:, :, None] * B + offset[None, None, :] # [S, max_num_blocks, B]
    slots = einops.rearrange(slots, "s m b -> s (m b)")          # [S, T]


    # get k, v -> [S, T, KVH, D]
    k = k_flat[slots, :, :]
    v = v_flat[slots, :, :]


    # GQA attention - broadcast KV in q group (assert H % KVH == 0)
    # logits = qkT / scale -> [S, KVH, G, T]
    scale = D ** 0.5
    q_gqa = einops.rearrange(q, "s (k g) d -> s k g d", g=group)
    logits = torch.einsum("s k g d, s t k d -> s k g t", q_gqa, k) / scale

    # tail mask (kv_index < cached kv = ctx_lens) [S, 1, 1, T]
    tail_mask = torch.arange(slots.shape[1], device=q.device)[None, :] < ctx_lens[:, None]
    logits = torch.where(tail_mask[:, None, None, :], logits, float('-inf'))


    # softmax -> P
    # p @ v -> o
    # [S, KVH, G, T] @ [S, T, KVH, D] -> []
    p = softmax(logits, -1)
    output = torch.einsum("s k g t, s t k d -> s k g d", p, v).to(q.dtype)

    return einops.rearrange(output, "s k g d -> s (k g) d")


def main():
    torch.manual_seed(0)
    num_blocks, B, KVH, D, group = 32, 4, 2, 8, 4
    H, S = KVH * group, 3

    k_cache = torch.randn(num_blocks, B, KVH, D)
    v_cache = torch.randn(num_blocks, B, KVH, D)
    q = torch.randn(S, H, D)
    ctx_lens = torch.tensor([7, 3, 16])          # tail block, padded row, exact fit
    max_blocks = int((ctx_lens.max().item() + B - 1) // B)

    # give each sequence distinct physical blocks; padded entries stay 0
    block_tables = torch.zeros(S, max_blocks, dtype=torch.long)
    perm, p = torch.randperm(num_blocks), 0
    for s in range(S):
        n = int((ctx_lens[s].item() + B - 1) // B)
        block_tables[s, :n], p = perm[p:p + n], p + n

    out = paged_attention_torch(q, k_cache, v_cache, block_tables, ctx_lens)

    # reference: per sequence, gather only real tokens and do plain MHA
    for s in range(S):
        n = int((ctx_lens[s].item() + B - 1) // B)
        blocks = block_tables[s, :n]
        k = torch.cat([k_cache[b] for b in blocks])[:ctx_lens[s]]   # [ctx, KVH, D]
        v = torch.cat([v_cache[b] for b in blocks])[:ctx_lens[s]]
        k = k.repeat_interleave(group, dim=1)                       # [ctx, H, D]
        v = v.repeat_interleave(group, dim=1)
        logits = torch.einsum("hd,thd->ht", q[s], k) / math.sqrt(D)
        ref = torch.einsum("ht,thd->hd", logits.softmax(-1), v)
        print(f"seq {s}: ctx={ctx_lens[s].item():3d}  "
              f"err={(out[s] - ref).abs().max().item():.2e}")


if __name__ == "__main__":
    main()