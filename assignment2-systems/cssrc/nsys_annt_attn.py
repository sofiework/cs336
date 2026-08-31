import torch.nn as nn
import torch
import einops
from cs336_basics.model import softmax

import torch.cuda.nvtx as nvtx

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attn(queries: torch.Tensor, keys: torch.Tensor,
                            values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    with nvtx.range("computing attention scores"):
        logits = torch.einsum("... n d, ... m d -> ... n m", queries, keys)
        d_k = queries.shape[-1]
        logits = logits / (d_k ** 0.5) # [..., seq_len, seq_len]

    # replace mask == False with -inf
    logits = logits.masked_fill(~mask, float('-inf'))

    with nvtx.range("computing softmax"):
        probs = softmax(logits, -1)

    with nvtx.range("computing out projection"):
        out = torch.einsum("... n m, ... m d -> ... n d", probs, values)
    return out