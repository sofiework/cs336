import torch
import einops, math
from collections.abc import Iterable


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # logits: [..., vocab], logits - lm_head output (before softmax)
    # target: [...], token index in vocab

    # l = log(sum exp(logits_i)) - logits_target
    logits = logits - logits.amax(dim=-1, keepdim=True) # softmax invariant to shift
    log_sum = torch.log(torch.exp(logits).sum(dim=-1))

    # einops.rearrange(targets, "... -> ... 1")
    target_logit = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1) # [...]

    return (log_sum - target_logit).mean()


def lr_schedule(t: int, alpha_max: float, alpha_min: float, Twarm: int, Tc: int) -> float:
    if t < Twarm:
        alpha_t = t / Twarm * alpha_max

    elif Twarm <= t <= Tc:
        alpha_t = alpha_min + 0.5 * (1 + math.cos((t - Twarm) / (Tc - Twarm) * math.pi)) * (alpha_max - alpha_min)
    else:
        alpha_t = alpha_min
    return alpha_t


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float=1e-6):
    params = [p for p in parameters if p.grad is not None]

    # tensor.item() turn a single element tensor into float
    l2_norm = math.sqrt(sum((p.grad ** 2).sum().item() for p in params))
    if l2_norm > max_l2_norm:
        scale = max_l2_norm / (l2_norm + eps)
        for p in params:
            p.grad *= scale
    return
