import torch, random, typing, os
import einops, math
from collections.abc import Iterable
import numpy as np
import torch.nn as nn
from typing import BinaryIO, IO

from cs336_basics.model import Transformer_LM


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


@torch.no_grad()
def evaluate(model: Transformer_LM, val_data: np.ndarray, 
             batch_size: int, context_length: int, device=None, num_batches=20) -> float:
    model.eval()
    total = 0.0 # total loss on eval data

    for _ in range(num_batches):
        inputs, targets = data_loader(val_data, batch_size, context_length, device)
        logits = model(inputs)
        total += cross_entropy(logits, targets).item()

    model.train()
    return total / num_batches # avg over num_batches


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float=1e-6):
    params = [p for p in parameters if p.grad is not None]

    # tensor.item() turn a single element tensor into float
    l2_norm = math.sqrt(sum((p.grad ** 2).sum().item() for p in params))
    if l2_norm > max_l2_norm:
        scale = max_l2_norm / (l2_norm + eps)
        for p in params:
            p.grad *= scale
    return l2_norm


def data_loader(x: np.ndarray, batch_size: int, context_length: int, device: str):
    """
    x: np.ndarray, uint16

    output: (
        inputs: torch.Tensor[batch_size, contect_length],
        target: torch.Tensor[batch_size, contect_length]
    )
    """
    inputs = []
    target = []

    for b in range(batch_size):
        i = random.randrange(len(x) - context_length)

        inputs.append(x[i : i + context_length])
        target.append(x[i + 1 : i + context_length + 1])

    return (torch.Tensor(inputs).to(device).to(torch.int64), 
            torch.Tensor(target).to(device).to(torch.int64))


"""
torch.nn.Module
    .state_dict() -> dict[str, Tensor]
    .load_state_dict() -> None

torch.save(obj, file)
torch.load(src)
"""
def save_checkpoint(model: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    iteration: int, 
                    out: str | os.PathLike | BinaryIO | IO[bytes]):
    
    model_state = model.state_dict()
    opt_state = optimizer.state_dict()

    states = {
        "iter": iteration,
        "model": model_state,
        "opt": opt_state
    }
    torch.save(states, out)
    return

def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes],
                    model: torch.nn.Module=None, 
                    optimizer: torch.optim.Optimizer=None):
    states = torch.load(src)
    if model is not None:
        model.load_state_dict(states["model"])
    if optimizer is not None:
        optimizer.load_state_dict(states["opt"])
    return states["iter"] if "iter" in states else None

