"""Minimal inference / text-generation framework for the CS336 assignment-1 model.

Reuses the hand-written modules from `cs336_basics` (Transformer_LM, Tokenizer, softmax,
cross_entropy) — nothing is re-implemented. Loads a trained checkpoint, rebuilds the model
by inferring its architecture from the checkpoint tensor shapes, and samples text with
temperature + top-p (nucleus) decoding.

Usage (from anywhere; venv active):
    python inference/generate.py --prompt "Once upon a time"
    python inference/generate.py --prompt "Once upon a time" --num-heads 16 --temperature 0
"""
import argparse
import os
import sys

import numpy as np
import torch

# Make `cs336_basics` importable regardless of CWD / whether it was pip-installed.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cs336_basics.model import Transformer_LM, softmax
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.utils import cross_entropy

EOT = "<|endoftext|>"


def load_state_dict(ckpt_path, device):
    """Load just the model weights from a checkpoint (dict-wrapped or raw state_dict)."""
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    return obj["model"] if isinstance(obj, dict) and "model" in obj else obj


def infer_arch(sd):
    """Recover (vocab_size, d_model, num_layers, d_ff) from checkpoint tensor shapes."""
    vocab_size, d_model = sd["token_embedding.embedding_matrix"].shape
    num_layers = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("layers."))
    d_ff = sd["layers.0.ffn.w1.W"].shape[0]
    return int(vocab_size), int(d_model), int(num_layers), int(d_ff)


def build_model(sd, num_heads, context_length, rope_theta, device):
    """Rebuild Transformer_LM from a state dict and load the weights. Returns (model, arch)."""
    vocab_size, d_model, num_layers, d_ff = infer_arch(sd)
    model = Transformer_LM(
        d_model=d_model,
        num_head=num_heads,
        d_ff=d_ff,
        vocab_size=vocab_size,
        num_layers=num_layers,
        theta=rope_theta,
        max_seq_len=context_length,
    ).to(device)
    model.load_state_dict(sd)  # RoPE cos/sin buffers are non-persistent -> rebuilt by ctor
    model.eval()
    return model, (vocab_size, d_model, num_layers, d_ff)


def head_candidates(d_model):
    """Head counts that evenly divide d_model AND give an even head dim (required by RoPE)."""
    return [h for h in range(1, d_model + 1) if d_model % h == 0 and (d_model // h) % 2 == 0]


@torch.no_grad()
def detect_num_heads(sd, context_length, rope_theta, device, val_path, n_windows=8, seed=0):
    """num_heads isn't stored in the checkpoint. Recover it by scoring each valid head count
    against a batch of validation tokens: the true value has a much lower cross-entropy."""
    _, d_model, _, _ = infer_arch(sd)
    data = np.load(val_path, mmap_mode="r")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(data) - context_length - 1, size=n_windows)
    batch = np.stack([data[s : s + context_length + 1].astype(np.int64) for s in starts])
    batch = torch.from_numpy(batch).to(device)
    inputs, targets = batch[:, :-1], batch[:, 1:]

    best_h, best_loss, scores = None, None, []
    for h in head_candidates(d_model):
        model, _ = build_model(sd, h, context_length, rope_theta, device)
        loss = cross_entropy(model(inputs), targets).item()
        scores.append((h, loss))
        if best_loss is None or loss < best_loss:
            best_h, best_loss = h, loss

    print("[detect] num_heads sweep (lower loss = correct):")
    for h, loss in scores:
        print(f"    num_heads={h:<4d} val_loss={loss:.4f}" + ("   <- selected" if h == best_h else ""))
    return best_h


def sample_next(logits, temperature, top_p):
    """Sample one token id from last-position logits [vocab] with temperature + top-p."""
    if temperature == 0:
        return int(logits.argmax())

    probs = softmax(logits / temperature, -1)

    if 0 < top_p < 1:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        # keep the smallest prefix whose cumulative prob first reaches top_p (>= 1 token)
        keep = (cumsum - sorted_probs) < top_p
        sorted_probs = sorted_probs * keep
        sorted_probs /= sorted_probs.sum()
        choice = torch.multinomial(sorted_probs, num_samples=1)
        return int(sorted_idx[choice])

    return int(torch.multinomial(probs, num_samples=1))


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens, temperature, top_p, context_length, eot_id, device):
    """Autoregressive decoding (full recompute each step; no KV cache). Returns generated text."""
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    new_ids = []
    for _ in range(max_new_tokens):
        cropped = ids[:, -context_length:]  # RoPE is indexed by absolute position -> must crop
        logits = model(cropped)[0, -1]
        next_id = sample_next(logits, temperature, top_p)
        if next_id == eot_id:
            break
        new_ids.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
    return tokenizer.decode(new_ids)


def main():
    p = argparse.ArgumentParser(description="Generate text from a trained CS336 assignment-1 model.")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=os.path.join(REPO_ROOT, "checkpoints", "ckpt_4000.pt"))
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8, help="0 = greedy/argmax")
    p.add_argument("--top-p", type=float, default=0.95, help="nucleus threshold; 1.0 disables")
    p.add_argument("--num-heads", type=int, default=None, help="omit to auto-detect from val loss")
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--rope-theta", type=float, default=10000.0)
    p.add_argument("--vocab-file", type=str, default=None, help="override; else auto by vocab_size")
    p.add_argument("--merges-file", type=str, default=None)
    p.add_argument("--val-path", type=str, default=os.path.join(REPO_ROOT, "data", "tinystories_valid.npy"),
                   help="validation tokens used only for --num-heads auto-detection")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    sd = load_state_dict(args.checkpoint, args.device)
    vocab_size = infer_arch(sd)[0]

    num_heads = args.num_heads
    if num_heads is None:
        num_heads = detect_num_heads(sd, args.context_length, args.rope_theta, args.device, args.val_path)

    model, (vocab_size, d_model, num_layers, d_ff) = build_model(
        sd, num_heads, args.context_length, args.rope_theta, args.device
    )

    # Pick the tokenizer that matches the model's vocab size (unless overridden).
    if args.vocab_file and args.merges_file:
        vocab_file, merges_file = args.vocab_file, args.merges_file
    else:
        stem = "tinystories" if vocab_size == 10000 else "owt"
        vocab_file = os.path.join(REPO_ROOT, f"{stem}_vocab.pkl")
        merges_file = os.path.join(REPO_ROOT, f"{stem}_merges.pkl")
    tokenizer = Tokenizer.from_files(vocab_file, merges_file, special_tokens=[EOT])
    eot_id = tokenizer.rev_vocab[EOT.encode("utf-8")]

    print(f"[config] device={args.device} vocab_size={vocab_size} d_model={d_model} "
          f"num_layers={num_layers} num_heads={num_heads} d_ff={d_ff} "
          f"context_length={args.context_length} rope_theta={args.rope_theta}")
    print(f"[config] checkpoint={args.checkpoint}")
    print(f"[config] tokenizer={os.path.basename(vocab_file)}, {os.path.basename(merges_file)}")
    print(f"[config] temperature={args.temperature} top_p={args.top_p} max_new_tokens={args.max_new_tokens}\n")

    text = generate(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature,
                    args.top_p, args.context_length, eot_id, args.device)
    print(args.prompt + text)


if __name__ == "__main__":
    main()
