from __future__ import annotations

import json
import os
import resource
import sys
import time
import pickle

import psutil
import pytest
import tiktoken

from cs336_basics.tokenizer import Tokenizer, train_bpe_parallel
from cs336_basics.pretokenization_example import find_chunk_boundaries


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    vocab, merges = train_bpe_parallel(input_path, vocab_size, special_tokens)
    return (vocab, merges)


def train_and_save(
    input_path: str | os.PathLike,
    vocab_size: int,
    name: str,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a BPE tokenizer, serialize it to disk, and print benchmarks."""
    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]

    start = time.perf_counter()
    vocab, merges = run_train_bpe(input_path, vocab_size, special_tokens)
    elapsed = time.perf_counter() - start

    # Peak memory. ru_maxrss is in KB on Linux, bytes on macOS.
    # Workers are separate processes, so include RUSAGE_CHILDREN.
    peak_kb = max(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    )
    peak_gb = peak_kb / (1024 ** 2)  # Linux (KB -> GB)

    # Serialize vocab and merges to disk (bytes-safe via pickle).
    with open(f"{name}_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open(f"{name}_merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    # Longest token in the vocabulary.
    longest = max(vocab.values(), key=len)

    print(f"=== {name} ===")
    print(f"training time: {elapsed:.1f} s")
    print(f"peak memory:   {peak_gb:.2f} GB")
    print(f"vocab size:    {len(vocab)}")
    print(f"longest token: {longest!r} ({len(longest)} bytes)")
    print()

    return vocab, merges


if __name__ == "__main__":
    # Run from the repo root so these resolve:
    #   uv run python tests/train_bpe_tinystories.py
    TinyStoryPath = "data/TinyStoriesV2-GPT4-train.txt"
    OpenWebPath = "data/owt_train.txt"
    special_tokens = ["<|endoftext|>"]

    # train_and_save(TinyStoryPath, 10000, "tinystories", special_tokens)
    train_and_save(OpenWebPath, 32000, "owt", special_tokens)