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


vocab_TinyStory = "tinystories_vocab.pkl"
merges_TinyStory = "tinystories_merges.pkl"

vocab_OWT = ""
merges_OWT = ""

TinyStoryPath = "data/TinyStoriesV2-GPT4-train.txt"
OpenWebPath = "data/owt_train.txt"

import random

def sample_docs(path, n=10, chunk_bytes=10 * 1024 * 1024, seed=0):
    size = os.path.getsize(path)
    rng = random.Random(seed)
    docs = []
    while len(docs) < n:
        off = rng.randrange(0, max(1, size - chunk_bytes))
        with open(path, "rb") as f:
            f.seek(off)
            text = f.read(chunk_bytes).decode("utf-8", errors="ignore")
        parts = text.split("<|endoftext|>")[1:-1]
        docs.extend(d for d in parts if d.strip())
    return docs[:n]

def run_tokenizer(vocab_path, merges_path, doc_path, special_tokens=None):
    if special_tokens is None:
        special_tokens = ["<|endoftext|>"] 

    T = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
    docs: list[str] = sample_docs(doc_path, 10)

    nbytes = sum(len(d.encode("utf-8")) for d in docs)

    start = time.perf_counter()
    ntoken = sum(len(T.encode(d)) for d in docs)
    elapsed = time.perf_counter() - start

    ratio = nbytes / ntoken
    throughput = nbytes / elapsed          # bytes/s
    pile_hours = 825 * 1024**3 / throughput / 3600

    print(f"ratio {ratio:.2f} B/tok | {throughput/1e6:.2f} MB/s | "
          f"Pile ≈ {pile_hours:.0f} h")
    return ratio, throughput


if __name__ == "__main__":
    # a
    print("Compression ratio")
    print("TinyStories:", run_tokenizer(vocab_TinyStory, merges_TinyStory, TinyStoryPath))
    print("OWT:", run_tokenizer(vocab_OWT, merges_OWT, OpenWebPath))

    # b
    print("OWT docs w/ TS tokenizer:", run_tokenizer(vocab_TinyStory, merges_TinyStory, OpenWebPath))