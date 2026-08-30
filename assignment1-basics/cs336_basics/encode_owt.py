"""Encode a corpus into a uint16 .npy array of token IDs, in parallel.

The single-process path in tests/tests_encode_bf16.py is fine for TinyStories
but would take hours on OpenWebText's 11.9 GB train split, so this splits the
file on <|endoftext|> boundaries and encodes the chunks across a process pool.

Run from the repo root:
    python cs336_basics/encode_owt.py --split valid
    python cs336_basics/encode_owt.py --split train
"""

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.tokenizer import Tokenizer

DATA = ROOT / "data"
DELIM = "<|endoftext|>"
CHUNK_BYTES = 32 << 20  # target text bytes per chunk; keeps worker memory bounded

# output names match PATH / VAL_PATH in cs336_basics/train.py
SPLITS = {
    "train": (DATA / "owt_train.txt", DATA / "owt_train.npy"),
    "valid": (DATA / "owt_valid.txt", DATA / "owt_val.npy"),
}

_TOK = None  # one Tokenizer per worker process, built once in the initializer


def _init_worker(vocab_path: str, merges_path: str) -> None:
    global _TOK
    _TOK = Tokenizer.from_files(vocab_path, merges_path, special_tokens=[DELIM])


def _encode_chunk(job):
    idx, start, end, in_path, tmp_dir = job
    with open(in_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    ids = np.asarray(_TOK.encode(text), dtype=np.uint16)
    ids.tofile(Path(tmp_dir) / f"part_{idx:05d}.bin")
    return idx, ids.size, end - start


def encode_split(in_path: Path, out_path: Path, num_workers: int) -> None:
    tmp_dir = out_path.parent / f".{out_path.stem}_parts"
    tmp_dir.mkdir(exist_ok=True)

    with open(in_path, "rb") as f:
        n_chunks = max(1, in_path.stat().st_size // CHUNK_BYTES)
        boundaries = find_chunk_boundaries(f, n_chunks, DELIM.encode("utf-8"))

    jobs = [
        (i, s, e, str(in_path), str(tmp_dir))
        for i, (s, e) in enumerate(zip(boundaries[:-1], boundaries[1:]))
    ]
    total_bytes = in_path.stat().st_size
    print(f"{in_path.name}: {total_bytes:,} bytes -> {len(jobs)} chunks on {num_workers} workers")

    sizes = {}
    done_bytes = 0
    t0 = time.time()
    with Pool(
        num_workers,
        initializer=_init_worker,
        initargs=(str(ROOT / "owt_vocab.pkl"), str(ROOT / "owt_merges.pkl")),
    ) as pool:
        for n, (idx, size, nbytes) in enumerate(pool.imap_unordered(_encode_chunk, jobs), 1):
            sizes[idx] = size
            done_bytes += nbytes
            if n % 10 == 0 or n == len(jobs):
                dt = time.time() - t0
                rate = done_bytes / dt
                eta = (total_bytes - done_bytes) / rate
                print(
                    f"  {n}/{len(jobs)} chunks | {done_bytes / 1e9:.2f}/{total_bytes / 1e9:.2f} GB"
                    f" | {rate / 1e6:.1f} MB/s | eta {eta / 60:.1f} min",
                    flush=True,
                )

    # stitch the parts into one .npy, in chunk order, without holding it all in RAM
    total_tokens = sum(sizes.values())
    arr = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint16, shape=(total_tokens,))
    off = 0
    for idx in range(len(jobs)):
        part = np.fromfile(tmp_dir / f"part_{idx:05d}.bin", dtype=np.uint16)
        arr[off : off + part.size] = part
        off += part.size
    arr.flush()
    del arr

    for p in tmp_dir.glob("part_*.bin"):
        p.unlink()
    tmp_dir.rmdir()

    print(
        f"{in_path.name}: {total_tokens:,} tokens -> {out_path.name}"
        f" ({total_bytes / total_tokens:.3f} bytes/token, {time.time() - t0:.0f}s)"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "valid", "both"], default="both")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    splits = ["valid", "train"] if args.split == "both" else [args.split]
    for s in splits:
        in_path, out_path = SPLITS[s]
        encode_split(in_path, out_path, args.workers)


if __name__ == "__main__":
    main()
