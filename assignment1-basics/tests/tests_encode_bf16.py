"""tokenizer_experiments (d): encode the train/valid splits into token IDs and
serialize each as a uint16 NumPy array (.npy) in the data/ folder.

NOTE: dtype is uint16 (token IDs are integers, 0..vocab_size-1), NOT bfloat16.

Run from the repo root:
    uv run python tests/tests_encode_bf16.py
"""

from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DELIM = "<|endoftext|>"


def encode_file_to_npy(tokenizer: Tokenizer, in_path: Path, out_path: Path) -> None:
    """Stream-encode a corpus and save the token IDs as a uint16 .npy array."""
    with open(in_path, "r", encoding="utf-8") as f:
        # encode_iterable streams line by line, so we never hold the whole file
        # (or a giant Python list) in memory; np.fromiter grows the array directly.
        arr = np.fromiter(tokenizer.encode_iterable(f), dtype=np.uint16)
    np.save(out_path, arr)
    print(f"{in_path.name}: {arr.size:,} tokens -> {out_path.name}")


def main() -> None:
    ts_tok = Tokenizer.from_files(
        str(ROOT / "tinystories_vocab.pkl"),
        str(ROOT / "tinystories_merges.pkl"),
        special_tokens=[DELIM],
    )
    owt_tok = Tokenizer.from_files(
        str(ROOT / "owt_vocab.pkl"),
        str(ROOT / "owt_merges.pkl"),
        special_tokens=[DELIM],
    )

    jobs = [
        (ts_tok, DATA / "TinyStoriesV2-GPT4-valid.txt", DATA / "tinystories_valid.npy"),
        (ts_tok, DATA / "TinyStoriesV2-GPT4-train.txt", DATA / "tinystories_train.npy"),
        (owt_tok, DATA / "owt_valid.txt", DATA / "owt_val.npy"),
        (owt_tok, DATA / "owt_train.txt", DATA / "owt_train.npy"),
    ]
    for tok, in_path, out_path in jobs:
        encode_file_to_npy(tok, in_path, out_path)


if __name__ == "__main__":
    main()
