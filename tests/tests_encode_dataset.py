"""tokenizer_experiments (a): sample 10 documents from TinyStories and
OpenWebText, encode them with the matching trained tokenizer, and report the
compression ratio (bytes / token).

Run from the repo root:
    uv run python tests/tests_encode_dataset.py
"""

import random
import time
from pathlib import Path

from cs336_basics.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
DELIM = "<|endoftext|>"
N_DOCS = 10
SEED = 0


def sample_documents(path: Path, n: int = N_DOCS, seed: int = SEED) -> list[str]:
    """Read a corpus (documents separated by <|endoftext|>) and return n random docs."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    docs = [d for d in text.split(DELIM) if d.strip()]
    random.seed(seed)  # fixed seed so the sample is reproducible
    return random.sample(docs, n)


def compression_ratio(tokenizer: Tokenizer, docs: list[str]) -> float:
    """Aggregate bytes/token over the sampled documents."""
    total_bytes = 0
    total_tokens = 0
    for d in docs:
        total_bytes += len(d.encode("utf-8"))
        total_tokens += len(tokenizer.encode(d))
    return total_bytes / total_tokens


def throughput(tokenizer: Tokenizer, docs: list[str]) -> float:
    """Bytes/second: time the encode of the sampled docs over their total byte size."""
    total_bytes = sum(len(d.encode("utf-8")) for d in docs)
    start = time.perf_counter()
    for d in docs:
        tokenizer.encode(d)
    elapsed = time.perf_counter() - start
    return total_bytes / elapsed


PILE_BYTES = 825 * 1e9  # 825 GB


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

    ts_docs = sample_documents(ROOT / "data" / "TinyStoriesV2-GPT4-valid.txt")
    owt_docs = sample_documents(ROOT / "data" / "owt_valid.txt")

    ts_ratio = compression_ratio(ts_tok, ts_docs)
    owt_ratio = compression_ratio(owt_tok, owt_docs)

    print(f"TinyStories tokenizer on TinyStories: {ts_ratio:.3f} bytes/token")
    print(f"OpenWebText tokenizer on OpenWebText: {owt_ratio:.3f} bytes/token")

    # (c) throughput + Pile estimate
    ts_bps = throughput(ts_tok, ts_docs)
    owt_bps = throughput(owt_tok, owt_docs)
    print(f"TinyStories tokenizer throughput: {ts_bps:,.0f} bytes/s "
          f"(Pile 825GB ~ {PILE_BYTES / ts_bps / 3600:.1f} h)")
    print(f"OpenWebText tokenizer throughput: {owt_bps:,.0f} bytes/s "
          f"(Pile 825GB ~ {PILE_BYTES / owt_bps / 3600:.1f} h)")


if __name__ == "__main__":
    main()
