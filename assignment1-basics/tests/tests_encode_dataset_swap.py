"""tokenizer_experiments (b): tokenize the OpenWebText sample with the
*TinyStories* tokenizer and compare its compression ratio to the OWT
tokenizer's on the same documents.

Run from the repo root:
    uv run python tests/tests_encode_dataset_swap.py
"""

from cs336_basics.tokenizer import Tokenizer

# reuse the helpers/constants from the part (a) script (same tests/ dir on sys.path)
from tests_encode_dataset import ROOT, DELIM, sample_documents, compression_ratio


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

    # same OWT sample as part (a) (fixed seed), tokenized two ways
    owt_docs = sample_documents(ROOT / "data" / "owt_valid.txt")

    native = compression_ratio(owt_tok, owt_docs)
    swapped = compression_ratio(ts_tok, owt_docs)

    print(f"OWT sample, OWT tokenizer (native):       {native:.3f} bytes/token")
    print(f"OWT sample, TinyStories tokenizer (swap): {swapped:.3f} bytes/token")


if __name__ == "__main__":
    main()
