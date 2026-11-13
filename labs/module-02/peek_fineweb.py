"""Stream a few thousand FineWeb-Edu documents (no full download) and measure them.

    python peek_fineweb.py --docs 2000 --show 3

Prints: edu score mix, average document length, UTF-8 bytes per GPT-2 token, and how many
parquet files of sample/100BT you need for a token target. Needs `datasets` and `tiktoken`
(both installed with the project). Uses HF_TOKEN from the environment if set.
"""

from __future__ import annotations

import argparse
import collections
import math

import tiktoken
from datasets import load_dataset

# Measured from the Hub file listing on 2026-11-13: sample/100BT = 140 files, 286.39 GB parquet,
# ~100B GPT-2 tokens. Typical full file: ~2.15 GB.
GB_PER_B_TOKENS = 2.864
FILE_GB = 2.15


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="sample-10BT")
    p.add_argument("--docs", type=int, default=2000)
    p.add_argument("--show", type=int, default=2, help="print the start of N documents")
    p.add_argument("--target-tokens", type=float, default=20e9)
    a = p.parse_args()

    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name=a.config, split="train", streaming=True)

    scores = collections.Counter()
    n_bytes = n_tokens = n_chars = 0
    for i, row in enumerate(ds):
        if i >= a.docs:
            break
        text = row["text"]
        if i < a.show:
            print(f"--- doc {i} score={row['score']:.2f} url={row['url']}\n{text[:400]}\n")
        scores[row["int_score"]] += 1
        n_chars += len(text)
        n_bytes += len(text.encode("utf-8"))
        n_tokens += len(enc.encode_ordinary(text)) + 1  # +1 for the end-of-text separator

    n = sum(scores.values())
    print(f"documents sampled      : {n}")
    print("int_score mix          : " + ", ".join(f"{k}: {v / n:.0%}" for k, v in sorted(scores.items())))
    print(f"avg chars per document : {n_chars / n:,.0f}")
    print(f"avg GPT-2 tokens / doc : {n_tokens / n:,.0f}")
    print(f"UTF-8 bytes per token  : {n_bytes / n_tokens:.2f}")
    gb = a.target_tokens / 1e9 * GB_PER_B_TOKENS
    files = math.ceil(gb / FILE_GB)
    print(f"for {a.target_tokens / 1e9:.0f}B GPT-2 tokens: ~{gb:.0f} GB parquet = ~{files} files of sample/100BT")


if __name__ == "__main__":
    main()
