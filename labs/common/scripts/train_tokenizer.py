"""Train your own byte-level BPE tokenizer on FineWeb-Edu parquet.

  python scripts/train_tokenizer.py --input data/raw/fineweb_edu_100BT --vocab-size 32768 \
      --max-docs 2000000 --out tokenizers/own32k

Writes OUT/tokenizer.json (load with mygpt.tokenizer.get_tokenizer(OUT)). ~2M documents
(~2-3B characters) is plenty for a 32k vocab; RAM use grows with --max-docs (budget ~30-60 GB
for 2M docs on a CPU box).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="parquet file or directory")
    p.add_argument("--vocab-size", type=int, default=32768)
    p.add_argument("--max-docs", type=int, default=2_000_000)
    p.add_argument("--min-frequency", type=int, default=2)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    from _common import iter_parquet_texts
    from mygpt.tokenizer import get_tokenizer, train_bpe

    t0 = time.time()
    texts = (text for _, _, text in iter_parquet_texts(args.input, max_docs=args.max_docs))
    path = train_bpe(texts, args.vocab_size, args.out, args.min_frequency)
    tok = get_tokenizer(args.out)
    sample = "The mitochondria is the powerhouse of the cell. 3.14159 réseau 日本語"
    ids = tok.encode(sample)
    assert tok.decode(ids) == sample, "round-trip failed"
    print(f"wrote {path} | vocab {tok.vocab_size} | eot_id {tok.eot_id} | {time.time() - t0:.0f}s")
    print(f"sample: {len(sample.encode())} bytes -> {len(ids)} tokens")
    return path


if __name__ == "__main__":
    main()
