"""Summarize a shards directory: tokens per split, bytes/token, shard sizes, a decoded sample.

  python scripts/token_report.py --shards data/fineweb_edu_10B_gpt2 --tokenizer gpt2
Writes SHARDS/token_report.json (or --out FILE).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shards", required=True)
    p.add_argument("--tokenizer", default=None, help="gpt2 | DIR (default: from meta.json)")
    p.add_argument("--out", default=None)
    p.add_argument("--sample-tokens", type=int, default=64)
    args = p.parse_args(argv)

    from mygpt.data import read_header, read_meta, read_shard

    meta = read_meta(args.shards)
    report = {"shards_dir": os.path.abspath(args.shards), "meta": meta, "splits": {}}
    for split in ("train", "val"):
        paths = sorted(glob.glob(os.path.join(args.shards, f"{split}_*.bin")))
        counts = [read_header(pth)["n_tokens"] for pth in paths]
        report["splits"][split] = {"shards": len(paths), "tokens": sum(counts),
                                   "min_shard": min(counts, default=0), "max_shard": max(counts, default=0),
                                   "gb_on_disk": round(sum(os.path.getsize(pth) for pth in paths) / 1e9, 3)}
        nbytes = meta.get(f"{split}_bytes")
        if nbytes and sum(counts):
            report["splits"][split]["bytes_per_token"] = round(nbytes / sum(counts), 4)
    total = sum(s["tokens"] for s in report["splits"].values())
    report["total_tokens"] = total
    if meta.get("total_bytes"):
        report["bytes_per_token"] = round(meta["total_bytes"] / max(total, 1), 4)

    from mygpt.data import resolve_tokenizer
    tok_name = args.tokenizer or resolve_tokenizer(args.shards, meta)
    first = sorted(glob.glob(os.path.join(args.shards, "train_*.bin")))
    if tok_name and tok_name != "synthetic" and first:
        from mygpt.tokenizer import get_tokenizer
        tok = get_tokenizer(tok_name)
        ids = read_shard(first[0])[:args.sample_tokens].tolist()
        report["sample_ids"] = ids
        report["sample_text"] = tok.decode(ids)

    out = args.out or os.path.join(args.shards, "token_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    for split, s in report["splits"].items():
        print(f"{split:5s}: {s['tokens']:>15,} tokens in {s['shards']} shards ({s['gb_on_disk']} GB)"
              + (f", {s['bytes_per_token']} bytes/token" if "bytes_per_token" in s else ""))
    print(f"total: {total:,} tokens -> {out}")
    return report


if __name__ == "__main__":
    main()
