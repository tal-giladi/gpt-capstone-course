"""Remove benchmark-contaminated documents from token shards, in place (lesson 03.3).

    python scripts/decontam.py --shards-or-parquet data/tok/gpt2-10BT --tokenizer gpt2 \
        --out reports/decontam-gpt2-10BT.json --drop reports/drop-gpt2-10BT.json
    python tools/filter_shards.py --shards data/tok/gpt2-10BT --drop reports/drop-gpt2-10BT.json

The drop list is what decontam.py writes in shard mode: [[shard_file, start, end], ...], token
ranges of whole documents (each starts at its end-of-text token). Every affected train shard is
rewritten without those ranges (atomically: write .tmp, then rename); untouched shards are not
rewritten. meta.json gets updated token/byte/document counts and a "decontam" record; the
original is kept as meta.pre_decontam.json. Refuses to run twice on the same directory, because
the ranges would no longer match. Validation shards are never changed.

A document that started in the previous shard shows up as a range starting at 0; only its part
in this shard is removed. That is rare and harmless.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict

import numpy as np


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shards", required=True)
    p.add_argument("--drop", required=True, help="drop list from decontam.py --drop (shard mode)")
    p.add_argument("--dry-run", action="store_true", help="only count what would be removed")
    args = p.parse_args(argv)

    from mygpt.data import read_meta, read_shard, resolve_tokenizer, write_meta, write_shard
    from mygpt.tokenizer import get_tokenizer

    meta = read_meta(args.shards)
    if not meta:
        raise SystemExit(f"no meta.json in {args.shards}")
    if "decontam" in meta:
        raise SystemExit(f"{args.shards} was already filtered ({meta['decontam']}); re-tokenize to start over")
    with open(args.drop, encoding="utf-8") as f:
        ranges = json.load(f)
    by_shard: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for item in ranges:
        if len(item) != 3:
            raise SystemExit("drop list is not in shard format [file, start, end]; run decontam.py on the shard dir")
        name, s, e = item
        if not name.startswith("train_"):
            raise SystemExit(f"refusing to modify non-train shard {name}")
        by_shard[name].append((int(s), int(e)))

    tok = get_tokenizer(resolve_tokenizer(args.shards, meta))
    eot = int(meta["eot_id"])
    drop_tokens = drop_bytes = drop_docs = 0
    for name in sorted(by_shard):
        path = os.path.join(args.shards, name)
        t = np.array(read_shard(path))  # a copy, so the file can be replaced below
        keep = np.ones(t.size, dtype=bool)
        for s, e in sorted(set(by_shard[name])):
            seg = t[s:e]
            drop_docs += int((seg == eot).sum())
            drop_bytes += len(tok.decode(seg[seg != eot].tolist()).encode("utf-8"))
            keep[s:e] = False
        removed = int(t.size - keep.sum())
        drop_tokens += removed
        print(f"{name}: -{removed:,} tokens in {len(by_shard[name])} range(s)")
        if not args.dry_run:
            write_shard(path, t[keep], int(meta["vocab_size"]))

    frac = drop_tokens / max(1, meta.get("train_tokens", 1))
    print(f"total: {drop_docs:,} documents, {drop_tokens:,} tokens ({100 * frac:.4f}% of train), "
          f"{drop_bytes / 1e6:.1f} MB of text")
    if args.dry_run:
        print("dry run: nothing written")
        return 0
    pre = os.path.join(args.shards, "meta.pre_decontam.json")
    if not os.path.exists(pre):
        shutil.copy(os.path.join(args.shards, "meta.json"), pre)
    for key, delta in (("train_tokens", drop_tokens), ("total_tokens", drop_tokens),
                       ("train_bytes", drop_bytes), ("total_bytes", drop_bytes), ("train_docs", drop_docs)):
        if key in meta:
            meta[key] -= delta
    meta["decontam"] = {"drop_list": os.path.basename(args.drop), "docs": drop_docs, "tokens": drop_tokens,
                        "bytes": drop_bytes, "shards_rewritten": sorted(by_shard), "val_changed": False}
    write_meta(args.shards, meta)
    print(f"updated {args.shards}/meta.json (original kept as meta.pre_decontam.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
