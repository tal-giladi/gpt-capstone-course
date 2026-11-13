"""Sanity-check a token shard directory before you trust it (lessons 03.2 and 03.3).

    python tools/shard_check.py --shards data/tok/bpe32k-100BT-30

Checks every *.bin shard against meta.json:
  - header magic/version/vocab size, and every token id < vocab size
  - token totals per split equal meta.json
  - number of end-of-text tokens equals the number of documents in meta.json
  - val_000000.bin and train_000000.bin start with end-of-text (a document boundary)
  - any other special token the tokenizer defines never occurs in the data (the course
    tokenizer has only <|endoftext|>, so usually there is nothing to check here)
Reads each shard once with numpy (about 1 s per 100M tokens). Exit code 1 on any problem.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np


def special_ids(shards: str, meta: dict) -> dict[str, int]:
    """Special tokens of the shards' tokenizer: GPT-2 has only <|endoftext|>."""
    if meta.get("tokenizer") == "gpt2":
        return {"<|endoftext|>": 50256}
    from mygpt.data import resolve_tokenizer
    tok = resolve_tokenizer(shards, meta)
    path = os.path.join(tok, "tokenizer.json") if tok and os.path.isdir(tok) else tok
    if not path or not os.path.exists(path):
        raise SystemExit(f"cannot find tokenizer.json for {shards} (meta tokenizer={meta.get('tokenizer')})")
    with open(path, encoding="utf-8") as f:
        return {t["content"]: t["id"] for t in json.load(f).get("added_tokens", []) if t.get("special", True)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shards", required=True)
    args = p.parse_args(argv)

    from mygpt.data import read_header, read_meta, read_shard

    meta = read_meta(args.shards)
    if not meta:
        print(f"FAIL no meta.json in {args.shards}")
        return 1
    vocab, eot = int(meta["vocab_size"]), int(meta["eot_id"])
    specials = special_ids(args.shards, meta)
    others = {s: i for s, i in specials.items() if i != eot}
    problems: list[str] = []
    totals = {}
    for split in ("val", "train"):
        paths = sorted(glob.glob(os.path.join(args.shards, f"{split}_*.bin")))
        if not paths:
            if split == "train" and meta.get("train_tokens", 1) == 0:
                print("  train: none (val-only directory)")
            else:
                problems.append(f"no {split}_*.bin shards")
            continue
        n_tok = n_eot = 0
        other_counts = {s: 0 for s in others}
        for i, path in enumerate(paths):
            h = read_header(path)
            if h["vocab_size"] != vocab:
                problems.append(f"{os.path.basename(path)}: header vocab {h['vocab_size']} != meta {vocab}")
            t = np.asarray(read_shard(path))
            if t.size and int(t.max()) >= vocab:
                problems.append(f"{os.path.basename(path)}: token id {int(t.max())} >= vocab {vocab}")
            counts = np.bincount(t, minlength=max(vocab, 1))
            n_tok += t.size
            n_eot += int(counts[eot])
            for s, sid in others.items():
                other_counts[s] += int(counts[sid])
            if i == 0 and (t.size == 0 or int(t[0]) != eot):
                problems.append(f"{os.path.basename(path)} does not start with end-of-text")
            print(f"\r{split}: {i + 1}/{len(paths)} shards, {n_tok:,} tokens", end="", flush=True)
        print()
        totals[split] = n_tok
        if n_tok != meta.get(f"{split}_tokens", n_tok):
            problems.append(f"{split}: {n_tok:,} tokens in shards != meta {meta[f'{split}_tokens']:,}")
        docs = meta.get(f"{split}_docs")
        if docs is not None and n_eot != docs:
            problems.append(f"{split}: {n_eot:,} end-of-text tokens != {docs:,} documents in meta "
                            "(literal special strings in the text, or a broken shard)")
        for s, c in other_counts.items():
            if c:
                problems.append(f"{split}: special token {s} occurs {c:,} times (must be 0 in pretraining data)")
        print(f"  {split}: {len(paths)} shards, {n_tok:,} tokens, {n_eot:,} documents"
              + (f", {meta[f'{split}_bytes'] / n_tok:.3f} bytes/token" if meta.get(f"{split}_bytes") and n_tok else ""))
    print(f"  vocab {vocab:,}, eot id {eot}, other specials checked: {', '.join(others) or 'none'}")
    for msg in problems:
        print(f"FAIL {msg}")
    print("OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
