"""13-gram overlap between your training data and eval benchmarks (token-level).

  # on tokenized shards (fast, numpy):
  python scripts/decontam.py --shards-or-parquet data/fineweb_edu_10B_gpt2 --tokenizer gpt2 \
      --benchmarks hellaswag,arc_easy,arc_challenge,piqa --out decontam_gpt2.json
  # on raw parquet, and write the flagged documents so tokenize_shards.py --exclude skips them:
  python scripts/decontam.py --shards-or-parquet data/raw/fineweb_edu_100BT --tokenizer tokenizers/own32k \
      --max-docs 5000000 --drop data/decontam_drop.json

Benchmarks are downloaded with `datasets` (HellaSwag val, ARC test, PIQA val: the splits
lm-eval scores). --benchmark-file FILE.jsonl ({"benchmark": ..., "text": ...} per line) adds
or replaces them offline. Report: per benchmark, how many items share >=1 13-gram with the
training data, plus the flagged training documents.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

_MULT = np.random.default_rng(13).integers(1, 2 ** 63, size=64, dtype=np.uint64) | np.uint64(1)


def ngram_hashes(tokens: np.ndarray, n: int) -> np.ndarray:
    """Hash of every length-n window (uint64, wrapping arithmetic), one array op per offset."""
    tokens = np.asarray(tokens, dtype=np.uint64)
    L = tokens.size - n + 1
    if L <= 0:
        return np.empty(0, dtype=np.uint64)
    h = np.zeros(L, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(n):
            h += tokens[j:j + L] * _MULT[j]
    return h


def load_benchmarks(names: list[str]) -> dict[str, list[str]]:
    from datasets import load_dataset
    out = {}
    for name in names:
        if name == "hellaswag":
            ds = load_dataset("Rowan/hellaswag", split="validation")
            out[name] = [r["ctx"] + " " + r["endings"][int(r["label"])] for r in ds]
        elif name in ("arc_easy", "arc_challenge"):
            cfg = "ARC-Easy" if name == "arc_easy" else "ARC-Challenge"
            ds = load_dataset("allenai/ai2_arc", cfg, split="test")
            out[name] = [r["question"] + " " + r["choices"]["text"][r["choices"]["label"].index(r["answerKey"])]
                         for r in ds]
        elif name == "piqa":
            ds = load_dataset("baber/piqa", split="validation")  # the copy lm-eval uses
            out[name] = [r["goal"] + " " + (r["sol1"] if int(r["label"]) == 0 else r["sol2"]) for r in ds]
        else:
            raise ValueError(f"unknown benchmark {name!r}; use --benchmark-file for custom sets")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shards-or-parquet", required=True, help="shards dir (train_*.bin) or parquet file/dir")
    p.add_argument("--tokenizer", required=True, help="gpt2 | DIR (must match the shards)")
    p.add_argument("--benchmarks", default="hellaswag,arc_easy,arc_challenge,piqa")
    p.add_argument("--benchmark-file", default=None, help="JSONL with benchmark/text fields (offline)")
    p.add_argument("--n", type=int, default=13)
    p.add_argument("--split", default="train", help="shards split to scan")
    p.add_argument("--max-docs", type=int, default=0, help="parquet mode: scan only the first N docs")
    p.add_argument("--chunk-tokens", type=int, default=20_000_000)
    p.add_argument("--out", default="decontam_report.json")
    p.add_argument("--drop", default=None, help="write flagged documents here (parquet mode: [[file,row],...])")
    args = p.parse_args(argv)
    assert args.n <= _MULT.size

    from mygpt.data import read_shard
    from mygpt.tokenizer import get_tokenizer

    tok = get_tokenizer(args.tokenizer)
    if args.benchmark_file:
        bench: dict[str, list[str]] = {}
        with open(args.benchmark_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    bench.setdefault(r["benchmark"], []).append(r["text"])
    else:
        bench = load_benchmarks([b for b in args.benchmarks.split(",") if b])

    # hash -> list of (benchmark, item index)
    owners: dict[int, list[tuple[str, int]]] = {}
    for name, texts in bench.items():
        for i, text in enumerate(texts):
            for h in set(ngram_hashes(np.array(tok.encode(text)), args.n).tolist()):
                owners.setdefault(h, []).append((name, i))
    table = np.array(sorted(owners), dtype=np.uint64)
    print(f"{sum(len(v) for v in bench.values())} benchmark items -> {table.size:,} distinct {args.n}-grams")

    hit_items: dict[str, set[int]] = {name: set() for name in bench}
    flagged: list = []
    scanned = 0

    def in_table(h: np.ndarray) -> np.ndarray:
        """Membership test against the sorted table (searchsorted is ~5x faster than np.isin)."""
        if table.size == 0:
            return np.zeros(h.shape, dtype=bool)
        i = np.searchsorted(table, h)
        i[i == table.size] = 0
        return table[i] == h

    def record(hashes: np.ndarray) -> bool:
        hits = np.unique(hashes[in_table(hashes)])
        for h in hits.tolist():
            for name, i in owners[h]:
                hit_items[name].add(i)
        return hits.size > 0

    src = args.shards_or_parquet
    shard_files = sorted(glob.glob(os.path.join(src, f"{args.split}_*.bin"))) if os.path.isdir(src) else []
    if shard_files:
        for path in shard_files:
            toks = read_shard(path)
            eots = np.flatnonzero(np.asarray(toks) == tok.eot_id)
            for start in range(0, toks.size, args.chunk_tokens):
                chunk = np.asarray(toks[start:start + args.chunk_tokens + args.n - 1])
                h = ngram_hashes(chunk, args.n)
                mask = in_table(h)
                if mask.any():
                    record(h[mask])
                    pos = np.flatnonzero(mask) + start
                    docs = np.unique(np.searchsorted(eots, pos, side="right") - 1)
                    for d in docs.tolist():
                        s = int(eots[d]) if d >= 0 else 0
                        e = int(eots[d + 1]) if d + 1 < eots.size else int(toks.size)
                        flagged.append([os.path.basename(path), s, e])
            scanned += int(toks.size)
            print(f"{os.path.basename(path)}: {scanned:,} tokens scanned, {len(flagged)} docs flagged", flush=True)
        flagged = [list(x) for x in sorted({tuple(f) for f in flagged})]
    else:
        from _common import iter_parquet_texts
        for file, row, text in iter_parquet_texts(src, max_docs=args.max_docs):
            ids = np.array(tok.encode(text))
            scanned += ids.size
            if record(ngram_hashes(ids, args.n)):
                flagged.append([os.path.basename(file), row])

    report = {"n": args.n, "tokenizer": args.tokenizer, "source": os.path.abspath(src),
              "mode": "shards" if shard_files else "parquet", "tokens_scanned": scanned,
              "flagged_docs": len(flagged), "benchmarks": {}}
    for name, texts in bench.items():
        k = len(hit_items[name])
        report["benchmarks"][name] = {"items": len(texts), "contaminated_items": k,
                                      "contaminated_frac": round(k / max(1, len(texts)), 5),
                                      "examples": [texts[i][:200] for i in sorted(hit_items[name])[:5]]}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    if args.drop:
        with open(args.drop, "w", encoding="utf-8") as f:
            json.dump(flagged, f)
    for name, r in report["benchmarks"].items():
        print(f"{name:14s} {r['contaminated_items']:>6}/{r['items']:<6} items overlap ({100 * r['contaminated_frac']:.2f}%)")
    print(f"{len(flagged)} training documents flagged -> {args.out}" + (f", drop list {args.drop}" if args.drop else ""))
    return report


if __name__ == "__main__":
    main()
