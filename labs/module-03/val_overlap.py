"""Check that validation documents are really held out, and build a shared validation set
(lesson 03.3).

    python tools/val_overlap.py \
        --main-raw data/raw/sample-100BT --main-shards data/tok/bpe32k-100BT-30 \
        --base-raw data/raw/sample-10BT  --base-shards data/tok/gpt2-10BT \
        --out reports/val_overlap.json --write-common-val data/common_val/common_val.parquet

tokenize_shards.py puts the first documents of each input (until --val-tokens) into val_*.bin
and everything after into train_*.bin. The two shard sets come from different FineWeb-Edu
samples (sample-10BT and the first 30 files of sample-100BT), which may share documents. This
script compares the documents' `id` column (FineWeb's unique <urn:uuid:...>) and reports:

  main val docs found anywhere in the baseline input  -> the baseline trains on them
  base val docs found anywhere in the main input      -> the main model trains on them

With --write-common-val it writes the main val documents that do NOT occur in the baseline
input to one parquet file (columns text, id). Tokenize it with both tokenizers to get one
validation text that neither model was trained on, for fair bits-per-byte comparisons.
Peak memory about 2 GB for ~30M document ids. No torch needed.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np


def parquet_files(path: str) -> list[str]:
    """Same order as scripts/_common.py (sorted, recursive), so row k here is document k there."""
    if os.path.isfile(path):
        return [path]
    files = sorted(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"no .parquet files under {path}")
    return files


def h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


def id_hashes(path: str, limit: int = 0) -> np.ndarray:
    import pyarrow.parquet as pq
    out: list[int] = []
    for f in parquet_files(path):
        for batch in pq.ParquetFile(f).iter_batches(columns=["id"], batch_size=65536):
            out.extend(h64(x) for x in batch.column(0).to_pylist())
            if limit and len(out) >= limit:
                return np.array(out[:limit], dtype=np.uint64)
    return np.array(out, dtype=np.uint64)


def member(x: np.ndarray, sorted_table: np.ndarray) -> np.ndarray:
    i = np.searchsorted(sorted_table, x)
    i[i == sorted_table.size] = 0
    return sorted_table[i] == x


def val_docs(shards: str) -> int:
    with open(os.path.join(shards, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    if "val_docs" not in meta:
        raise SystemExit(f"{shards}/meta.json has no val_docs")
    if meta.get("decontam", {}).get("val_changed"):
        raise SystemExit("val split was modified after tokenizing; row order no longer matches")
    return int(meta["val_docs"])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--main-raw", required=True)
    p.add_argument("--main-shards", required=True)
    p.add_argument("--base-raw", required=True)
    p.add_argument("--base-shards", required=True)
    p.add_argument("--out", default="val_overlap.json")
    p.add_argument("--write-common-val", default=None, help="parquet path for the shared val set")
    args = p.parse_args(argv)

    n_main_val, n_base_val = val_docs(args.main_shards), val_docs(args.base_shards)
    print("reading document ids (a few minutes for ~30M documents) ...", flush=True)
    main_all = id_hashes(args.main_raw)
    base_all = id_hashes(args.base_raw)
    main_val, base_val = main_all[:n_main_val], base_all[:n_base_val]
    main_sorted, base_sorted = np.sort(main_all), np.sort(base_all)

    main_val_in_base = member(main_val, base_sorted)
    base_val_in_main = member(base_val, main_sorted)
    shared_docs = int(member(base_all, main_sorted).sum())
    rep = {
        "main": {"raw": os.path.abspath(args.main_raw), "docs": int(main_all.size), "val_docs": n_main_val},
        "base": {"raw": os.path.abspath(args.base_raw), "docs": int(base_all.size), "val_docs": n_base_val},
        "docs_in_both_inputs": shared_docs,
        "main_val_docs_in_base_input": int(main_val_in_base.sum()),
        "base_val_docs_in_main_input": int(base_val_in_main.sum()),
        "duplicate_ids_main": int(main_all.size - np.unique(main_all).size),
        "duplicate_ids_base": int(base_all.size - np.unique(base_all).size),
    }
    print(f"main input: {main_all.size:,} docs ({n_main_val:,} val) | base input: {base_all.size:,} docs ({n_base_val:,} val)")
    print(f"documents present in both inputs:         {shared_docs:,}")
    print(f"main val docs the baseline trains on:     {rep['main_val_docs_in_base_input']:,} "
          f"({100 * main_val_in_base.mean():.2f}% of main val)")
    print(f"base val docs the main model trains on:   {rep['base_val_docs_in_main_input']:,} "
          f"({100 * base_val_in_main.mean():.2f}% of base val)")

    if args.write_common_val:
        import pyarrow as pa
        import pyarrow.parquet as pq
        keep = ~main_val_in_base
        texts, ids, row = [], [], 0
        for f in parquet_files(args.main_raw):
            for batch in pq.ParquetFile(f).iter_batches(columns=["text", "id"], batch_size=4096):
                for text, doc_id in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
                    if row >= n_main_val:
                        break
                    if keep[row]:
                        texts.append(text or "")
                        ids.append(doc_id)
                    row += 1
                if row >= n_main_val:
                    break
            if row >= n_main_val:
                break
        os.makedirs(os.path.dirname(os.path.abspath(args.write_common_val)), exist_ok=True)
        pq.write_table(pa.table({"text": texts, "id": ids}), args.write_common_val)
        nbytes = sum(len(t.encode("utf-8")) for t in texts)
        rep["common_val"] = {"path": os.path.abspath(args.write_common_val), "docs": len(texts), "bytes": nbytes}
        print(f"common val set: {len(texts):,} docs, {nbytes / 1e6:.1f} MB -> {args.write_common_val}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
