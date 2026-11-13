"""Small helpers shared by the scripts (kept out of the mygpt package on purpose)."""
from __future__ import annotations

import glob
import os
from typing import Iterator


def parse_value(text: str):
    """'3' -> 3, '1e-4' -> 1e-4, 'true'/'false' -> bool, 'none' -> None, anything else stays a
    string (so 'no' stays 'no', unlike YAML)."""
    low = text.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def parse_set(items: list[str]) -> dict:
    """['max_steps=3', 'save_strategy=no'] -> {'max_steps': 3, 'save_strategy': 'no'}"""
    out = {}
    for item in items:
        key, sep, val = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        out[key.strip()] = parse_value(val)
    return out


def parquet_files(path: str) -> list[str]:
    """All *.parquet under path (recursive), sorted; or [path] if it is a file."""
    if os.path.isfile(path):
        return [path]
    files = sorted(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no .parquet files under {path}")
    return files


def iter_parquet_texts(path: str, column: str = "text", max_docs: int = 0,
                       batch_size: int = 1024) -> Iterator[tuple[str, int, str]]:
    """Yield (file, row_index, text) without loading whole files into memory."""
    import pyarrow.parquet as pq
    n = 0
    for f in parquet_files(path):
        row = 0
        for batch in pq.ParquetFile(f).iter_batches(columns=[column], batch_size=batch_size):
            for text in batch.column(0).to_pylist():
                yield f, row, text or ""
                row += 1
                n += 1
                if max_docs and n >= max_docs:
                    return


# ---- shared by sft.py / dpo.py (no heavy imports) -------------------------------------------

def hf_checkpoints(out_dir: str) -> list[str]:
    """HF Trainer checkpoint dirs (OUT/checkpoint-N) sorted by step, oldest first."""
    if not os.path.isdir(out_dir):
        return []
    found = []
    for name in os.listdir(out_dir):
        path = os.path.join(out_dir, name)
        step = name[len("checkpoint-"):]
        if name.startswith("checkpoint-") and step.isdigit() and os.path.isdir(path):
            found.append((int(step), path))
    return [p for _, p in sorted(found)]


def resume_checkpoint(out_dir: str, resume: bool) -> str | None:
    """Path of the newest checkpoint to resume from, or None (fresh start). Prints what it does."""
    if not resume:
        return None
    ckpts = hf_checkpoints(out_dir)
    if not ckpts:
        print(f"--resume: no checkpoint-* in {out_dir}, starting from step 0", flush=True)
        return None
    print(f"--resume: resuming from {ckpts[-1]}", flush=True)
    return ckpts[-1]


def remove_checkpoints(out_dir: str) -> list[str]:
    """Delete OUT/checkpoint-* (optimizer state etc.) so OUT holds only the final model."""
    import shutil
    removed = hf_checkpoints(out_dir)
    for path in removed:
        shutil.rmtree(path)
    return removed


def load_hf_dataset(name: str, config: str | None, split: str, revision: str | None):
    """load_dataset for a hub id (pinned to `revision` if given) or a local .jsonl/.json file."""
    from datasets import load_dataset
    if name.endswith((".jsonl", ".json")):
        if revision:
            raise SystemExit("--dataset-revision only applies to hub datasets, not local files")
        return load_dataset("json", data_files=name, split="train")
    return load_dataset(name, config, split=split, revision=revision)
