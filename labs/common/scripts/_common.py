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
