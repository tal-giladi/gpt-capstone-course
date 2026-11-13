"""Small helpers shared by the scripts (kept out of the mygpt package on purpose)."""
from __future__ import annotations

import glob
import os
from typing import Iterator


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
