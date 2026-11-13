# Code requests from Module 2 (for labs/common owner)

Lessons 02.1–02.3 are written against the contract. These are behaviours the lessons assume or
would benefit from; please confirm or adjust and tell the module author if a lesson must change.

1. **`scripts/download_fineweb.py` output layout.** Lesson 02.3 assumes the `.parquet` files land
   directly in `--out` (e.g. `data/raw/sample-10BT/000_00000.parquet`), not in a nested
   `sample/10BT/` folder. The manifest tool (`labs/module-02/fineweb_manifest.py`) checks
   `--dir/<filename>`. If the script keeps the repo path, either flatten it or tell me and I change
   the lesson commands.
2. **`--max-files N` takes the first N files in sorted path order** (`000_00000` … `002_00009` for
   N=30 of sample/100BT). The manifest tool assumes the same order.
3. **Add `--revision SHA`** (default: main) to `download_fineweb.py` and pass it to
   `snapshot_download`/`hf_hub_download`, so learners can pin the dataset commit at download time.
   The lesson currently pins at verification time instead.
4. **Use `HF_TOKEN` from the environment and do not set `HF_HUB_ENABLE_HF_TRANSFER`** (ignored since
   huggingface_hub 1.0; `hf_xet` is the default). The lesson sets `HF_XET_HIGH_PERFORMANCE=1`.
   Please require `huggingface_hub>=1.2` (automatic 429 retry).
5. **Install on a CPU box without CUDA torch.** Lesson 02.3 runs
   `uv pip install -e . --torch-backend cpu`. If `download_fineweb.py` does not import torch at all,
   that is even better; please keep data scripts torch-free at import time if easy.
6. **`scripts/sync.sh` convention.** Lessons define the remote via env vars (`RCLONE_CONFIG_R2_*`)
   and a project root `MYGPT_REMOTE=r2:mygpt`. Please make sync.sh read `MYGPT_REMOTE` and use
   the bucket layout fixed in lesson 02.2:
   `raw/fineweb-edu/<subset>/`, `manifests/`, `tokenizer/`, `tok/<tokenizer>-<subset>/`,
   `ckpt/<run_name>/`, `hf/`, `tmp/`. sync.sh must only ever use `rclone copy` (never `sync`) by
   default.
