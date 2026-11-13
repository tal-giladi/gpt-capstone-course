# Module 2 lab files — data and storage

Helpers for lessons 02.2 and 02.3. The project code itself (`scripts/download_fineweb.py`,
`scripts/sync.sh`) is in `labs/common/`.

| File | Used in | What it does |
|---|---|---|
| `r2.env.example` | 02.2 | rclone remote `r2` configured only through environment variables. Copy it outside the repo, fill in your R2 keys, `chmod 600`. |
| `fineweb_manifest.py` | 02.3 | Checks downloaded FineWeb-Edu parquet files (size + sha256) against the Hugging Face repo at a fixed commit and writes a JSON manifest. |
| `DATA_CARD.template.md` | 02.3 | Template for the `DATA_CARD.md` artifact of the module. |

Requirements: Python 3.11+, `huggingface_hub` >= 1.2 (installed with the project), rclone >= 1.74.
Checked against the Hugging Face Hub API and rclone docs on 2026-11-13.

Never commit a filled-in `r2.env`; it holds a secret that can read and delete your bucket.
