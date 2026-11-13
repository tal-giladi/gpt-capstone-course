"""Verify downloaded FineWeb-Edu parquet files against the Hugging Face repo and write a manifest.

Usage (from the folder that holds the downloaded files):

    python fineweb_manifest.py --dir data/raw/sample-10BT --prefix sample/10BT \
        --out manifests/sample-10BT.json
    python fineweb_manifest.py --dir data/raw/sample-100BT --prefix sample/100BT \
        --max-files 30 --out manifests/sample-100BT-30.json

It asks the Hub for the file list (size + sha256) at one commit of the dataset repo, checks that
every expected local file exists with the same size and sha256, and writes a JSON manifest you
keep next to the data in your bucket and cite in DATA_CARD.md. Exit code 1 if anything differs.

Needs only `huggingface_hub` (already installed with the project). Set HF_TOKEN to avoid
anonymous rate limits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi

REPO = "HuggingFaceFW/fineweb-edu"
# Published sizes of the sample subsets in GPT-2 tokens, used for a rough token estimate only.
NOMINAL_TOKENS = {"sample/10BT": 10e9, "sample/100BT": 100e9, "sample/350BT": 350e9}


def sha256_of(path: Path, chunk: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, type=Path, help="local folder with the .parquet files")
    p.add_argument("--prefix", required=True, help="path in the repo, e.g. sample/10BT")
    p.add_argument("--repo", default=REPO)
    p.add_argument("--revision", default=None, help="commit sha to check against (default: current main)")
    p.add_argument("--max-files", type=int, default=None, help="only the first N files in sorted order")
    p.add_argument("--no-hash", action="store_true", help="check sizes only (fast)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()

    api = HfApi()
    revision = a.revision or api.dataset_info(a.repo).sha
    remote = sorted(
        (f for f in api.list_repo_tree(a.repo, path_in_repo=a.prefix, repo_type="dataset", revision=revision)
         if f.path.endswith(".parquet")),
        key=lambda f: f.path,
    )
    if a.max_files:
        remote = remote[: a.max_files]
    if not remote:
        print(f"No parquet files under {a.prefix} at {revision}", file=sys.stderr)
        return 1

    def check(f):
        local = a.dir / Path(f.path).name
        row = {"file": Path(f.path).name, "repo_path": f.path, "size": f.size,
               "sha256": f.lfs.sha256 if f.lfs else None, "status": "ok"}
        if not local.exists():
            row["status"] = "missing"
        elif local.stat().st_size != f.size:
            row["status"] = f"size {local.stat().st_size} != {f.size}"
        elif not a.no_hash and row["sha256"] and sha256_of(local) != row["sha256"]:
            row["status"] = "sha256 mismatch"
        return row

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        rows = list(pool.map(check, remote))

    bad = [r for r in rows if r["status"] != "ok"]
    total = sum(r["size"] for r in rows)
    all_files = list(api.list_repo_tree(a.repo, path_in_repo=a.prefix, repo_type="dataset", revision=revision))
    subset_bytes = sum(f.size for f in all_files if f.path.endswith(".parquet"))
    nominal = NOMINAL_TOKENS.get(a.prefix.rstrip("/"))
    est_tokens = round(nominal * total / subset_bytes) if nominal and subset_bytes else None

    manifest = {
        "repo": a.repo,
        "revision": revision,
        "prefix": a.prefix,
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hash_checked": not a.no_hash,
        "files": rows,
        "n_files": len(rows),
        "total_bytes": total,
        "est_gpt2_tokens": est_tokens,
        "problems": len(bad),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"{a.repo}@{revision[:12]} {a.prefix}: {len(rows)} files, {total / 1e9:.2f} GB, "
          f"~{(est_tokens or 0) / 1e9:.2f}B GPT-2 tokens (estimate)")
    for r in bad:
        print(f"  PROBLEM {r['file']}: {r['status']}")
    print("OK" if not bad else f"{len(bad)} problem(s) - re-download those files")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
