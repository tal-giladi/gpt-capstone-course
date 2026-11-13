"""Download FineWeb-Edu parquet files from the Hugging Face Hub, flat into --out.

  python scripts/download_fineweb.py --subset sample-10BT --out data/raw/sample-10BT
  python scripts/download_fineweb.py --subset sample-100BT --max-files 30 --out data/raw/sample-100BT \
      --revision <dataset-commit-sha>

Files land directly in --out (data/raw/sample-10BT/000_00000.parquet, ...). --max-files N takes
the first N files in sorted path order (000_00000, 000_00001, ..., 001_00000, ...). sample-10BT
is ~28.5 GB of parquet (~10B GPT-2 tokens). Re-running skips files already present.
Auth: HF_TOKEN from the environment (optional, higher rate limits). Downloads use hf_xet
(the huggingface_hub default); set HF_XET_HIGH_PERFORMANCE=1 on a big box. No torch needed.
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

REPO = "HuggingFaceFW/fineweb-edu"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subset", choices=["sample-10BT", "sample-100BT", "sample-350BT"], default="sample-10BT")
    p.add_argument("--max-files", type=int, default=0, help="only the first N parquet files in sorted order (0 = all)")
    p.add_argument("--revision", default="main", help="dataset commit SHA to pin (default: main)")
    p.add_argument("--out", required=True, help="directory the .parquet files are written into (flat)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--list", action="store_true", help="only list the files that would be downloaded")
    args = p.parse_args(argv)

    from huggingface_hub import HfApi, hf_hub_download

    folder = "sample/" + args.subset.split("-", 1)[1]  # sample-10BT -> sample/10BT
    api = HfApi()  # picks up HF_TOKEN from the environment
    files = sorted(f for f in api.list_repo_files(REPO, repo_type="dataset", revision=args.revision)
                   if fnmatch.fnmatch(f, f"{folder}/*.parquet"))
    if args.max_files:
        files = files[:args.max_files]
    print(f"{len(files)} files from {REPO}/{folder} @ {args.revision}")
    for f in files:
        print("  ", f)
    if args.list:
        return files

    os.makedirs(args.out, exist_ok=True)
    staging = os.path.join(args.out, ".staging")

    def fetch(remote: str) -> str:
        dest = os.path.join(args.out, os.path.basename(remote))
        if os.path.exists(dest):
            return f"skip {os.path.basename(remote)} (exists)"
        path = hf_hub_download(REPO, remote, repo_type="dataset", revision=args.revision, local_dir=staging)
        os.replace(path, dest)
        return f"ok   {os.path.basename(remote)} ({os.path.getsize(dest) / 1e9:.2f} GB)"

    with ThreadPoolExecutor(args.workers) as pool:
        for msg in pool.map(fetch, files):
            print(msg, flush=True)
    shutil.rmtree(staging, ignore_errors=True)
    total = sum(os.path.getsize(os.path.join(args.out, os.path.basename(f))) for f in files)
    print(f"done: {len(files)} files, {total / 1e9:.1f} GB in {args.out}")
    return files


if __name__ == "__main__":
    main()
