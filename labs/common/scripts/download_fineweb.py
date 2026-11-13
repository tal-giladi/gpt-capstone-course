"""Download FineWeb-Edu parquet files from the Hugging Face Hub.

  python scripts/download_fineweb.py --subset sample-10BT --out data/raw/fineweb_edu_10BT
  python scripts/download_fineweb.py --subset sample-100BT --max-files 40 --out data/raw/fineweb_edu_100BT

sample-10BT is ~28.5 GB of parquet (~10B GPT-2 tokens). sample-100BT is much bigger; use
--max-files to take only what you need (each file is roughly 2 GB / ~1B tokens; check the
printed list). Set HF_TOKEN for higher rate limits. Re-running skips files already present.
"""
from __future__ import annotations

import argparse
import fnmatch
import os

REPO = "HuggingFaceFW/fineweb-edu"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subset", choices=["sample-10BT", "sample-100BT", "sample-350BT"], default="sample-10BT")
    p.add_argument("--max-files", type=int, default=0, help="download only the first N parquet files (0 = all)")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--list", action="store_true", help="only list the files that would be downloaded")
    args = p.parse_args(argv)

    from huggingface_hub import HfApi, snapshot_download

    folder = "sample/" + args.subset.split("-", 1)[1]  # sample-10BT -> sample/10BT
    files = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset")
                   if fnmatch.fnmatch(f, f"{folder}/*.parquet"))
    if args.max_files:
        files = files[:args.max_files]
    print(f"{len(files)} files from {REPO}/{folder}")
    for f in files:
        print("  ", f)
    if args.list:
        return files
    os.makedirs(args.out, exist_ok=True)
    snapshot_download(REPO, repo_type="dataset", allow_patterns=files, local_dir=args.out, max_workers=args.workers)
    print(f"done -> {os.path.join(args.out, folder)}")
    return files


if __name__ == "__main__":
    main()
