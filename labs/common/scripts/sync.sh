#!/usr/bin/env bash
# rclone helpers for moving data and checkpoints between a box and your bucket.
# Configure the remote once with `rclone config` (e.g. a Cloudflare R2 remote named "r2").
#
#   bash scripts/sync.sh push data/fineweb_edu_10B_gpt2  r2:my-gpt/data/fineweb_edu_10B_gpt2
#   bash scripts/sync.sh pull r2:my-gpt/data/fineweb_edu_10B_gpt2  data/fineweb_edu_10B_gpt2
#   bash scripts/sync.sh push-ckpt runs/baseline_124m  r2:my-gpt/checkpoints/baseline_124m
#   bash scripts/sync.sh pull-ckpt r2:my-gpt/checkpoints/baseline_124m  runs/baseline_124m   # latest ckpt + log only
#   bash scripts/sync.sh ls r2:my-gpt
#
# `copy` never deletes anything at the destination (we never use `rclone sync` here).
set -euo pipefail

FLAGS=(--transfers 16 --checkers 32 --fast-list --progress --retries 5 --low-level-retries 20)
cmd="${1:-}"; shift || true

case "$cmd" in
  push|pull)
    [[ $# -eq 2 ]] || { echo "usage: $0 $cmd SRC DST" >&2; exit 2; }
    rclone copy "$1" "$2" "${FLAGS[@]}"
    ;;
  push-ckpt)
    [[ $# -eq 2 ]] || { echo "usage: $0 push-ckpt RUN_DIR REMOTE" >&2; exit 2; }
    rclone copy "$1" "$2" --include "ckpt_*.pt" --include "log.jsonl" "${FLAGS[@]}"
    ;;
  pull-ckpt)
    [[ $# -eq 2 ]] || { echo "usage: $0 pull-ckpt REMOTE RUN_DIR" >&2; exit 2; }
    latest=$(rclone lsf "$1" --include "ckpt_*.pt" | sort | tail -1)
    [[ -n "$latest" ]] || { echo "no checkpoints at $1" >&2; exit 1; }
    mkdir -p "$2"
    rclone copy "$1/$latest" "$2" "${FLAGS[@]}"
    rclone copy "$1" "$2" --include "log.jsonl" "${FLAGS[@]}" || true
    echo "pulled $latest -> $2"
    ;;
  ls)
    rclone lsf "${1:?usage: $0 ls REMOTE}" --format "sp" --recursive --max-depth 2
    ;;
  *)
    sed -n '2,11p' "$0"; exit 2 ;;
esac
