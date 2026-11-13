#!/usr/bin/env bash
# rclone helpers for the project bucket. Only ever `rclone copy` (never `sync`), so nothing at
# the destination is deleted. The bucket root comes from MYGPT_REMOTE (default r2:mygpt); the
# remote itself is configured via RCLONE_CONFIG_R2_* env vars or `rclone config`.
#
# Bucket layout (lesson 02.2):
#   raw/fineweb-edu/<subset>/   manifests/   tokenizer/   tok/<tokenizer>-<subset>/
#   ckpt/<run_name>/            hf/          tmp/
#
#   bash scripts/sync.sh push-raw  sample-10BT data/raw/sample-10BT
#   bash scripts/sync.sh pull-raw  sample-10BT data/raw/sample-10BT
#   bash scripts/sync.sh push-tok  gpt2-10BT   data/tok/gpt2-10BT
#   bash scripts/sync.sh pull-tok  gpt2-10BT   data/tok/gpt2-10BT
#   bash scripts/sync.sh push-tokenizer tokenizers/bpe32k        # -> tokenizer/
#   bash scripts/sync.sh pull-tokenizer tokenizers/bpe32k
#   bash scripts/sync.sh push-manifests manifests                # -> manifests/
#   bash scripts/sync.sh push-ckpt baseline_124m runs/baseline_124m
#   bash scripts/sync.sh pull-ckpt baseline_124m runs/baseline_124m   # newest ckpt + log.jsonl only
#   bash scripts/sync.sh push-hf   my-gpt-350m hf/my-gpt-350m
#   bash scripts/sync.sh push LOCAL_PATH SUBPATH  |  pull SUBPATH LOCAL_PATH   # anything else
#   bash scripts/sync.sh ls [SUBPATH]
set -euo pipefail

REMOTE="${MYGPT_REMOTE:-r2:mygpt}"
REMOTE="${REMOTE%/}"
FLAGS=(--transfers 16 --checkers 32 --fast-list --progress --retries 5 --low-level-retries 20)
cmd="${1:-}"; shift || true

need() { [[ $# -ge 2 ]] || { echo "usage: $0 $cmd $USAGE" >&2; exit 2; }; }
copy() { echo "rclone copy $1 -> $2"; rclone copy "$1" "$2" "${FLAGS[@]}" "${@:3}"; }

case "$cmd" in
  push-raw)       USAGE="SUBSET LOCAL_DIR";   need "$@"; copy "$2" "$REMOTE/raw/fineweb-edu/$1" ;;
  pull-raw)       USAGE="SUBSET LOCAL_DIR";   need "$@"; copy "$REMOTE/raw/fineweb-edu/$1" "$2" ;;
  push-tok)       USAGE="NAME LOCAL_DIR";     need "$@"; copy "$2" "$REMOTE/tok/$1" ;;
  pull-tok)       USAGE="NAME LOCAL_DIR";     need "$@"; copy "$REMOTE/tok/$1" "$2" ;;
  push-tokenizer) USAGE="LOCAL_DIR";          need "$@" x; copy "$1" "$REMOTE/tokenizer" ;;
  pull-tokenizer) USAGE="LOCAL_DIR";          need "$@" x; copy "$REMOTE/tokenizer" "$1" ;;
  push-manifests) USAGE="LOCAL_DIR";          need "$@" x; copy "$1" "$REMOTE/manifests" ;;
  pull-manifests) USAGE="LOCAL_DIR";          need "$@" x; copy "$REMOTE/manifests" "$1" ;;
  push-hf)        USAGE="NAME LOCAL_DIR";     need "$@"; copy "$2" "$REMOTE/hf/$1" ;;
  pull-hf)        USAGE="NAME LOCAL_DIR";     need "$@"; copy "$REMOTE/hf/$1" "$2" ;;
  push-ckpt)
    USAGE="RUN_NAME RUN_DIR"; need "$@"
    copy "$2" "$REMOTE/ckpt/$1" --include "ckpt_*.pt" --include "log.jsonl" ;;
  pull-ckpt)
    USAGE="RUN_NAME RUN_DIR"; need "$@"
    latest=$(rclone lsf "$REMOTE/ckpt/$1" --include "ckpt_*.pt" | sort | tail -1)
    [[ -n "$latest" ]] || { echo "no checkpoints at $REMOTE/ckpt/$1" >&2; exit 1; }
    mkdir -p "$2"
    copy "$REMOTE/ckpt/$1/$latest" "$2"
    copy "$REMOTE/ckpt/$1" "$2" --include "log.jsonl" || true
    echo "pulled $latest -> $2" ;;
  push)           USAGE="LOCAL_PATH SUBPATH"; need "$@"; copy "$1" "$REMOTE/${2#/}" ;;
  pull)           USAGE="SUBPATH LOCAL_PATH"; need "$@"; copy "$REMOTE/${1#/}" "$2" ;;
  ls)             rclone lsf "$REMOTE/${1:-}" --format "sp" --recursive --max-depth 2 ;;
  *)              sed -n '2,22p' "$0"; exit 2 ;;
esac
