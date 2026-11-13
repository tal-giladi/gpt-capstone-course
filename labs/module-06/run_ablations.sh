#!/usr/bin/env bash
# Run every active line of an ablation queue file, one after another, on one GPU (Module 6).
#
#   bash tools/run_ablations.sh tools/ablations.txt                 # all active lines
#   bash tools/run_ablations.sh tools/ablations.txt abl_rope         # only these names
#   DRY_RUN=1 bash tools/run_ablations.sh tools/ablations.txt        # print the commands only
#
# Queue format: "NAME OVERRIDES..." per line; "#" comments; "@MACRO overrides..." defines a macro
# whose name is replaced by its overrides in later lines (see ablations.txt).
# Each run: python -m mygpt.train --config $CONFIG OVERRIDES train.out_dir=runs/NAME
#   train.run_name=NAME, plus train.wandb_project=$WANDB_PROJECT if set.
# Safe to rerun after a crash or on a fresh box: a run marked DONE (locally or in the bucket) is
# skipped, an unfinished run resumes from its newest local checkpoint (mygpt.train auto-resume).
# After each finished run, log.jsonl and a DONE marker are copied to $MYGPT_REMOTE/ckpt/NAME/
# (checkpoints only with PUSH_CKPT=1: they are 1.3-1.6 GB each and ablations do not need them).
# A run that fails (e.g. NaN, exit 1) is reported and the queue continues with the next line.
set -uo pipefail

QUEUE="${1:?usage: run_ablations.sh QUEUE_FILE [NAME ...]}"; shift
ONLY=("$@")
CONFIG="${CONFIG:-configs/ablation_base.yaml}"
PYTHON="${PYTHON:-python}"
RUNS="${RUNS:-runs}"
REMOTE="${MYGPT_REMOTE:-r2:mygpt}"; REMOTE="${REMOTE%/}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
PUSH_CKPT="${PUSH_CKPT:-0}"
DRY_RUN="${DRY_RUN:-0}"

have_rclone=0; command -v rclone >/dev/null && [[ -n "${RCLONE_CONFIG_R2_TYPE:-}" || -f "$HOME/.config/rclone/rclone.conf" ]] && have_rclone=1
[[ $have_rclone == 1 ]] || echo "[queue] WARNING: rclone/remote not configured: results stay on this box only"

declare -A MACRO=()
summary=()
while IFS= read -r raw || [[ -n "$raw" ]]; do
  line="${raw%%#*}"; line="$(echo "$line" | xargs)"   # strip comments and surrounding space
  [[ -z "$line" ]] && continue
  if [[ "$line" == @* ]]; then
    key="${line%% *}"; key="${key#@}"; MACRO[$key]="${line#* }"; continue
  fi
  read -r -a words <<< "$line"
  name="${words[0]}"
  if [[ ${#ONLY[@]} -gt 0 && ! " ${ONLY[*]} " == *" $name "* ]]; then continue; fi
  overrides=()
  for w in "${words[@]:1}"; do
    if [[ -n "${MACRO[$w]:-}" ]]; then read -r -a m <<< "${MACRO[$w]}"; overrides+=("${m[@]}"); else overrides+=("$w"); fi
  done
  out="$RUNS/$name"
  if [[ -f "$out/DONE" ]] || { [[ $have_rclone == 1 ]] && [[ -n "$(rclone lsf "$REMOTE/ckpt/$name/" --include DONE 2>/dev/null)" ]]; }; then
    echo "[queue] $name: DONE, skipping"; summary+=("$name skipped(done)"); continue
  fi
  cmd=("$PYTHON" -m mygpt.train --config "$CONFIG" "${overrides[@]}" "train.out_dir=$out" "train.run_name=$name")
  [[ -n "$WANDB_PROJECT" ]] && cmd+=("train.wandb_project=$WANDB_PROJECT")
  echo "[queue] $(date -u +%H:%M:%S) start $name: ${cmd[*]}"
  if [[ $DRY_RUN == 1 ]]; then summary+=("$name dry-run"); continue; fi
  mkdir -p "$out"
  t0=$(date +%s)
  "${cmd[@]}" < /dev/null 2>&1 | tee -a "$out/train.out"
  rc=${PIPESTATUS[0]}
  mins=$(( ($(date +%s) - t0) / 60 ))
  if [[ $rc -eq 0 ]]; then
    date -u +%FT%TZ > "$out/DONE"
    if [[ $have_rclone == 1 ]]; then
      rclone copy "$out" "$REMOTE/ckpt/$name" --include "log.jsonl" --include "DONE" --include "train.out" --retries 5
      [[ $PUSH_CKPT == 1 ]] && rclone copy "$out" "$REMOTE/ckpt/$name" --include "ckpt_*.pt" --retries 5
    fi
    summary+=("$name ok ${mins}min")
  else
    echo "[queue] $name FAILED with exit code $rc (see $out/train.out); continuing"
    summary+=("$name FAILED rc=$rc ${mins}min")
  fi
done < "$QUEUE"

echo "[queue] finished $(date -u +%H:%M:%S):"
printf '  %s\n' "${summary[@]}"
