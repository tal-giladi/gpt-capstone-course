#!/usr/bin/env bash
# Power the box off after N minutes of GPU idle, so a finished or crashed run stops billing.
#
#   nohup bash scripts/autoshutdown.sh --minutes 20 > autoshutdown.log 2>&1 &
#   bash scripts/autoshutdown.sh --minutes 1 --dry-run        # test: prints instead of shutting down
#
# "Idle" = every GPU below --threshold % utilization at each poll. On RunPod the pod is
# stopped with runpodctl (a plain `shutdown` only restarts the container there); elsewhere
# `sudo shutdown -h now`. Stopped is not deleted: on most clouds you must also terminate
# the instance (and its disk) to stop all charges.
set -euo pipefail

MINUTES=20
THRESHOLD=5
POLL=60
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --minutes) MINUTES="$2"; shift 2 ;;
    --threshold) THRESHOLD="$2"; shift 2 ;;
    --poll) POLL="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 1; }
idle_since=$(date +%s)
echo "$(date -Is) watching GPUs: shutdown after ${MINUTES} min below ${THRESHOLD}% (poll ${POLL}s)"

power_off() {
  echo "$(date -Is) GPUs idle for ${MINUTES} min -> powering off"
  if [[ $DRY_RUN -eq 1 ]]; then echo "(dry run) would power off now"; exit 0; fi
  if [[ -n "${RUNPOD_POD_ID:-}" ]] && command -v runpodctl >/dev/null; then
    runpodctl stop pod "$RUNPOD_POD_ID"
  else
    sudo shutdown -h now
  fi
}

while true; do
  max_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | sort -n | tail -1 | tr -d ' ')
  now=$(date +%s)
  if [[ "${max_util:-0}" -ge "$THRESHOLD" ]]; then
    idle_since=$now
  elif (( now - idle_since >= MINUTES * 60 )); then
    power_off
  fi
  sleep "$POLL"
done
