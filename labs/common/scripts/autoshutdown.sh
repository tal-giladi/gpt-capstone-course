#!/usr/bin/env bash
# Stop the box after the GPUs have been idle for N minutes, so a finished or crashed run stops
# billing.
#
#   nohup bash scripts/autoshutdown.sh --idle-minutes 20 > autoshutdown.log 2>&1 &
#   bash scripts/autoshutdown.sh --idle-minutes 5 --stop-cmd 'runpodctl pod stop $RUNPOD_POD_ID'
#   bash scripts/autoshutdown.sh --idle-minutes 1 --grace-minutes 0 --dry-run   # test it
#
# Idle = every GPU below --threshold % (default 5) at each poll (every --poll s, default 30)
# for --idle-minutes consecutive minutes. Nothing is stopped during the first --grace-minutes
# (default 10) after start, so setup time does not count as idle.
# Default --stop-cmd: on RunPod (RUNPOD_POD_ID set) `runpodctl pod stop $RUNPOD_POD_ID`
# (falling back to `runpodctl stop pod ...`); elsewhere `sudo shutdown -h now`.
# WARNING: `shutdown`/`poweroff` inside a RunPod container does not stop the pod, and on
# Lambda / Hetzner / Vast.ai a shutdown does NOT stop billing: pass a --stop-cmd that calls the
# provider API (Lambda terminate endpoint, `vastai stop instance ID`, ...).
set -uo pipefail

IDLE_MINUTES=20
GRACE_MINUTES=10
THRESHOLD=5
POLL=30
STOP_CMD=""
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --idle-minutes) IDLE_MINUTES="$2"; shift 2 ;;
    --grace-minutes) GRACE_MINUTES="$2"; shift 2 ;;
    --threshold) THRESHOLD="$2"; shift 2 ;;
    --poll) POLL="$2"; shift 2 ;;
    --stop-cmd) STOP_CMD="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

log() { echo "$(date -Is) $*"; }
command -v nvidia-smi >/dev/null || { log "nvidia-smi not found"; exit 1; }

if [[ -z "$STOP_CMD" ]]; then
  if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
    STOP_CMD='runpodctl pod stop "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID"'
  else
    STOP_CMD='sudo shutdown -h now'
    log "WARNING: default stop command is 'sudo shutdown -h now'. On Lambda, Hetzner and Vast.ai this does"
    log "WARNING: NOT stop billing; pass --stop-cmd with the provider's stop/terminate API call."
  fi
fi

start=$(date +%s)
idle_since=""
log "watching GPUs: stop after ${IDLE_MINUTES} min below ${THRESHOLD}% (poll ${POLL}s, grace ${GRACE_MINUTES} min)"
log "stop command: ${STOP_CMD}"

while true; do
  utils=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ' | paste -sd, -)
  max_util=$(echo "$utils" | tr ',' '\n' | sort -n | tail -1)
  now=$(date +%s)
  if [[ "${max_util:-0}" -ge "$THRESHOLD" ]]; then
    idle_since=""
    log "busy (util ${utils}%)"
  else
    [[ -z "$idle_since" ]] && idle_since=$now
    idle_min=$(( (now - idle_since) / 60 ))
    if (( now - start < GRACE_MINUTES * 60 )); then
      log "idle ${idle_min} min (util ${utils}%), still in grace period"
    elif (( now - idle_since >= IDLE_MINUTES * 60 )); then
      log "idle ${idle_min} min >= ${IDLE_MINUTES}: stopping with: ${STOP_CMD}"
      if [[ $DRY_RUN -eq 1 ]]; then log "(dry run) not stopping"; exit 0; fi
      bash -c "$STOP_CMD"
      exit $?
    else
      log "idle ${idle_min}/${IDLE_MINUTES} min (util ${utils}%)"
    fi
  fi
  sleep "$POLL"
done
