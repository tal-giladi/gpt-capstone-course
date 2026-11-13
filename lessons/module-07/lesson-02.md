---
id: "07.2"
module: 7
minutes: 30
practice_minutes: 60
prerequisites: ["07.1"]
objectives:
  - Tell a healthy main-run curve (loss, val bpb, HellaSwag, grad norm, tokens/s, the WSD decay drop) from one that needs action.
  - Decide whether a loss spike is ignored or rolled back, and perform a rollback with the right checkpoint, skip_batches and, if needed, a lower learning rate.
  - Diagnose a throughput drop or a hang (thermal or slow GPU, data stall, NCCL) and resume on a new node after node death.
  - Track cost against the plan during the run, shorten the run safely if money runs short, and keep a run log of every incident.
volatility: implementation
sources:
  - title: "Chowdhery et al., PaLM: Scaling Language Modeling with Pathways (section on training instability: restart before the spike, skip batches)"
    url: https://arxiv.org/abs/2204.02311
  - title: "Zhang et al., OPT: Open Pre-trained Transformer Language Models (loss divergences, lowering LR and restarting; logbook)"
    url: https://arxiv.org/abs/2205.01068
  - title: "Hägele et al., Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations (WSD: the loss drop in the cooldown)"
    url: https://arxiv.org/abs/2405.18392
  - title: "NVIDIA: Xid errors (GPU fallen off the bus, ECC and NVLink errors)"
    url: https://docs.nvidia.com/deploy/xid-errors/contents.html
  - title: "PyTorch: ProcessGroupNCCL environment variables (watchdog, error handling)"
    url: https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html
  - title: "RunPod docs: Pod pricing (billing per second; pods stop at $0 balance)"
    url: https://docs.runpod.io/pods/pricing
last_verified: "2026-11-13"
---

# 07.2 · Babysitting a long run

A long run mostly takes care of itself, and the skill is knowing which of the few things that do go wrong need you. A loss spike that heals in two hundred steps is noise; one that does not heal means rolling back to a checkpoint and stepping past the data that triggered it. A run that slows by 20% costs you an extra hour of an $35-an-hour node; a run that hangs silently costs everything until someone looks. This lesson gives you a routine for checking in, a picture of what your healthy curves look like, and a tested procedure for each failure, so that every decision in the night is one you already made in daylight and wrote into the run log.

## Goal

Your main run reaches its last step with every check-in, incident and decision recorded in `reports/main-run-log.md`: what you saw (with the `watch_run.py` incident table), what you did (rollback, skipped batches, LR change, resume on a new node, shortened run) and what it cost. The bucket holds a continuous chain of checkpoints for the run you are keeping, with any rolled-back checkpoints moved aside so no auto-resume can pick them up.

## What you need

- The running main run from [07.1](lesson-01.md), its node, the ntfy alerts and the W&B run.
- Your rollback and resume procedure from [05.2](../module-05/lesson-02.md) and the baseline curves in `results.md` from [05.3](../module-05/lesson-03.md).
- About 10 minutes per check-in, every 1–2 hours while you are awake, plus whatever an incident takes (15–45 minutes each).
- **Expected cost:** the node time of the run itself (in 07.1's prediction, billed in 07.3). Incidents add: a rollback ~$5–10, a dead node ~$10–20 (lost steps plus setup of a new node), a 20% slowdown left alone ~$35 per extra hour.

## Choices and costs

### What it costs to look away

| Failure | How you notice | Cost per hour unnoticed | What catches it |
|---|---|---|---|
| Hang (NCCL, a dead GPU) with the node still up | No new log lines, GPUs at 100% or 0% | $35 (the node bills, nothing trains) | `watch_run.py` stale alert after 10 min; `autoshutdown.sh` only if GPUs go idle |
| Node dies | Watcher silent, W&B marks the run crashed | ~$0 billed, but no progress | W&B crash email; the provider console |
| Unrecovered loss spike | Loss stays high, grad norm high | $35 of training that will be thrown away | Watcher spike alert; your check-ins |
| NaN stop (exit code 1) | Run ends; GPUs go idle | $0 after the idle guard stops the pod (20 min) | Watcher stale alert, W&B finished/crashed email |
| 20% slowdown | tokens/s below the run's median | ~$7 per hour of run | Watcher slow alert (3 records in a row) |

The rule that follows: phone alerts for the cheap-to-detect failures, a short check-in every 1–2 hours for the rest, and no shame in sleeping through a 5-hour run: the idle guard and the hard stop cap the worst case at a few hours of node time.

### Spikes: ignore, or roll back?

| What you see | Verdict | Action |
|---|---|---|
| Loss up 0.05–0.3 for a few records, back within ~300 steps; grad norm briefly up | Normal at this scale | Log it (the watcher lists it as "recovered") and do nothing |
| Val loss or HellaSwag one eval worse than the last | Noise (HellaSwag's standard error is ~0.005) | Look at the trend over 3+ evals |
| Loss drops visibly in the last 20% of steps | WSD decay, by design | Expected; it is where most of the final gain comes from |
| Loss up > 0.3 and not back after ~500 steps, or still rising; grad norm several times its usual value | Divergence starting | Roll back (procedure below) |
| `non-finite loss/grad`, exit code 1 | Divergence finished | Roll back |
| Two rollbacks fail at about the same step | Data at that position, or LR too high for this state | Skip more batches and lower the LR; if it fails a third time, stop and investigate before paying more |

This is how the large published runs handled it: PaLM restarted from a checkpoint about 100 steps before a spike and skipped roughly 200–500 batches (PUBLICLY DOCUMENTED, PaLM, "Training instability"); OPT lowered the learning rate and restarted from an earlier checkpoint (PUBLICLY DOCUMENTED, OPT logbook). At 350M parameters with QK-norm and gradient clipping, spikes are much rarer than at 100B+ (REASONABLE PRACTICE: you may see none).

### When money runs short: shorten with WSD

Your schedule is warmup → constant LR → linear decay over the last 20% of steps, and the decay start is computed from `train_tokens`. Lowering `train_tokens` on resume moves the decay earlier and still gives a properly decayed model (the lab code tests this: a run resumed from a stable-phase checkpoint with a smaller `train_tokens` matches an uninterrupted shorter run exactly).

| New `train_tokens` | Total steps | Decay starts at step | Saves vs 20B (example speed) |
|---|---|---|---|
| 20B (plan) | 38,146 | 30,516 | — |
| 18B | 34,332 | 27,465 | ~0.4 h, ~$15 |
| 16B | 30,517 | 24,413 | ~0.9 h, ~$30 |

It only works **before** the new decay start; after that the LR would jump down instead of decaying smoothly. Once the original decay has started (step 30,516), finish the run.

## Do this

### 1. The check-in routine (10 min, every 1–2 hours)

From your laptop: the W&B run (loss, val_bpb, hellaswag, grad_norm, tokens_per_s against `step`) and the latest watcher line. On the node:

```bash
tail -2 ~/mygpt/runs/watch.out
nvidia-smi --query-gpu=index,utilization.gpu,clocks.sm,temperature.gpu,power.draw,memory.used --format=csv
rclone lsf r2:mygpt/ckpt/main_350m/ --include "ckpt_*.pt" | tail -2      # newest upload <= ~7 min behind the run
df -h ~/mygpt | tail -1
```

Add one timeline row to `reports/main-run-log.md` per check-in (time, step, "healthy" or what you saw). Expect all 8 GPUs at ~100% utilization, similar clocks (~1,980 MHz SM boost on H100 SXM under load, a little lower when power-capped), ~600–700 W, temperatures below ~80 °C.

### 2. What healthy looks like

| Steps (tokens) | Train loss | Val bpb | HellaSwag (in-training) | Grad norm | Notes |
|---|---|---|---|---|---|
| 1 | ~10.4 | | | high | $\ln 32{,}768$ |
| 0–476 (warmup, 0.25B) | 10.4 → ~5 | | ~0.25 | high, falling | LR ramps up |
| ~2,000 (1B) | ~3.4–3.7 | ~1.05–1.15 | ~0.27–0.29 | settles | Fast part over |
| ~10,000 (5.2B) | slow decline | ~0.93–0.98 | ~0.32–0.36 | flat | Below your baseline's curve at equal tokens |
| ~30,000 (15.7B, end of stable phase) | nearly flat | ~0.89–0.93 | ~0.36–0.40 | flat | Constant LR: slow progress is normal |
| 30,516–38,146 (decay) | drops visibly | falls ~0.03–0.05 | rises ~0.01–0.03 | changes as LR falls | The WSD cooldown |
| 38,146 (20B) | | ~0.84–0.89 | ~0.38–0.45 | | Targets: GPT-2 medium 0.375, GPT-3 Medium 0.436 (Karpathy's eval) |

All numbers are ESTIMATES from your baseline, published GPT-2/3 numbers and the size of WSD cooldown drops reported by Hägele et al.; your own tokenizer and recipe shift them. Train loss is in nats per token of your 32k tokenizer, so compare models in bpb, not loss. What matters is the shape: smooth decline, val bpb tracking train loss (single epoch), grad norm stable after warmup, tokens/s flat except short dips after each val eval (every 500 steps) and longer ones after each HellaSwag eval (every 2,000 steps).

### 3. Roll back a spike or a NaN (20–30 min)

Example: the watcher reports an unrecovered spike starting at step 29,430; grad norm began rising around 29,300. The last checkpoint safely before that is `ckpt_00029000.pt`.

```bash
cd ~/mygpt
pkill -TERM -f mygpt.train; sleep 60                     # if still running; it saves one more (bad) checkpoint
ls runs/main_350m/                                       # keep_last 3: e.g. 28000? 29000, 30000 (bad)
mkdir -p runs/main_350m/rolled_back
for s in 00030000; do                                    # every checkpoint AFTER the one you roll back to
  mv runs/main_350m/ckpt_$s.pt runs/main_350m/rolled_back/ 2>/dev/null || true
  rclone moveto r2:mygpt/ckpt/main_350m/ckpt_$s.pt r2:mygpt/ckpt/main_350m/rolled_back/ckpt_$s.pt
done
ls runs/main_350m/ckpt_00029000.pt || rclone copy r2:mygpt/ckpt/main_350m/ckpt_00029000.pt runs/main_350m/
rclone lsf r2:mygpt/ckpt/main_350m/ --include "ckpt_*.pt" | tail -1   # must be ckpt_00029000.pt
```

Moving the bad checkpoints aside **in the bucket too** matters: if this node dies later, auto-resume on a fresh node pulls the newest top-level `ckpt_*.pt`, and that must not be the diverged one. (`rclone lsf` without `-R` does not look into `rolled_back/`.)

Choose how many batches to skip so the resumed run steps past the stretch that triggered the spike plus a margin, as PaLM did: $N = S_{\text{spike}} - S_{\text{ckpt}} + 200 = 29{,}430 - 29{,}000 + 200 = 630$ batches (330M tokens, 1.7% of the run). Your config uses ~97% of the available tokens, so the skip still fits inside one epoch for up to ~1,100 batches. Edit the launch line, commit, relaunch:

```bash
sed -i 's|--config configs/main_350m.yaml|--config configs/main_350m.yaml train.skip_batches=630|' launch/main_8xh100.sh
git commit -am "Main run: roll back to step 29000, skip 630 batches (spike at 29430)" && git push
tmux send-keys -t train "bash launch/main_8xh100.sh" Enter
```

The run prints `resumed from runs/main_350m/ckpt_00029000.pt at step 29000` and `skip_batches=630: advanced the train loader ...`, and logs a `skip_batches` record. The skip is applied once: later resumes from checkpoints written by this run do not skip again, so you can leave the override in the script.

If the spike comes back at about the same step, add a learning-rate cut on the next rollback, e.g. `train.muon_lr=0.014 train.lr=4e-4` (a third lower; for an AdamW recipe only `train.lr`). Training settings may change on resume; model settings may not. Record every change in the run log's incident block with the commit hash.

### 4. Diagnose a slowdown (10–20 min)

The watcher alerts `slow` when the last three records are all below 85% of the run's median tokens/s.

| Check | Command | Points to |
|---|---|---|
| One GPU at a lower clock, or hot | `nvidia-smi -q -d PERFORMANCE,TEMPERATURE` (look at "Clocks Event Reasons": HW slowdown, thermal or power brake) | A throttling GPU slows every step: DDP waits for the slowest rank. Cannot be fixed from inside a rented node: stop at the next checkpoint, resume on a new node |
| GPU utilization below ~90% on all GPUs | `nvidia-smi dmon -s u -c 10` | The GPUs wait for something: data or the CPU |
| CPU or disk saturated | `top -o %CPU`, `df -h`, `cat /proc/loadavg` | Another process (a stray `rclone`, a `pip install`, a second run) competing; the data loader reads memory-mapped shards from local disk and needs little |
| Errors from the GPU driver | `dmesg -T`, look for lines with `Xid` (often not permitted inside a container; then the provider console or support) | Xid 79 (fallen off the bus), 48 / 63 / 64 (ECC), 74 (NVLink): hardware. Move to a new node |
| Slow only right after checkpoints | Watcher shows dips every 1,000 steps | Normal: rank 0 writes ~2.9 GB while the others wait a few seconds |

Rule of thumb: a slowdown you cannot explain in 15 minutes and that costs more than ~10% for more than an hour is worth a node change. A graceful stop (`pkill -TERM -f mygpt.train`) writes a checkpoint and exits 0; resume on the new node as in step 6.

### 5. Hang: no new steps (10 min)

Symptoms: watcher `stale` alert, `tail runs/main_350m.out` shows no new step, GPUs at 100% (NCCL waiting in a collective) or 0%. If `py-spy` is installed you can see where rank 0 is stuck (`py-spy dump --pid $(pgrep -f mygpt.train | head -1)`); either way:

```bash
pkill -KILL -f mygpt.train; pkill -KILL -f torchrun
nvidia-smi                                   # all 8 GPUs listed, none showing ERR!?
python tools/gpu_check.py                    # from Module 4: HEALTHY?
```

If all 8 GPUs are healthy, relaunch on the same node (`bash launch/main_8xh100.sh`); it resumes from the last checkpoint. If a GPU is missing or `gpu_check.py` fails, move to a new node. A stuck collective eventually ends with a NCCL watchdog timeout error after several minutes; do not wait for it if the watcher already told you.

### 6. Node death: resume on a new node (30 min)

The watcher went silent and W&B sent "run crashed"; the provider console shows the pod as exited or unreachable.

1. Terminate the dead pod in the console (a stopped pod with a volume disk still bills storage).
2. Deploy a new 8×H100 node and repeat 07.1 steps 4–6: hard stop, `setup_box.sh`, idle guard, `pull-tok`. Then pull the newest checkpoint and the log before launching:

   ```bash
   bash scripts/sync.sh pull-ckpt main_350m runs/main_350m
   python tools/preflight.py --config configs/main_350m.yaml --gpus 8 --tag recipe-v1   # WARN: will RESUME, expected
   bash launch/main_8xh100.sh
   ```

3. Restart the watcher with the **original** `--started` time plus the dead node's hours added to `--plan-hours`, so its cost line stays honest.

The new node does not have to be at the same provider: the checkpoints and the log are in your bucket, so any 8×H100 SXM node from 07.1's table works, with that provider's stop command in the guards (07.1 step 4). Change `--price-per-hour` on the watcher if the price differs, and add both providers' charges in 07.3.

If no 8-GPU node is available, resume on 4 GPUs (`--nproc_per_node=4` in a copy of the launch script): accumulation becomes $524{,}288 / (32 \times 1024 \times 4) = 4$, the loss curve is the same, wall clock doubles and cost per token stays about the same. The data position is stored as a position in the global token stream, so nothing repeats or skips (05.2 drill D).

### 7. Cost against the plan (5 min per check-in)

The watcher prints `$X so far, ~$Y total (plan $Z)` and alerts at 1.15× plan. When it does:

1. Find the cause (slowdown? rollbacks? a later start than planned?) and write it down.
2. Compare the projected total with the money left (`budget.md`). If it still fits with ~$60 to spare for Modules 8–11, continue.
3. If it does not, and you are before the new decay start in the table above, shorten: graceful stop, add `train.train_tokens=16_000_000_000` to the launch line, commit with the reason, relaunch. Record the new token count; Module 8 reports it.

### 8. Close the log (10 min, when the run ends)

```bash
python tools/watch_run.py runs/main_350m --target-tokens 20e9 --markdown >> reports/main-run-log.md
python tools/run_summary.py runs/main_350m --gpus 8 --single-gpu-tps <04.3 main tok/s> --markdown >> reports/main-run-log.md
git add reports/main-run-log.md && git commit -m "Main run log" && git push
```

Then go straight to [07.3](lesson-03.md): the node is still billing.

## How you know it worked

- The run reached its last step (38,146, or your shortened total) and logged a final `val_loss`, `val_bpb` and `hellaswag`; its val bpb is clearly below your baseline's.
- `reports/main-run-log.md` has a timeline row per check-in, an incident block for every alert you acted on, and the watcher's incident table; every rollback, skip or LR change has a commit hash.
- `rclone lsf r2:mygpt/ckpt/main_350m/ --include "ckpt_*.pt"` lists only checkpoints of the run you kept; anything rolled back sits under `rolled_back/`.
- Node hours so far are within your 07.1 prediction plus the documented incidents.

## When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| After a rollback the loss jumps 0.1+ above the pre-spike curve and stays | Resumed from the wrong checkpoint (a diverged one) or a fresh optimizer | Check the `resumed from` line names the checkpoint you chose; only resume through auto-resume from a complete checkpoint |
| Rollback resumes from the bad checkpoint again | A newer checkpoint left in `out_dir` (e.g. the one SIGTERM saved) or in the bucket on a fresh node | Move every checkpoint after your choice aside, locally **and** in the bucket; list both before relaunching |
| `skip_batches` line missing after relaunch | Override not in the launch line, or the checkpoint was written by a run that already used the same value | `grep skip_batches runs/main_350m.out`; use a different value to skip again |
| Watcher `stale` alert, but the run is fine | A long HellaSwag eval plus checkpoint at the same step, or the watcher's `--stale-minutes` too low | HellaSwag on 8 GPUs takes about a minute; raise `--stale-minutes` only if you see this repeatedly |
| Watcher alerts `cost` right after a node change | `--started` reset to the new node's start, or the dead node's hours not added | Restart it with the original start and the corrected plan |
| Loss falls faster than expected from step ~30,500 | WSD decay | Expected; note it in the log |
| Pod stopped mid-run with balance $0 | Credit ran out | Top up, deploy, resume (step 6); set a low-balance alert if the provider offers one |
| Hard stop fires before the end | Run slower than predicted plus incidents | Resume on a new node; set the new hard stop from the remaining ETA × 2 |

## Artifact

- `reports/main-run-log.md` (timeline, incidents, watcher incident table, `run_summary.py` table) and every launch-script change, committed.
- In the bucket: the continuous checkpoint chain in `ckpt/main_350m/` and `log.jsonl`; rolled-back checkpoints under `ckpt/main_350m/rolled_back/` (deleted in 07.3).
- `budget.md`: one line per incident that cost money, for example:

  ```text
  | 2026-11-17 | 07.2 | Main run incident: spike at 29430, rollback to 29000 + skip 630 batches (~0.25 h repeated) | ~$9 | $____ |
  ```
