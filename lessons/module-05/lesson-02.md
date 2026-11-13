---
id: "05.2"
module: 5
minutes: 20
practice_minutes: 60
prerequisites: ["05.1"]
objectives:
  - Choose a checkpoint interval from the cost of lost work versus checkpoint time.
  - Stop a multi-GPU run with SIGTERM and with SIGKILL and resume it, including on a fresh box from the bucket.
  - Verify that a resumed run continues the uninterrupted loss curve, and read a log with re-logged steps.
  - Recognise a NaN stop and follow the rollback procedure instead of restarting from scratch.
volatility: implementation
sources:
  - title: "PyTorch: torchrun (elastic launch; workers get SIGTERM, then SIGKILL after a timeout)"
    url: https://docs.pytorch.org/docs/stable/elastic/run.html
  - title: "RunPod: manage pods (stop, terminate)"
    url: https://docs.runpod.io/pods/manage-pods
  - title: "runpodctl stop pod"
    url: https://docs.runpod.io/runpodctl/reference/runpodctl-stop-pod
  - title: "RunPod pod pricing and billing"
    url: https://docs.runpod.io/pods/pricing
  - title: "Karpathy, llm.c discussion #481 (checkpointing and resuming the 124M run)"
    url: https://github.com/karpathy/llm.c/discussions/481
last_verified: "2026-11-13"
---

# 05.2 · Checkpoints, resume and surviving a dead node

On a rented node, the question is not whether a run gets interrupted but when: a host fails, a GPU throws an Xid error, a spot instance is reclaimed, or you stop it yourself because the loss looks wrong. A run you can resume costs you a few minutes when that happens; a run you cannot resume costs the whole run again. In this lesson you break a real multi-GPU run on purpose, four different ways, on the cheap rehearsal box, and bring it back each time, so that the recovery procedure is something you have done rather than something you have read.

## Goal

You have a written, tested recovery procedure: what `mygpt.train` saves and where, how it resumes after a polite stop (SIGTERM), a hard kill (SIGKILL), a lost box (pull from the bucket) and a NaN, and proof that a resumed run continues the same loss curve as an uninterrupted one (`run_summary.py --compare`). Your drill notes are committed as `reports/drill-05.2.md`, and you have chosen `ckpt_every` for the baseline and main run with the arithmetic written down.

## What you need

- The 2-GPU rehearsal box from [05.1](lesson-01.md) still running (or a new one set up the same way, ~15 minutes), with the two `gpt2-10BT` shards and HellaSwag cached.
- `tools/run_summary.py` from the [Module 5 lab files](../../labs/module-05/).
- About 80 minutes, ~60 of them on the box.
- **Expected cost: $0.50–3** (about 1 h of 2×RTX 4090 Community at ~$0.68/h, or 2×A100 at ~$2.78/h).

## Choices and costs

### What a checkpoint holds

`mygpt.train` writes `out_dir/ckpt_<step>.pt` every `ckpt_every` optimizer steps, at the end, and after a SIGTERM. Each file holds the model, every optimizer's state, the step, the data loader position, the RNG states and the full config. It is written to a `.tmp` file first and renamed, so a kill in the middle of a save never leaves a broken "latest" checkpoint. Only the newest `keep_last` stay on local disk. With `ckpt_remote` set, each new checkpoint is copied to the bucket in the background with `rclone copy`, together with the current `log.jsonl`; the final `log.jsonl` is copied again when the run ends or stops cleanly.

| Run | Weights fp32 | Optimizer state | Checkpoint file |
|---|---|---|---|
| Baseline 124M, AdamW | 0.50 GB | 1.0 GB (two moments) | ~1.5 GB |
| Main 337M, Muon + AdamW | 1.35 GB | ~1.5 GB (Muon: one momentum buffer for block matrices; AdamW: two moments for embeddings) | ~2.9 GB |

### Choosing `ckpt_every`

A failure at a random moment loses on average half an interval of work, plus the time to get a new box running. Saving blocks all GPUs for a few seconds each time (rank 0 writes to local NVMe, the others wait at a barrier; the upload runs in the background).

| | Baseline on 8×H100 (~0.16 s/step) | Main run on 8×H100 (~0.4 s/step) |
|---|---|---|
| Interval 1000 steps | ~2.7 min of node time, ~$1.60 at $35/h | ~6.7 min, ~$3.90 |
| Average loss per failure (half interval) | ~$0.80 | ~$2.00 |
| Save stalls per run (19 / 38 saves × ~3–5 s) | ~1–2 min, ~$1 | ~2–3 min, ~$1.50 |
| Bucket storage for all uploaded checkpoints | 19 × 1.5 GB = 29 GB, ~$0.43/month on R2 | 38 × 2.9 GB = 110 GB, ~$1.65/month |

Step times are ESTIMATES from 04.3's example numbers; use your own. Both configs ship with `ckpt_every: 1000`, which keeps the expected loss per failure around a dollar or two. Going to 250 would quadruple save stalls and bucket clutter to save about a dollar per failure. Keep 1000.

### Which signal you actually get

| Event | What your processes receive | Result with `mygpt.train` |
|---|---|---|
| You run `pkill -TERM -f mygpt.train` or `kill -TERM` | SIGTERM | Finishes the current step, saves `ckpt_<step>.pt`, exits 0 |
| SIGTERM sent only to `torchrun` | torchrun forwards SIGTERM to every worker, then SIGKILLs any still alive after 30 s (its default) | Same as above, if the save fits in 30 s (it does: a few seconds) |
| Provider stops or reclaims the pod | Depends on the container's init process; often SIGKILL after a short grace period, sometimes with no warning at all | Resume from the last `ckpt_every` checkpoint |
| Host dies, GPU falls off the bus, OOM killer | Nothing, or SIGKILL | Resume from the last checkpoint, possibly on a new box |
| Loss or gradient becomes NaN/inf | Training code detects it | Stops with exit code 1 and prints the last good checkpoint; nothing new is saved |

So SIGTERM handling is a bonus. The guarantee is the interval checkpoint already sitting in the bucket.

> [!NOTE]
> RunPod documented a 5-second SIGTERM-then-SIGKILL window for spot pods in older docs; current pages describe interruption without that detail. Check the live docs before relying on any warning period, and do not use spot capacity for the baseline or main run.

## Do this

All commands run on the rehearsal box in `~/mygpt` with the venv active. One drill run is small: a 65,536-token batch (accumulation 2 on 2 GPUs), 300 steps, a checkpoint every 50 steps.

### 1. Define the drill command (2 min)

```bash
cat > drill.sh <<'EOF'
#!/usr/bin/env bash
# usage: bash drill.sh OUT_NAME [extra overrides...]
name=$1; shift
torchrun --standalone --nproc_per_node=2 -m mygpt.train --config configs/baseline_124m.yaml \
  train.micro_batch_size=16 train.total_batch_tokens=65536 train.train_tokens=19_660_800 \
  train.warmup_tokens=1_310_720 train.eval_every=50 train.hellaswag_every=0 train.ckpt_every=50 \
  train.out_dir=runs/$name train.run_name=$name train.ckpt_remote=$MYGPT_REMOTE/ckpt/$name \
  train.wandb_project=my-gpt-drill "$@" 2>&1 | tee -a runs/$name.out
EOF
mkdir -p runs
```

### 2. Reference run, uninterrupted (5 min)

```bash
bash drill.sh drill_ref
```

It prints 300 steps and ends with `ckpt_00000300.pt`. This is what the interrupted run must reproduce.

### 3. Drill A: polite stop with SIGTERM (5 min)

In tmux window 0 start `bash drill.sh drill`. When it passes step ~120, in window 1:

```bash
pkill -TERM -f mygpt.train
```

Window 0 prints `[train] SIGTERM received: will checkpoint after this step and exit`, then `[train] saved runs/drill/ckpt_00000121.pt after SIGTERM; exiting` (your step number will differ). Restart with the identical command:

```bash
bash drill.sh drill        # -> [train] resumed from runs/drill/ckpt_00000121.pt at step 121
```

### 4. Drill B: hard kill with SIGKILL (5 min)

When the resumed run passes step ~180:

```bash
pkill -KILL -f mygpt.train
ls runs/drill/                       # newest checkpoint is ckpt_00000150.pt; 151-18x are lost
bash drill.sh drill                  # -> resumed from ... ckpt_00000150.pt at step 150
```

Steps 151 to ~180 run a second time and are logged a second time. `run_summary.py` keeps the last record of each step.

### 5. Drill C: the box is gone (10 min)

Simulate a dead node by removing all local state, then recover the way you would on a new box: pull the newest checkpoint **and** the log from the bucket before launching.

```bash
pkill -KILL -f mygpt.train            # at step ~240
mv runs/drill runs/drill_lost         # pretend the disk is gone
bash scripts/sync.sh pull-ckpt drill runs/drill
bash drill.sh drill                   # -> resumed from ... ckpt_00000200.pt (or 250) at step 200
```

`mygpt.train` would also pull the newest checkpoint by itself (because `ckpt_remote` is set and `out_dir` is empty), together with the bucket's `log.jsonl`. That log is uploaded with every checkpoint, so after a SIGKILL it ends at about the newest checkpoint in the bucket; the steps after it are missing from the log, but the resume re-runs them anyway. W&B (same run id = `run_name`) has the complete curves.

> [!IMPORTANT]
> On a real new box, `sync.sh pull-ckpt <run> runs/<run>` before relaunching, so you can check what came back before the GPUs start. Auto-resume pulls the log only into an `out_dir` that has no `log.jsonl`; a run that starts with a fresh log begins at the resume step, and at the end it overwrites the longer one in the bucket.

### 6. Compare with the reference (3 min)

```bash
python tools/run_summary.py runs/drill --compare runs/drill_ref
```

Look at two things. `resumes` and the step list show the history the log kept: the part pulled from the bucket (up to the last checkpoint upload before the SIGKILL, step 200 or 250) and then everything from the drill C resume onward, which starts at that same checkpoint, so there is no gap. Without the pulled log, everything before the resume would be missing, which is why you pull the log before relaunching. The `compare` block shows the train-loss difference to the reference at the steps both logs contain, including the final step 300. On CPU a resumed run matches bit for bit (the project's tests check this); on GPUs non-deterministic kernels make differences of up to ~0.01–0.02 normal. A jump of 0.1 or more right after a resume means something was not restored (wrong data position or a fresh optimizer).

### 7. Optional drill D: finish on fewer GPUs (3 min)

Stop the run again and resume it as a single process on one GPU:

```bash
pkill -TERM -f mygpt.train
python -m mygpt.train --config configs/baseline_124m.yaml train.micro_batch_size=16 \
  train.total_batch_tokens=65536 train.train_tokens=19_660_800 train.warmup_tokens=1_310_720 \
  train.eval_every=50 train.hellaswag_every=0 train.ckpt_every=50 train.out_dir=runs/drill \
  train.run_name=drill train.ckpt_remote=$MYGPT_REMOTE/ckpt/drill
```

It resumes with accumulation 4 instead of 2. The loader's position is stored as a position in the global token stream, so the data continues exactly where it stopped. This is your plan B if the 8-GPU node disappears late in a run and only smaller boxes are free.

### 8. Drill E: what a NaN stop looks like (3 min)

```bash
bash drill.sh drill_nan train.lr=1e6
```

Expect `[train] non-finite loss/grad at step 2 (loss=nan, ...). Stopping. Last good checkpoint: none` (or step 3), followed by torchrun's `ChildFailedError` summary with `exitcode: 1`; that summary is normal for a worker that exits with an error. On a real run the message names the newest checkpoint written before the problem.

The rollback procedure for a real NaN or a loss spike that does not recover:

1. Note the step and look at `grad_norm` in `log.jsonl` or W&B for the steps before it. A run that blew up usually shows rising gradient norms first.
2. Pick the last checkpoint from before the trouble started (with `keep_last: 3` you have the last three locally and all of them in the bucket). Move newer ones out of `out_dir` so auto-resume picks the one you chose.
3. Relaunch from it with a lower peak LR, for example `train.lr=4e-4` (train settings may change on resume; model settings may not).
4. If it fails again at the same step with the same data, suspect the data at that position and stop to investigate rather than burning more GPU time.

### 9. Write down and clean up (10 min)

Write `reports/drill-05.2.md` (in your repo): the four messages you saw, the resume steps, the `--compare` numbers, and your `ckpt_every` decision with the arithmetic. Then:

```bash
rm -rf runs/drill_lost
```

Copy `reports/drill-05.2.md` to your laptop or push it, then terminate the pod.

> [!CAUTION]
> Deleting from the bucket cannot be undone. If you want to free the drill space, delete only the drill prefixes, by name, after checking the listing: `rclone lsf r2:mygpt/ckpt/` first, then `rclone purge r2:mygpt/ckpt/drill_nan` (and `drill`, `drill_ref`, `rehearsal`). Never purge `ckpt/` itself.

## How you know it worked

- Drill A printed the SIGTERM message, saved a checkpoint at a step that is not a multiple of 50, and the restart printed `resumed from ... at step <that step>`.
- Drill B lost only the steps after the last multiple of 50, and the restart resumed exactly there.
- Drill C resumed on an empty `out_dir` from a checkpoint pulled from `r2:mygpt/ckpt/drill/`.
- `run_summary.py --compare` reports a final train loss within ~0.02 of the reference at step 300.
- Drill E stopped with the non-finite message, exit code 1 in torchrun's summary, and named the last good checkpoint (or `none`).
- `reports/drill-05.2.md` exists and states your `ckpt_every` for the baseline and main run.

## When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `ValueError: ... was trained with a different model config; use a new out_dir` | You changed a `model.*` setting (or the config file) and reused the out_dir | Model settings cannot change on resume. Use the original settings, or a new `out_dir` for a new run |
| Resume starts at step 0 although checkpoints exist | `out_dir` differs from the first launch (typo, relative path from another directory) | Launch from the repo root with exactly the same `train.out_dir` |
| After SIGTERM, torchrun prints `SignalException` and a worker was killed before saving | The save took longer than torchrun's 30 s grace (slow disk, huge checkpoint) | Send SIGTERM to the workers too (`pkill -TERM -f mygpt.train` reaches both); resume from the interval checkpoint |
| Loss jumps by 0.1+ right after a resume | Data position or optimizer state not restored (e.g. you loaded weights manually into a new run) | Resume only through `mygpt.train` auto-resume from a complete checkpoint |
| `rclone lsf` shows no checkpoints in `ckpt/<run>/` | `ckpt_remote` empty or misspelled, or rclone not configured in this shell | `grep ckpt_remote runs/<run>.out`; check `rclone lsf r2:mygpt/`; never start a long run without seeing the first upload |
| The log in the bucket is shorter than the run | The run was SIGKILLed; the log is uploaded with each checkpoint and on a clean end or stop, so it ends near the newest checkpoint | Expected: the resume re-runs those steps; on a new box `sync.sh pull-ckpt` before relaunching |
| Disk full while saving | Checkpoints of several runs plus shards on a small disk | `keep_last` limits one run; remove old run dirs that are safely in the bucket |

## Artifact

- `reports/drill-05.2.md` committed: observed messages, resume steps, `--compare` result, chosen `ckpt_every` with arithmetic, and the four-step rollback procedure in your own words.
- `drill.sh` is throwaway; nothing else is kept from the drill. The rehearsal box is terminated.
- `budget.md` ledger line:

  ```text
  | 2026-11-13 | 05.2 | RunPod 2x RTX 4090 Community, ~1 h at $0.68/h, kill-and-resume drill | ~$0.70 | $____ |
  ```
