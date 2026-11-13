---
id: "07.1"
module: 7
minutes: 25
practice_minutes: 75
prerequisites: ["06.4", "05.2"]
objectives:
  - Choose the provider and pricing model for a 5–10 hour 8×H100 job from today's prices, availability and failure risk, and justify the choice in budget.md.
  - Run a scripted pre-flight check (frozen commit, data, bucket, W&B, GPUs, disk, stop guards, money) and fix every FAIL before the node trains.
  - Launch the main run with phone alerts and judge in the first 15 minutes whether it is the run you priced.
volatility: implementation
sources:
  - title: "RunPod GPU pricing"
    url: https://www.runpod.io/pricing
  - title: "RunPod docs: Pod pricing (per-second billing, container, volume and network volume storage)"
    url: https://docs.runpod.io/pods/pricing
  - title: "RunPod docs: Savings plans"
    url: https://docs.runpod.io/pods/savings-plans
  - title: "Lambda pricing (on-demand and 1-Click Clusters)"
    url: https://lambda.ai/service/gpu-cloud/reserved-cloud-pricing
  - title: "W&B docs: alerts and notifications (run finished, run crashed)"
    url: https://docs.wandb.ai/support/models/articles/how-do-i-set-up-wandb-alerts-and-notifications
  - title: "ntfy docs: publishing a notification with one HTTP request"
    url: https://docs.ntfy.sh/publish/
  - title: "PyTorch: torchrun (elastic launch)"
    url: https://docs.pytorch.org/docs/stable/elastic/run.html
last_verified: "2026-11-13"
---

# 07.1 · Pre-flight checklist and launch

This is the most expensive hour of the course: in it you rent an 8×H100 node for somewhere between five and ten hours and start the run that produces your model. Everything about the run was decided in 06.4 and frozen at the tag `recipe-v1`; what is left is the part pilots do before take-off. You check the money, choose where to rent on the day, verify on the node itself that the commit, the data, the bucket, the logging, the GPUs and the stop guards are what you think they are, and only then press enter. A script does most of the checking, so that "I was sure the checkpoint path was set" is never the reason you lose $200.

## Goal

The main run `main_350m` is training on an 8×H100 SXM node from the commit tagged `recipe-v1`. `tools/preflight.py` printed `READY` on that node before launch, an idle guard and a hard stop are armed, `tools/watch_run.py` sends alerts to your phone, the first checkpoint is visible in `r2:mygpt/ckpt/main_350m/`, and `reports/main-run-log.md` records the provider, price, start time, prediction and the first-15-minutes checks.

## What you need

- The frozen recipe from [06.4](../module-06/lesson-04.md): `configs/main_350m.yaml` at tag `recipe-v1` with `ckpt_remote: r2:mygpt/ckpt/main_350m` and `wandb_project: my-gpt`, and the main-run estimate in `budget.md`.
- The `mygpt-8gpu` RunPod template and `launch/baseline_8xh100.sh` from [05.1](../module-05/lesson-01.md); your recovery procedure from [05.2](../module-05/lesson-02.md).
- Your measured per-GPU tokens/s for `main_350m` ([04.3](../module-04/lesson-03.md)) and the 8-GPU scaling efficiency ([05.3](../module-05/lesson-03.md)).
- The [Module 7 lab files](../../labs/module-07/) copied into `tools/` and committed (`preflight.py`, `watch_run.py`, `RUN_LOG.template.md`).
- The free ntfy app on your phone (Android or iOS) for alerts.
- Provider credit of at least 1.5× your estimate (a RunPod pod is stopped when the balance reaches $0, and so is your run).
- About 1.5 hours of your time today; the run itself takes 5–10 hours.
- **Expected cost of this lesson: ~$20** (the first ~35 minutes of node time: setup, preflight, the first 15 minutes of training). The whole run, ~$187 at the example speed, is billed in 07.2 and 07.3 and reconciled in 07.3.

## Choices and costs

### Predict the bill once more, with today's price

From 06.4, with your $r$ (per-GPU tokens/s for `main_350m`) and $s$ (scaling efficiency):

$$t_{\text{train}} = \frac{2\times10^{10}}{8 \cdot r \cdot s} \qquad \text{billed} \approx 1.25 \cdot t_{\text{train}}$$

The factor 1.25 (what `budget_calc.py` adds) covers setup, `torch.compile`, 77 val evals, 20 HellaSwag evals, 39 checkpoint saves, one resume and the close-out. With 04.3's example $r = 175{,}000$ and $s = 0.93$: $t_{\text{train}} = 15{,}361$ s = 4.3 h, so about 5.3 billed hours, the $187 from 06.4. If your $r$ was lower (a slower host, no `torch.compile`, a 2,048 context) the same formula gives 8–10 hours; that is still one run, but it changes the hard-stop time and how much margin you need.

### Where to rent it (checked 2026-11-13)

| Option | 8×H100 SXM price | ~5.3 h (example speed) | ~9 h (slower host) | Trade-offs |
|---|---|---|---|---|
| **RunPod Secure Cloud, on-demand** | $35.12/h ($4.39/GPU-h) | ~$186 | ~$316 | Your template, secrets and stop commands already work; per-second billing; 8× availability varies by hour |
| Lambda on-demand | $31.92/h until 2026-11-15, **$36.72/h from 2026-11-16** | ~$195 (new price) | ~$330 | Reliable whole nodes; a shutdown does not stop billing, terminate via dashboard or API |
| Vast.ai verified datacenter host, on-demand | ~$21–24/h (varies per host) | ~$111–127 | ~$189–216 | Cheapest; host quality varies, so check the host's reliability score, bandwidth price and NVLink before renting |
| RunPod Community Cloud | H100 SXM from $2.69/GPU-h, but 8-GPU community nodes are rare | | | Fine for 1-GPU work; do not plan the main run on finding one |
| Spot / interruptible (RunPod spot, Vast interruptible) | ~40–60% cheaper | | | Can be taken away with little or no warning; each interruption costs ~7 min of lost steps plus 15–30 min to get going again, and an 8-GPU replacement may not exist. Not for this run |
| Reserved: RunPod savings plan | ~15–20% off (older RunPod figures; current rates only in the console) | | | Month-long, non-refundable commitment per GPU type: you would pay for ~720 hours to save on ~6 |
| Reserved: Lambda 1-Click Clusters | $5.54–6.16/GPU-h | | | 16+ GPUs and a 2-week minimum. Built for a different project size |

ESTIMATE: billed hours include setup, evals, checkpoints and one resume, from 06.4's example speed; use your own numbers.

**Recommended:** on-demand, single 8×H100 SXM node, on the provider whose stop/terminate path you have already tested: RunPod Secure Cloud. Fallback, in this order: Lambda on-demand, a verified Vast.ai datacenter host (≥ 99% reliability, all-NVLink, CUDA 13 driver). Rent it at a time when you can be at the keyboard for the first hour and reachable by phone for the rest.

> [!NOTE]
> Prices from the RunPod and Lambda pricing pages and the Vast.ai offer list on 2026-11-13; Lambda's increase takes effect on 2026-11-16. Savings-plan percentages come from RunPod's launch announcement and may have changed. Re-check the live price and 8-GPU availability on launch day and write both into `budget.md` before you deploy.

### Alerts: what tells you something is wrong

| Signal | Tool | Cost | Covers |
|---|---|---|---|
| Phone push from `tools/watch_run.py --ntfy <topic>` | ntfy.sh public server + ntfy app | $0 | Stale log (hang, dead node), NaN, unrecovered loss spike, sustained slowdown, cost overrun |
| W&B "Run finished" / "Run crashed" email | W&B User Settings → Alerts | $0 (free tier) | The run ended, or stopped sending heartbeats (process killed, node gone) |
| Provider low-balance email | Provider billing settings, where offered | $0 | Running out of credit before the run ends |
| You, looking at the W&B charts | Laptop or phone browser | $0 | Everything else: the shape of the curves |

The watcher runs on the node, so if the node dies it falls silent. W&B's crash email is the alert that comes from outside the node; turn it on.

> [!WARNING]
> Anyone who knows an ntfy.sh topic name can read and post to it. Use a long random topic (`mygpt-` plus 12 random characters) and never put secrets, keys or bucket names in alert text. Your watcher sends only step numbers and metrics.

> [!WARNING]
> From the moment the node is running it costs **$21–37 an hour**, whether it trains or not. A forgotten finished node costs $500–900 a day. Arm both stops before you do anything else on it.

## Do this

### 1. The day before: money and the go decision (15 min, laptop)

Update `budget.md` with every actual so far, then re-run 06.4's calculation with today's price:

```bash
python tools/budget_calc.py --tps-baseline <05.3 per-GPU tok/s> --tps-ablation <04.3 tok/s> \
  --tps-main <04.3 main tok/s> --price-1gpu 2.69 --price-8gpu <today's 8x per-GPU price> --scaling <05.3 s>
```

Write the launch block in `budget.md`: provider, price, predicted billed hours and cost, money left, and the ratio. Go only if money left ≥ 1.25 × the estimate (06.4's rule); otherwise apply 06.4's shrink options, commit and re-tag before renting anything. Top up the provider credit to at least 1.5 × the estimate.

### 2. Set up the alerts (10 min, laptop and phone)

1. W&B → avatar → **User Settings** → **Alerts**: enable **Run finished** and **Run crashed** by email.
2. Make a topic name and subscribe to it in the ntfy app:

   ```bash
   echo "mygpt-$(python -c 'import secrets; print(secrets.token_hex(6))')"
   curl -d "test from laptop" ntfy.sh/<your-topic>        # the phone should buzz within seconds
   ```

3. Put the topic name in your password manager, not in the repo.

### 3. Copy the tools, write the launch script, check the tag (10 min, laptop)

```bash
cp <course download>/labs/module-07/{preflight.py,watch_run.py,RUN_LOG.template.md} tools/
cp tools/RUN_LOG.template.md reports/main-run-log.md
cat > launch/main_8xh100.sh <<'EOF'
#!/usr/bin/env bash
# 07.1: the main run on one 8xH100 SXM node. Run inside tmux from the repo root.
# Everything is in configs/main_350m.yaml at tag recipe-v1; any override added later is a recorded decision (07.2).
set -euo pipefail
mkdir -p runs
torchrun --standalone --nproc_per_node=8 -m mygpt.train --config configs/main_350m.yaml \
  2>&1 | tee -a runs/main_350m.out
EOF
git add tools/ launch/main_8xh100.sh reports/main-run-log.md && git commit -m "Main run: launch script and tools"
git push && git log --oneline -1 && git merge-base --is-ancestor recipe-v1 HEAD && echo "contains recipe-v1"
grep -E "ckpt_remote|wandb_project|run_name|train_tokens" configs/main_350m.yaml
```

The `grep` must show `ckpt_remote: r2:mygpt/ckpt/main_350m`, `wandb_project: my-gpt`, `run_name: main_350m` and your token count. If `ckpt_remote` is empty, stop: fix it in a commit, not on the node.

### 4. Deploy and arm both stops (5 min)

RunPod → **Secure Cloud** → **H100 SXM**, GPU count **8**, template `mygpt-8gpu` (150 GB container disk), On-Demand → **Deploy**. Write the start time into the run log. In the web terminal or SSH, first:

```bash
nohup bash -c 'sleep 11h; runpodctl pod stop $RUNPOD_POD_ID' >/tmp/hardstop.log 2>&1 &
```

Use twice your predicted billed hours, rounded up (11 h for a 5.3 h prediction). On Lambda or Vast.ai, replace the stop command with the provider's terminate/stop API call you tested in 04.2.

### 5. Set up the node and pull the data (15 min)

```bash
git clone https://github.com/<you>/<your-repo>.git ~/mygpt && bash ~/mygpt/tools/setup_box.sh 2>&1 | tee /tmp/setup.log
cd ~/mygpt && . .venv/bin/activate
tmux new -d -s guard "bash scripts/autoshutdown.sh --idle-minutes 20 --stop-cmd 'runpodctl pod stop $RUNPOD_POD_ID' 2>&1 | tee /tmp/guard.log"
bash scripts/sync.sh pull-tok bpe32k-100BT-30 data/tok/bpe32k-100BT-30     # ~45 GB, 3–10 min
python -c "from mygpt.evaluate import load_hellaswag; print(len(load_hellaswag()))"   # 10042
```

`setup_box.sh` must end with `gpu_check.py` printing `HEALTHY` for 8 GPUs. If one GPU is slow or missing, terminate and redeploy now: ten minutes of a bad node is $6, ten hours is $350.

### 6. Pre-flight (5 min)

```bash
python tools/preflight.py --config configs/main_350m.yaml --gpus 8 --tag recipe-v1 \
  --budget-left <money left> --estimate <predicted cost> | tee reports/preflight-main.txt
```

Every line must be `PASS`, or a `WARN` you understand and write into the run log. Typical first-time `FAIL`s and their fixes are in the table at the end. Do not launch on `NOT READY`.

### 7. Launch (2 min)

```bash
tmux new -s train
bash launch/main_8xh100.sh
```

Detach (<kbd>Ctrl</kbd>+<kbd>b</kbd> <kbd>d</kbd>) and start the watcher in its own window:

```bash
tmux new -d -s watch "python tools/watch_run.py runs/main_350m --target-tokens 20e9 --price-per-hour <price> \
  --started <node start, e.g. 2026-11-17T09:05> --plan-hours <predicted billed h> --interval 120 \
  --ntfy <your-topic> 2>&1 | tee -a runs/watch.out"
```

Your phone gets `watch_run started`. Open the W&B run `main_350m` in project `my-gpt` on your laptop.

### 8. The first 15 minutes (15 min)

Stop the run (<kbd>Ctrl</kbd>+<kbd>c</kbd> in the train window, then terminate) if any check fails; each is cheaper to fix now than at hour five. Fill in the run log's table as you go:

1. Header: `[train] params 337.2M | world 8 | micro 32x1024 | accum 2 | steps 38146 | device cuda:0 | dtype bf16` (your parameter count and steps if your recipe differs).
2. Loss: about 10.4 at step 1 (a uniform guess over 32,768 tokens is $\ln 32{,}768 = 10.40$), below ~6 by step 300, falling smoothly through warmup (476 steps).
3. Speed after compile: `tail -3 runs/watch.out` shows node tokens/s. It should be about $8 \cdot r \cdot s$ (example: $8 \times 175{,}000 \times 0.93 = 1.30$M tokens/s, ~0.40 s per step). Below 85% of that: check `nvidia-smi` for a GPU below ~100% utilization or at a lower clock, then stop and redeploy.
4. Step 1,000 (about 7 minutes in at the example speed): `rclone lsf r2:mygpt/ckpt/main_350m/ --format "sp"` lists `ckpt_00001000.pt` (~2.9 GB) and `log.jsonl`. No upload means no recovery: stop and fix.
5. The ETA in the watcher line and the projected cost match your prediction within ~10%.

Write `first 15 minutes OK` (or what you changed) into the run log, commit it from the node or your laptop, and leave the run alone. 07.2 tells you what to watch next.

## How you know it worked

- `reports/preflight-main.txt` ends with `READY: 0 FAIL` and the commit it names is at or after `recipe-v1`.
- `tmux ls` on the node shows three sessions: `guard`, `train`, `watch`; `/tmp/hardstop.log` exists and `ps aux | grep "sleep 11h"` shows the hard stop.
- The watcher's line shows node tokens/s within ~15% of $8 \cdot r \cdot s$ and an ETA within ~10% of your prediction; your phone received the start message.
- `r2:mygpt/ckpt/main_350m/` contains `ckpt_00001000.pt` of ~2.9 GB and `log.jsonl`; W&B shows the run with loss, val loss and tokens/s.
- `reports/main-run-log.md` has the launch table and the first-15-minutes table filled in.

## When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| No 8×H100 SXM available on RunPod Secure | Capacity changes through the day | Check again in an hour, or use the fallback (Lambda; a verified Vast.ai host); there, pass the provider stop command to the hard stop and `autoshutdown.sh` |
| `FAIL git ... uncommitted changes` | You edited a file on the node, or `setup_box.sh` changed a tracked file | `git diff`; commit the change on your laptop if it is intended (and say why in the run log), else `git checkout -- <file>` |
| `FAIL git tag recipe-v1 missing` | Tags not pushed, or cloned a branch without the tag | `git fetch --tags`; on the laptop `git push --tags` |
| `FAIL data ... incomplete` or shard count differs from the bucket | The pull was interrupted or the disk filled | Rerun `sync.sh pull-tok` (copy skips complete files); check `df -h` |
| `FAIL data train_tokens ... would repeat data` | `train_tokens` larger than the decontaminated train split | Back to 06.4: set ~97% of the available tokens, commit, re-tag |
| `FAIL gpus ... already in use` | Another process (an old run, a notebook kernel) holds GPU memory | `nvidia-smi` shows the PID; kill it, or redeploy if it is not yours |
| `FAIL gpus no NVLink` | A PCIe node behind the same label | Terminate and redeploy (or choose another host); all-reduce over PCIe costs a visible share of every step |
| `WARN bucket ... already has checkpoints` | A previous attempt uploaded to `ckpt/main_350m/` | If that attempt is the run you want to continue: fine, it resumes. If not: change `run_name`, `out_dir` and `ckpt_remote` in a commit; never resume from a run you do not trust |
| Node tokens/s 20%+ below prediction from the first minutes | A slow GPU, or your benchmark used a different `micro_batch_size`, context or `compile` setting | Compare the header with your 04.3 benchmark settings; `nvidia-smi` for clocks; stop now rather than pay the difference for hours |
| ntfy test reaches nothing | App not subscribed to the exact topic, or the box cannot reach ntfy.sh | Subscribe again; `curl -d test ntfy.sh/<topic>` from the node; if blocked, rely on W&B emails and check more often |

## Artifact

- In git: `launch/main_8xh100.sh`, `tools/preflight.py`, `tools/watch_run.py`, `reports/preflight-main.txt`, and `reports/main-run-log.md` with the launch and first-15-minutes tables.
- In the bucket: `ckpt/main_350m/ckpt_00001000.pt` and `log.jsonl` (the run keeps adding to them).
- `budget.md`: the launch block (provider, price, start time, prediction, money left, ratio, commit) and an open ledger line that 07.3 completes:

  ```text
  | 2026-11-17 | 07.1–07.3 | RunPod Secure 8x H100 SXM, main run 337M x 20B, predicted 5.3 h at $35.12/h | ~$187 | $____ |
  ```
