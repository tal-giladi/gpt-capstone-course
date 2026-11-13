---
id: "06.1"
module: 6
minutes: 20
practice_minutes: 90
prerequisites: ["03.3", "04.3", "05.2"]
objectives:
  - List what must be held fixed for an ablation to be fair and check it in the project's configs.
  - Measure run-to-run noise with two control seeds and turn it into a decision threshold.
  - Price every planned ablation on 1×H100 from your 04.3 numbers and fit the plan into about $60.
  - Run an unattended, resumable ablation queue and produce the first version of the ablation table.
volatility: implementation
sources:
  - title: "Wen, Hall, Ma, Liang: Fantastic Pretraining Optimizers and Where to Find Them (arXiv 2509.02046, ICLR 2026)"
    url: https://arxiv.org/abs/2509.02046v1
  - title: "Keller Jordan et al., modded-nanogpt (speedrun; run-to-run variance and record rules)"
    url: https://github.com/KellerJordan/modded-nanogpt
  - title: "Hägele et al.: Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations (arXiv 2405.18392)"
    url: https://arxiv.org/abs/2405.18392v3
  - title: "RunPod GPU pricing"
    url: https://www.runpod.io/pricing
  - title: "Vast.ai pricing"
    url: https://vast.ai/pricing
  - title: "Verda (formerly DataCrunch) GPU pricing"
    url: https://verda.com/pricing
last_verified: "2026-11-13"
---

# 06.1 · Designing a fair ablation

Your baseline proves you can train GPT-2. The model you publish should be better than GPT-2, and every change you make has to earn its place with evidence: you change one thing, train with everything else identical, and keep the change only if the difference is larger than the noise. With about $60 for all ablations, each run is a short 1B-token training of a ~110M model on one H100, around 35 minutes and $1.60. This lesson sets the rules that make those runs comparable, measures how noisy they are, and puts the whole plan on a budget.

## Goal

You have run the control configuration twice with different seeds on 1×H100, built the first version of `results/ablations.md` with the noise band and the decision threshold that follows from it, and written a priced ablation plan into `budget.md`. The queue runner and table tools are in your repo and tested.

## What you need

- Your own-tokenizer shards `tok/bpe32k-100BT-30/` from [03.2](../module-03/lesson-02.md) (decontaminated in [03.3](../module-03/lesson-03.md)), the `ablation_base` speed from [04.3](../module-04/lesson-03.md), and the recovery habits from [05.2](../module-05/lesson-02.md).
- The [Module 6 lab files](../../labs/module-06/) copied into `tools/` of your project repo and committed: `ablations.txt`, `run_ablations.sh`, `ablation_table.py`.
- About 2 hours, ~75 minutes of it on a rented GPU (unattended for most of it).
- **Expected cost: $3–4** (about 1.3 h of 1×H100 SXM Community at $2.69/h).

## Choices and costs

### What "fair" means here

| Hold fixed | How the project does it | What breaks if you don't |
|---|---|---|
| Token budget | `train_tokens: 1_000_000_000` = 1,907 steps of 524,288 tokens | A run with more tokens wins for that reason alone |
| Data and its order | Same `data_dir`; the loader walks the shards in file order with no shuffling, so the same batch size gives every run the identical token stream | Different data order adds noise; different data changes the question |
| Batch size and schedule | Same `total_batch_tokens`, warmup, cosine to 10% | The LR schedule interacts with every change you test |
| Evaluation | Same val split, same `eval_tokens`, same metric (`val_bpb`) at the same step | A "better" number from an easier or shorter eval |
| Seed | `seed: 1337` for every ablation; the seed changes only the initial weights | You would be measuring luck |
| Everything not under test | One override per run on top of `configs/ablation_base.yaml` | Two changes at once cannot be separated |

Things that may differ: GPU type, `compile`, micro-batch size and speed. They change how long a run takes, not what it learns, which is why you compare at **matched tokens**, never at matched wall-clock time.

### Compare at the end of the schedule

Different settings produce curves that cross. A higher learning rate often looks worse early and better after decay, and a WSD schedule looks worse than cosine until its final decay phase. Wen et al. (2025) show optimizer rankings flipping during LR decay and warn that comparing intermediate checkpoints is misleading. So the table compares the **final** evaluation (step 1,907) of every run, and a run that did not finish is reported as unfinished, never compared at an earlier point.

### Why val bpb

The ablations use your own 32,768-token BPE, the baseline used GPT-2's tokenizer. Loss per token is not comparable across tokenizers (a tokenizer with longer tokens has a higher loss per token for the same text), but bits per byte of text is:

$$\text{bpb} = \frac{\text{val\_loss}}{\ln 2} \cdot \frac{\text{val tokens}}{\text{val bytes}}$$

Within the ablations (one tokenizer) bpb and loss rank runs identically; bpb also lets you put the ablation runs and the baseline on one axis.

### Noise and the decision threshold

Two runs that differ only in the seed do not end at the same loss. The modded-nanogpt speedrun reports run-to-run standard deviations of up to about 0.005 in val loss for similar models, which is about $0.005 / 0.693 / 4.5 \approx 0.0016$ bpb at ~4.5 bytes per token (ESTIMATE; your controls measure it). Rule for this course:

> A change counts as better or worse only if its difference from the control mean is more than **twice the spread** between the control seeds.

`ablation_table.py` applies this rule. Two seeds give a rough spread; if the budget allows, a third control seed makes it more trustworthy.

### Small-scale results are evidence, not proof

A 1B-token, ~110M-parameter run is cheap, but effects can shrink at 350M and 20B tokens. Wen et al. found optimizer speedups of 1.3–1.4× over a well-tuned AdamW below 1B parameters falling to about 1.1× at 1.2B. Prefer changes that help here **and** are standard at scale (RoPE, RMSNorm and SwiGLU are in Llama and Qwen), and be sceptical of a large win that nothing else in the literature shows.

### What the ablations cost (checked 2026-11-13)

With 04.3's example `ablation_base` speed of 520,000 tokens/s per GPU (use yours):

$$t = \frac{10^{9}}{520{,}000} = 1{,}923\ \text{s} \approx 32\ \text{min} \;+\; \sim 3\ \text{min compile and evals} \approx 35\ \text{min}$$

| Block | Runs | GPU-h | Cost at $2.69/h (RunPod Community H100 SXM) |
|---|---|---|---|
| 06.1 controls (2 seeds) | 2 | 1.2 | $3.15 |
| 06.2 architecture (6 single changes + 2 exportable bundles) | 8 | 4.7 | $12.60 |
| 06.3 optimizer and schedule (LR ×2, WSD, Muon ×3, Muon+WSD) | 7 | 4.1 | $11.00 |
| Box setup, data pull, idle (3 sessions × ~20 min) | | 1.0 | $2.70 |
| Margin: a third seed, two reruns, one extra idea | ~4 | 2.3 | $6.20 |
| **Total** | **~21** | **~13.3** | **~$36** of the $60 plan line |

| Alternative box | Price | Trade-off |
|---|---|---|
| Vast.ai H100 SXM (verified host) | from ~$1.87/h, median ~$2.40 | Cheaper; host quality varies; bandwidth billed per TB |
| RunPod Community H100 PCIe | $1.99/h | Cheaper per hour, slower per token; use only if 04.3 showed a lower $ per 1B tokens |
| Verda H100 SXM spot | $2.595/h ($5.189 on-demand) | EU datacenters; spot can be reclaimed, like any interruptible box |
| Interruptible / spot H100 | ~50% off | An interruption loses the unfinished run (~$1.50); the queue skips finished runs and resumes on a new box |

> [!NOTE]
> Prices checked 2026-11-13 on the RunPod pricing page and the Vast.ai offer list. Re-check before renting and put the per-run cost with your price into `budget.md`.

## Do this

### 1. Install the tools and test them on CPU (10 min, laptop)

Copy the Module 6 lab files into `tools/`, then run the queue and the table on the smoke config so you see the whole flow in a minute:

```bash
cd ~/mygpt && . .venv/bin/activate
CONFIG=configs/smoke_cpu.yaml RUNS=runs/queue-test MYGPT_REMOTE=r2:mygpt/tmp/queue-test \
  bash tools/run_ablations.sh tools/ablations.txt
python tools/ablation_table.py "runs/queue-test/*" --queue tools/ablations.txt
DRY_RUN=1 bash tools/run_ablations.sh tools/ablations.txt      # the real commands, printed only
git add tools/ && git commit -m "Module 6 ablation tools" && git push
```

`MYGPT_REMOTE` points the test at `tmp/` in your bucket: without it, the smoke runs would upload `DONE` markers under the real control names and the real queue would later skip them. The smoke table shows two controls, their spread, and the threshold. The dry run prints two `python -m mygpt.train --config configs/ablation_base.yaml train.seed=... train.out_dir=runs/abl_control_s13xx ...` commands.

### 2. Rent the ablation box (10 min)

Deploy **1× H100 SXM** on Community Cloud from `mygpt-gpu`, arm the hard stop, set up and start the idle guard:

```bash
nohup bash -c 'sleep 3h; runpodctl pod stop $RUNPOD_POD_ID' >/tmp/hardstop.log 2>&1 &
git clone https://github.com/<you>/<your-repo>.git ~/mygpt && bash ~/mygpt/tools/setup_box.sh
cd ~/mygpt && . .venv/bin/activate
tmux new -d -s guard "bash scripts/autoshutdown.sh --idle-minutes 20 --stop-cmd 'runpodctl pod stop $RUNPOD_POD_ID' 2>&1 | tee /tmp/guard.log"
```

The idle guard is what makes an unattended queue safe: when the last run ends, the pod stops itself 20 minutes later. On Vast.ai or Lambda use that provider's `--stop-cmd` from 04.2 in both lines and load `r2.env`/`secrets.env` yourself; the queue only needs the remote `r2` and, for W&B, `WANDB_API_KEY`.

### 3. Pull the first 20 shards (2 min)

One ablation reads ~1B tokens, about 10 shards of 100M tokens. Pull 20 (~4 GB) so no run wraps around:

```bash
rclone copy r2:mygpt/tok/bpe32k-100BT-30/ data/tok/bpe32k-100BT-30 \
  --include "meta.json" --include "tokenizer.json" --include "val_*" \
  --include "train_0000[0-1][0-9].bin" --multi-thread-streams 8 --progress
```

### 4. Start the control runs (2 min, then ~70 min unattended)

```bash
tmux new -s queue
WANDB_PROJECT=my-gpt-ablations bash tools/run_ablations.sh tools/ablations.txt 2>&1 | tee -a runs/queue.log
```

Watch the first minutes of `abl_control_s1337`: the header must read `[train] params 111.0M | world 1 | micro 64x1024 | accum 8 | steps 1907 | ... | dtype bf16` (111M, not 124M: the 32k vocabulary has a smaller embedding than GPT-2's 50k), and tokens/s must be close to your 04.3 `ablation_base` number. Then detach and leave.

### 5. Build the table (5 min)

When both controls show `ok` in the queue summary:

```bash
mkdir -p results
python tools/ablation_table.py "runs/abl_*" --queue tools/ablations.txt --price-per-gpu-hour 2.69 | tee results/ablations.md
```

### 6. Decide on the noise and write the plan (10 min)

- Spread below ~0.005 bpb: two seeds are enough; the threshold is twice the spread.
- Spread above ~0.01 bpb: something is not identical between the runs (different data, a resumed run with a changed setting). Find it before running anything else.
- If you want a firmer threshold, add `abl_control_s1339 train.seed=1339` to the queue (+$1.60).

Write the plan into `budget.md`: the runs of 06.2 and 06.3 with your per-run cost, and the running total against $60.

### 7. Keep going or stop

If you go straight on to 06.2, leave the box running and continue there. Otherwise terminate it: the logs are in `r2:mygpt/ckpt/abl_control_*/`, and on a new box the queue skips both controls because their `DONE` markers are in the bucket.

## How you know it worked

- `runs/queue.log` ends with both controls `ok`, each in roughly 35 minutes (your 04.3 speed decides).
- `results/ablations.md` starts with a line like `Metric val_bpb at step 1,907 (1.00B tokens); control mean 1.1xxx over 2 run(s), seed spread 0.00xx, effect threshold 0.00xx.` Expect a control val_bpb around 1.05–1.20 (ESTIMATE: a ~110M model after 1B tokens; your 10B-token baseline was below 1.0) and a spread of a few thousandths.
- `rclone lsf r2:mygpt/ckpt/abl_control_s1337/` lists `DONE`, `log.jsonl` and `train.out`.
- `budget.md` holds the ablation plan with per-run cost and a total under $60.

## When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| Header shows `params 124.5M` or `vocab 50304` | You ran `baseline_124m.yaml` instead of `ablation_base.yaml` | The queue uses `CONFIG` (default `configs/ablation_base.yaml`); unset a leftover `CONFIG` variable |
| `ValueError: shards use vocab 50257 > model.vocab_size 32768` | `data_dir` points at the GPT-2 shards | Pull `tok/bpe32k-100BT-30` into `data/tok/bpe32k-100BT-30` |
| The two controls differ by more than ~0.01 bpb | Not identical: one resumed with a different setting, or a shard was missing so one run wrapped around | Compare the two `train.out` headers and `ls data/tok/bpe32k-100BT-30`; rerun the odd one in a new name |
| Tokens/s far below 04.3 | `compile` off, a slow host, or another process on the GPU | Check the header and `nvidia-smi`; change host if it persists |
| Queue stopped after the first run and the pod is gone | The hard stop or idle guard fired (for example, the queue waited on a prompt) | Rent again and rerun the same command: finished runs are skipped, the unfinished one starts over (ablations do not upload checkpoints) |
| `ablation_table.py` says `no control run matches 'abl_control'` | Glob did not match, or you renamed the controls | Quote the glob (`"runs/abl_*"`), or pass `--control name1,name2` |

## Artifact

- `tools/ablations.txt`, `tools/run_ablations.sh`, `tools/ablation_table.py` committed.
- `results/ablations.md` (version 1: the two controls, spread and threshold) committed; logs in `r2:mygpt/ckpt/abl_control_*/`; W&B project `my-gpt-ablations`.
- `budget.md`: the priced ablation plan and the ledger line:

  ```text
  | 2026-11-13 | 06.1 | RunPod 1x H100 SXM Community, ~1.3 h at $2.69/h, 2 control runs (1B tokens each) | ~$3.50 | $____ |
  ```
