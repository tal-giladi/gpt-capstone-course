---
id: "04.3"
module: 4
minutes: 20
practice_minutes: 60
prerequisites: ["03.3", "04.2"]
objectives:
  - Estimate training compute with 6·N·D and convert it to GPU-hours and dollars at a given MFU.
  - Measure tokens/s, MFU and dollars per billion tokens for the baseline, ablation and main-run configs on one H100.
  - Convert the measurements into a revised budget for baseline, ablations and main run with scaling and overhead factors.
  - Decide from the revised budget whether the 350M × 20B main run stays as planned or shrinks.
volatility: implementation
sources:
  - title: "Scaling Laws for Neural Language Models (Kaplan et al., 2020): C ≈ 6ND"
    url: https://arxiv.org/abs/2001.08361
  - title: "NVIDIA H100 specifications (BF16 tensor-core peak)"
    url: https://www.nvidia.com/en-us/data-center/h100/
  - title: "RunPod GPU pricing"
    url: https://www.runpod.io/pricing
  - title: "Lambda pricing"
    url: https://lambda.com/pricing
last_verified: "2026-11-13"
---

# 04.3 · Benchmarking throughput and fixing the budget

The course budget was written with an assumed price and an assumed speed. You now have a real GPU and real code, so you replace both assumptions with measurements: thirty training steps per config tell you tokens per second, and tokens per second times price tells you what each run will cost. If the numbers do not fit under the $500 cap with a healthy margin, this is the cheapest moment to change the plan.

## Goal

You end with a benchmark table (tokens/s per GPU, MFU, dollars per billion tokens) for `baseline_124m`, `ablation_base` and `main_350m` on a 1×H100 SXM, a revised budget table in `budget.md` produced from those numbers, and a written decision: keep the ~350M × 20B main run, or shrink it, with the arithmetic to back it.

## What you need

- The `mygpt-gpu` template and tools from [04.2](lesson-02.md); token shards and `meta.json` in your bucket from [03.3](../module-03/lesson-03.md) (the benchmark uses the configs as they will really run).
- Your provider price table in `budget.md` from [04.1](lesson-01.md).
- About 80 minutes, ~45 of them on the GPU.
- **Expected cost: $2–3** on Community Cloud H100 SXM ($2.69/h); add ~$2 if you also benchmark a B200 (optional step 5).

## Choices and costs

### First, the estimate you are about to check

Training compute for a dense transformer is about $C \approx 6ND$ FLOPs ($N$ parameters, $D$ training tokens: 2 for the forward pass, 4 for the backward). Attention adds $6 \cdot n_{\text{layer}} \cdot T \cdot d$ per token, about 7% at context $T = 1024$ for both models (14% at 2048), so 6ND is a slight underestimate.

$$C_{\text{baseline}} = 6 \times 1.24\times10^{8} \times 10^{10} = 7.44\times10^{18}\ \text{FLOP}$$

$$C_{\text{main}} = 6 \times 3.5\times10^{8} \times 2\times10^{10} = 4.2\times10^{19}\ \text{FLOP}$$

One H100 SXM delivers $989\times10^{12}$ dense BF16 FLOP/s at 100% MFU (model FLOPs utilization). Real training reaches 30–50%. GPU-hours $= C / (\text{MFU} \times 989\times10^{12} \times 3600)$:

| Run | MFU 30% | MFU 40% | MFU 50% | Cost at $4.00/GPU-h (40%) | Wall clock on 8 GPUs (40%) |
|---|---|---|---|---|---|
| Baseline 124M × 10B | 7.0 GPU-h | 5.2 GPU-h | 4.2 GPU-h | $21 | 0.65 h |
| Main 350M × 20B | 39.3 GPU-h | 29.5 GPU-h | 23.6 GPU-h | $118 | 3.7 h |

The same numbers as speed: at 40% MFU one H100 trains the 124M model at $0.4 \times 989\times10^{12} / (6 \times 1.24\times10^{8}) \approx 532{,}000$ tokens/s and the 350M model at about 188,000 tokens/s. These are the numbers your benchmark confirms or corrects.

### From tokens/s to dollars per billion tokens

With $r$ = measured tokens/s per GPU, $p$ = price per GPU-hour and $s$ = multi-GPU scaling efficiency (0.9 is a planning value for 8×H100 SXM with NVLink; Module 5 measures it):

$$\text{USD per }10^9\text{ tokens} = \frac{p \times 10^9}{3600 \times r \times s}$$

Example: $r = 175{,}000$, $p = 4.39$, $s = 0.9$ gives $4.39 \times 10^9 / (3600 \times 157{,}500) \approx 7.74$ USD per billion tokens, so 20B tokens cost about $155 before overhead.

### What to benchmark on

| GPU (RunPod, checked 2026-11-13) | $/GPU-h | Why benchmark it |
|---|---|---|
| **H100 SXM** Community $2.69 / Secure $4.39 | the default | The chip of your 8-GPU nodes; required |
| B200 Community $5.98 / Secure $7.99 | optional | 2.3× the dense peak at ~1.8× the price; only a measurement shows if a 350M model can use it |
| H100 PCIe Community $1.99 / Secure $2.89 | optional | Cheaper per hour for 1-GPU ablations, if its dollars per billion tokens come out lower |

> [!NOTE]
> Use the price you will actually pay for each run, not this table: the 1-GPU price for ablations and the 8-GPU node's per-GPU price for the baseline and main run. Re-check both on the live pages (Lambda's prices rise on 2026-11-16) and update `budget.md`.

## Do this

### 1. Launch and set up (10 min)

Deploy a 1×H100 SXM pod from your `mygpt-gpu` template, arm the hard stop as in 04.2, then:

```bash
nohup bash -c 'sleep 2h; runpodctl pod stop $RUNPOD_POD_ID' >/tmp/hardstop.log 2>&1 &
git clone https://github.com/<you>/<your-repo>.git ~/mygpt && bash ~/mygpt/tools/setup_box.sh
cd ~/mygpt && . .venv/bin/activate && tmux new -s bench
```

`gpu_check.py` must end with `HEALTHY` before you benchmark; a slow host gives you a wrong budget.

### 2. Pull what the configs read (3 min)

```bash
rclone lsf r2:mygpt/tok/                      # your shard folders from Module 3
rclone copy r2:mygpt/tok/<baseline-folder>/ <data_dir of baseline_124m.yaml> \
  --include "meta.json" --include "val_*" --include "train_00000[0-1].bin" --multi-thread-streams 8
rclone copy r2:mygpt/tok/<main-folder>/ <data_dir of main_350m.yaml> \
  --include "meta.json" --include "val_*" --include "train_00000[0-1].bin" --multi-thread-streams 8
```

Two training shards (200M tokens each) are far more than 30 steps need. Use the `data_dir` paths written in the configs, so the benchmark reads data exactly as training will.

### 3. Benchmark the three configs (15 min)

```bash
python scripts/bench.py --config configs/baseline_124m.yaml --steps 30 --price-per-gpu-hour 4.39 | tee bench-baseline.txt
python scripts/bench.py --config configs/ablation_base.yaml --steps 30 --price-per-gpu-hour 2.69 | tee bench-ablation.txt
python scripts/bench.py --config configs/main_350m.yaml   --steps 30 --price-per-gpu-hour 4.39 | tee bench-main.txt
```

The baseline and main configs are written for 8 GPUs; on one GPU `bench.py` runs the same micro-batch and sequence length, so its tokens/s is the per-GPU number you need. The first steps include `torch.compile` warm-up; run each command a second time and keep the second result if they differ.

### 4. Turn the numbers into a budget (5 min)

Fill in your three per-GPU tokens/s and the prices you will pay:

```bash
python tools/budget_calc.py --tps-baseline <baseline tok/s> --tps-ablation <ablation tok/s> \
  --tps-main <main tok/s> --price-1gpu 2.69 --price-8gpu 4.39 | tee budget-revised.md
```

For example, with 480,000 / 520,000 / 175,000 tokens/s the calculator prints:

```text
| Run | Tokens/s per GPU | GPUs | Wall clock | GPU-h | $ per 1B tokens | Cost |
|---|---|---|---|---|---|---|
| Baseline 124M x 10B | 480,000 | 8 | 1.0 h | 8.0 | $2.82 | $35 |
| Ablations 10 x 1.0B | 520,000 | 1 | 6.7 h | 6.7 | $1.44 | $18 |
| Main 350M x 20B | 175,000 | 8 | 5.5 h | 44.1 | $7.74 | $194 |
| Other items from the course plan | | | | | | $95 |
| **Subtotal** | | | | | | **$342** |
| Contingency to the $500 cap | | | | | | $158 (46%) |
```

GPU-hours include a ×1.25 overhead for compile, evals, checkpoints and one restart, and a 0.9 scaling factor on 8 GPUs. "Other items" are the course plan's fixed lines (dev box, CPU boxes, eval, SFT+DPO, storage).

### 5. Optional: is a B200 cheaper per token? (15 min, ~$2)

Deploy a 1×B200 from the same template (it needs a CUDA 13 driver too), run `setup_box.sh` and only the main config:

```bash
python scripts/bench.py --config configs/main_350m.yaml --steps 30 --price-per-gpu-hour 7.99
```

Compare its dollars per billion tokens with the H100's. Use the B200 for the main run only if it is clearly cheaper per token **and** 8×B200 nodes are available at that price.

### 6. Decide and write it down (10 min, laptop)

Copy `bench-*.txt` and `budget-revised.md` to your repo (`scp -P <PORT> root@<POD_IP>:mygpt/{bench-*.txt,budget-revised.md} .`), then terminate the pod. Apply this rule:

| Subtotal from the calculator | Decision |
|---|---|
| <= $400 (at least 25% contingency) | Keep ~350M × 20B |
| $400–450 | Keep 350M, cut main-run tokens to 15B (cost scales linearly with D), or use a cheaper 8× provider |
| > $450 | Shrink the model to ~250M (cost scales with N) and keep 20B tokens, or take the plan's cheaper path (124–200M on 10B) |

Write the decision and the numbers into `budget.md`, replacing the plan's estimates.

## How you know it worked

- Each `bench.py` run prints tokens/s, MFU and dollars per billion tokens, and the two repeats agree within ~5%.
- MFU lands roughly in 30–50%. On a healthy H100 SXM expect on the order of 400,000–650,000 tokens/s for the 124M configs and 140,000–230,000 for the 350M config (estimates from the 6ND math at 30–50% MFU; your exact numbers depend on the final config).
- The ratio of the 124M to the 350M tokens/s is close to 350/124 ≈ 2.8. A very different ratio means one config is misconfigured (for example compile off in one of them).
- `budget-revised.md` shows a subtotal and contingency, and `budget.md` contains a decision line.
- The pod is terminated and RunPod usage shows about $2–3 for this lesson.

## When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| MFU below 25% | `compile: false`, fp32 instead of bf16, tiny micro-batch, or a slow host | Check the config's `dtype`, `compile`, `micro_batch_size`; check `gpu_check.py` passed |
| CUDA out of memory on `main_350m` | Micro-batch sized for a different GPU | Lower `micro_batch_size` in the config (gradient accumulation keeps the total batch); rerun |
| First run much slower than the second | `torch.compile` warm-up inside the measured steps | Use the second run, or `--steps 60` |
| Numbers jump between runs by >10% | Another process on the GPU, or thermal/power throttling on a community host | `nvidia-smi` for other processes; check power draw near 700 W under load; change host if it persists |
| `FileNotFoundError` for shards | `data_dir` in the config does not match where you copied them | Copy into exactly the config's `data_dir` (or symlink) |
| Calculator says less than 25% contingency | Real speed or price worse than planned | Follow the decision table now; it is far cheaper than discovering it at hour 5 of the main run |

## Artifact

- `bench-baseline.txt`, `bench-ablation.txt`, `bench-main.txt` and `budget-revised.md` committed in your repo (for example under `bench/`).
- `budget.md` with the revised budget table replacing the plan's estimates and one decision line, for example: `Main run: 350M x 20B kept; est. $194 on RunPod Secure 8xH100, 5.5 h wall clock.`
- `budget.md` ledger line:

  ```text
  | 2026-11-13 | 04.3 | RunPod 1x H100 SXM Community, ~0.75 h at $2.69/h, benchmarks | ~$2.00 | $____ |
  ```
