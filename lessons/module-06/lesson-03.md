---
id: "06.3"
module: 6
minutes: 25
practice_minutes: 60
prerequisites: ["06.2"]
objectives:
  - Summarise what current public evidence says about Muon versus AdamW and about WSD versus cosine schedules, and how strong each claim is.
  - Run an LR sweep for AdamW, Muon at three learning rates, a WSD schedule and the Muon+WSD combination on your chosen architecture.
  - Compare optimizers fairly (tuned baseline, end of schedule, matched tokens) and pick the optimizer, learning rates and schedule for the main run.
volatility: implementation
sources:
  - title: "Keller Jordan, Muon: An optimizer for hidden layers in neural networks (blog, Dec 2024)"
    url: https://kellerjordan.github.io/posts/muon/
  - title: "Keller Jordan, Muon reference implementation"
    url: https://github.com/KellerJordan/Muon
  - title: "Kimi Team (Moonshot AI), Muon is Scalable for LLM Training (arXiv 2502.16982)"
    url: https://arxiv.org/abs/2502.16982v1
  - title: "Wen, Hall, Ma, Liang: Fantastic Pretraining Optimizers and Where to Find Them (arXiv 2509.02046, ICLR 2026)"
    url: https://arxiv.org/abs/2509.02046v1
  - title: "Keller Jordan et al., modded-nanogpt (Muon in the speedrun records)"
    url: https://github.com/KellerJordan/modded-nanogpt
  - title: "Hu et al., MiniCPM: warmup-stable-decay (WSD) scheduler (arXiv 2404.06395)"
    url: https://arxiv.org/abs/2404.06395
  - title: "Hägele et al., Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations (arXiv 2405.18392)"
    url: https://arxiv.org/abs/2405.18392v3
last_verified: "2026-11-13"
---

# 06.3 · Optimizer and schedule ablations

Architecture changes move the loss by a little; the optimizer and the learning-rate schedule can move it by more, and they are where the most exaggerated claims live. Muon has driven most of the NanoGPT speedrun's records and has been used for a 16B-parameter MoE model, yet the most careful comparison so far finds its advantage over a well-tuned AdamW to be modest and shrinking with scale. In this lesson you test both claims on your own architecture and data, with a tuned AdamW as the opponent, and settle the optimizer, the learning rates and the schedule for the main run.

## Goal

`results/ablations.md` gains seven runs on your chosen architecture: AdamW at two more learning rates, AdamW with WSD, Muon at three learning rates, and Muon with WSD, compared against your 06.2 winner as the control. Below the table you have written the optimizer, `lr`, `muon_lr`, `weight_decay`, `schedule` and `decay_frac` for the main run, with the evidence.

## What you need

- Your architecture decision from [06.2](lesson-02.md) and its run (`abl_qwen3` or `abl_llama`) as the control; the ablation box set up as in [06.1](lesson-01.md).
- About 4.5 hours of wall clock for 7 runs, mostly unattended; ~40 minutes of your time.
- **Expected cost: $11–13** (7 runs × ~35–38 min on 1×H100 SXM Community at $2.69/h, plus setup).

## Choices and costs

### What the public evidence says (checked 2026-11-13)

| Claim | Source | Strength |
|---|---|---|
| Muon (momentum, then Newton–Schulz orthogonalisation of each 2D update) trains transformer hidden layers faster than AdamW | Keller Jordan's blog post and reference code (Dec 2024); the modded-nanogpt speedrun, where Muon has been part of the record recipes since 2024 (8×H100, 124M-class models, val loss 3.28 target) | PUBLICLY DOCUMENTED for small models in a heavily tuned setting |
| Muon reaches AdamW's loss with ~52% of the training FLOPs (~2× compute efficiency) at compute-optimal budgets, once weight decay and per-matrix update scaling are added | Kimi Team, "Muon is Scalable for LLM Training" (Feb 2025); used for Moonlight (3B active / 16B MoE, 5.7T tokens) | PUBLICLY DOCUMENTED by the authors; their own scaling-law experiments |
| Against a carefully tuned AdamW, the best matrix-based optimizers (Muon among them) give 1.3–1.4× speedup below 1B parameters, falling to ~1.1× at 1.2B; many reported 2× gains come from weak baselines; rankings can flip during LR decay | Wen, Hall, Ma, Liang, "Fantastic Pretraining Optimizers and Where to Find Them" (Sep 2025, ICLR 2026) | PUBLICLY DOCUMENTED, independent benchmark with per-optimizer tuning |
| Warmup-stable-decay (constant LR, then a short decay) matches cosine at the end, and lets you choose the training length late or branch decays off a stable checkpoint | MiniCPM (Apr 2024) introduced the name; Hägele et al. (NeurIPS 2024) show constant LR + cooldown matches cosine and scales predictably | PUBLICLY DOCUMENTED |

What this means for you: expect Muon to win at ~110M parameters by a margin that is real but smaller than headline numbers, and expect the margin to shrink at 350M (INFERENCE from Wen et al.). Expect WSD and cosine to end within noise of each other; WSD's value is operational.

### How `mygpt` implements them

- `train.optimizer=muon`: Muon updates every 2D matrix inside the transformer blocks (attention and MLP weights), with momentum 0.95 (Nesterov), 5 Newton–Schulz steps, and the update scaled by $\sqrt{\max(1, \text{rows}/\text{cols})}$. Embeddings, the output head, norms and biases stay on AdamW with `train.lr`. Both follow the same schedule. Under DDP each matrix is orthogonalised on one rank and broadcast, so checkpoints from rank 0 are complete.
- Weight decay is **separate** per optimizer: `train.weight_decay` (0.1) applies to the AdamW groups (embeddings, head) and `train.muon_weight_decay` (default 0.0) to the Muon block matrices. For Muon, decay is applied as $p \leftarrow p\,(1 - \text{lr}\cdot\text{wd})$ with Muon's LR of ~0.02, so 0.1 would shrink block weights ~33× harder per step than AdamW's 6e-4 × 0.1. The Muon runs therefore use `muon_weight_decay=0.0` (as `main_350m.yaml` does) and keep AdamW's 0.1 on the embeddings, so the only change against the AdamW runs is the optimizer of the block matrices.
- `train.schedule=wsd`: warmup, constant LR, then linear decay over the last `decay_frac` (0.2) of steps to `min_lr_ratio` (0.0 in these runs). Cosine in the control decays to 10%; the end-of-run comparison absorbs that difference.

### The runs

All on top of your architecture (the `@ARCH` macro in `ablations.txt`; edit it if you chose Llama or untied):

| Run | Overrides | Question |
|---|---|---|
| control | your 06.2 winner (AdamW 6e-4, cosine) | |
| `abl_arch_lr3e-4`, `abl_arch_lr1.2e-3` | `train.lr=3e-4` / `1.2e-3` | Is 6e-4 the right AdamW LR? Muon must beat the **best** AdamW, not the default |
| `abl_arch_wsd` | AdamW + `schedule=wsd decay_frac=0.2 min_lr_ratio=0.0` | Does WSD lose anything against cosine? |
| `abl_arch_muon`, `_muon_lr01`, `_muon_lr04` | `optimizer=muon muon_lr=0.02/0.01/0.04 muon_weight_decay=0.0` | Does Muon win, and is 0.02 near its best LR? |
| `abl_arch_muon_wsd` | Muon 0.02 + WSD | The likely main-run combination, measured as a whole |

Muon adds a few percent of time per step (Newton–Schulz on each matrix); 7 runs ≈ 4.4 GPU-h ≈ $12 at $2.69/h.

> [!NOTE]
> Price checked 2026-11-13 (RunPod Community H100 SXM $2.69/h). Muon and WSD are active research areas; check the modded-nanogpt README and recent optimizer benchmarks for anything that changes the picture before you decide.

## Do this

### 1. Point the macro at your architecture and activate the block (5 min)

In `tools/ablations.txt`, uncomment the `@ARCH` line and make it match your 06.2 choice exactly (for Llama: drop `model.qk_norm=true`; untied: add `model.tie_embeddings=false`). Uncomment the seven `abl_arch_*` lines, then:

```bash
DRY_RUN=1 bash tools/run_ablations.sh tools/ablations.txt
```

Every new command must contain your architecture overrides plus the run's own.

### 2. Run the queue (5 min, then ~4.5 hours unattended)

```bash
cd ~/mygpt && git pull && . .venv/bin/activate
tmux new -s queue
WANDB_PROJECT=my-gpt-ablations bash tools/run_ablations.sh tools/ablations.txt 2>&1 | tee -a runs/queue.log
```

Set the hard stop to `sleep 6h` on a new box. In the first minutes of `abl_arch_muon`, check that the loss falls at least as fast as in the control; a Muon run whose loss rises or stalls in the first 100 steps has a wrong setting (most often `weight_decay` left at 0.1).

### 3. Build the table with the right control (10 min)

```bash
python tools/ablation_table.py "runs/abl_*" --queue tools/ablations.txt --control abl_qwen3 \
  --noise <06.1 seed spread> --price-per-gpu-hour 2.69 | tee results/ablations-opt.md
```

(Use `--control abl_llama` if that was your choice.) The new control is a single run, so `--noise` passes in the seed spread your two 06.1 controls measured: same token budget and data, so the same kind of noise.

### 4. Read it in this order (15 min)

1. **AdamW LR**: which of 3e-4, 6e-4, 1.2e-3 is best? That run is the fair AdamW baseline.
2. **Muon LR**: which of 0.01, 0.02, 0.04 is best? If the best is at an end of the range (0.04), the true optimum may be higher; note it rather than spending more runs.
3. **Muon vs AdamW**: best Muon minus best AdamW. Beyond the threshold?
4. **WSD vs cosine**: `abl_arch_wsd` vs the control, and `abl_arch_muon_wsd` vs `abl_arch_muon`. Within noise is the expected and acceptable outcome.
5. In W&B, overlay `val_bpb` for the control and both WSD runs: the WSD curves sit above cosine until ~step 1,525 and then drop below or onto it during the decay. This is why only final values are compared.

### 5. Decide the optimizer recipe (10 min)

| Result | Main-run recipe |
|---|---|
| Best Muon beats best AdamW beyond the threshold | `optimizer: muon`, `muon_lr` = best Muon LR, `muon_weight_decay: 0.0`, `weight_decay: 0.1` (AdamW part), `lr` (AdamW part) = best AdamW LR |
| Muon within noise of the best AdamW | AdamW at its best LR is the simpler, better-understood choice; Muon is acceptable if you prefer the speedrun-style recipe, but write that the evidence was neutral |
| WSD within noise of cosine (or better) | `schedule: wsd`, `decay_frac: 0.2`, `min_lr_ratio: 0.0`: same loss, and in Module 7 you can stop the stable phase early and still decay to a usable model |
| WSD worse beyond the threshold | Keep cosine and look at the WSD run's decay (did it reach the end?) |

Write the recipe under the table, for example: `Optimizer: Muon (muon_lr 0.02) + AdamW 6e-4 for embeddings/head, wd 0.1 AdamW / 0.0 Muon, WSD decay_frac 0.2. Best Muon -0.0xx bpb vs best AdamW (lr 1.2e-3); threshold 0.00x.` Then merge `results/ablations-opt.md` into `results/ablations.md` under a heading `## Optimizer and schedule (control: abl_qwen3)`.

### 6. Terminate the box

> [!WARNING]
> This is the last planned GPU work of the module. Check `runs/queue.log` shows the last run `ok`, `rclone lsf r2:mygpt/ckpt/ | grep abl_` lists all runs, then terminate the pod in the console. Do not leave the idle guard to do it.

## How you know it worked

- Seven new `ok` runs; every Muon run header shows the same parameter count as the control (Muon changes the optimizer, not the model).
- The table compares all runs at step 1,907; the AdamW LR runs bracket the control (one side better, or both worse: then 6e-4 was already about right).
- You wrote one recipe line with numbers, and it names a single optimizer, LRs and schedule.
- The total ablation spend in `budget.md` (06.1–06.3) is under $60.

## When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| Muon loss stalls or rises after ~200 steps | `muon_weight_decay` 0.1 with Muon's LR shrinks the block weights each step | Rerun with `train.muon_weight_decay=0.0` (the default) under a new name |
| Muon loss diverges (NaN) at 0.04 | LR too high for this model | Report it; 0.04 is outside the stable range, use the best stable LR |
| `abl_arch_lr1.2e-3` diverges but the QK-norm architecture was supposed to allow higher LRs | Too high anyway, or a different problem (data, host) | Look at the step and grad norms; one NaN at 2× LR is still information: stay at or below 6e-4–1.2e-3 |
| WSD run's final value looks bad and its LR column never decayed | `schedule=wsd` with `decay_frac=0` or a typo | Check the `lr` column in the last 20% of the log; rerun with the exact overrides |
| Table shows "noise unknown (1 control)" | You left out `--noise` | Pass `--noise` with the 06.1 seed spread |
| Spend is heading above $60 | Extra reruns or a pricier box | Drop `abl_arch_muon_lr01` or the third seed; the decision rarely depends on them |

## Artifact

- `results/ablations.md` (version 3) committed: architecture runs, optimizer and schedule runs, verdicts, and the recipe line.
- `tools/ablations.txt` with every block that ran active, committed.
- `budget.md` ledger line, plus an "Ablations total" line summing 06.1–06.3:

  ```text
  | 2026-11-13 | 06.3 | RunPod 1x H100 SXM Community, ~4.6 h at $2.69/h, 7 optimizer/schedule ablations | ~$12.50 | $____ |
  ```
