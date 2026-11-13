---
id: "05.1"
module: 5
minutes: 20
practice_minutes: 75
prerequisites: ["03.2", "04.3"]
objectives:
  - Explain every setting in baseline_124m.yaml and where it comes from (GPT-2, GPT-3 Small, llm.c).
  - Derive steps, gradient accumulation and warmup steps for 2 and 8 GPUs from the config.
  - Choose an 8×H100 SXM node and provider for the baseline with today's prices.
  - Rehearse the exact torchrun launch on a cheap 2-GPU box with W&B, bucket checkpoints and HellaSwag.
volatility: implementation
sources:
  - title: "Karpathy, llm.c discussion #481: Reproducing GPT-2 (124M) in llm.c in 90 minutes for $20"
    url: https://github.com/karpathy/llm.c/discussions/481
  - title: "Karpathy, build-nanogpt (video + code: Let's reproduce GPT-2 124M)"
    url: https://github.com/karpathy/build-nanogpt
  - title: "Radford et al., Language Models are Unsupervised Multitask Learners (GPT-2)"
    url: https://cdn.openai.com/better-language-models/language-models.pdf
  - title: "Brown et al., Language Models are Few-Shot Learners (GPT-3; Small model hyperparameters)"
    url: https://papers.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html
  - title: "PyTorch: torchrun (elastic launch)"
    url: https://docs.pytorch.org/docs/stable/elastic/run.html
  - title: "RunPod GPU pricing"
    url: https://www.runpod.io/pricing
  - title: "Lambda pricing"
    url: https://lambda.com/pricing
  - title: "W&B quickstart (API key via WANDB_API_KEY)"
    url: https://docs.wandb.ai/quickstart/
last_verified: "2026-11-13"
---

# 05.1 · The baseline config and a multi-GPU launch

The baseline is your honest reference point: a faithful GPT-2 124M trained on 10B tokens with the GPT-2 tokenizer, so that every later number (your ablations, your 350M model) can be compared against something known. The run itself is cheap, about an hour on one 8×H100 node, but the node bills $30–37 an hour from the second it starts. This lesson makes sure nothing about the launch is new when you rent it: you read the config line by line, pick the node, and rehearse the exact `torchrun` command on a 2-GPU box that costs well under a dollar an hour.

## Goal

You understand every key in `configs/baseline_124m.yaml` and can say which published recipe it comes from. You have a RunPod template for 8-GPU nodes with your W&B key, today's price and availability of an 8×H100 SXM node in `budget.md`, and a successful 100-step rehearsal on 2 GPUs: `torchrun` with DDP, W&B logging, checkpoints uploaded to `ckpt/rehearsal/` in your bucket and an in-training HellaSwag score. The launch command for 05.3 is committed in your repo.

## What you need

- Token shards `tok/gpt2-10BT/` in your bucket from [03.2](../module-03/lesson-02.md), your benchmark numbers and revised budget from [04.3](../module-04/lesson-03.md), and the `mygpt-gpu` template from [04.2](../module-04/lesson-02.md).
- Your W&B API key in your password manager ([01.2](../module-01/lesson-02.md)).
- The [Module 5 lab files](../../labs/module-05/) copied into `tools/` of your project repo and committed (`run_summary.py`).
- About 95 minutes, ~45 of them on a rented 2-GPU box.
- **Expected cost: $0.50–2** (2×RTX 4090 Community at ~$0.68/h, or 2×A100 SXM Community at ~$2.78/h if no 4090 host with CUDA 13 is free).

## Choices and costs

### The config, line by line

GPT-2's paper fixes the architecture (12 layers, 12 heads, width 768, context 1024, vocabulary 50,257, pre-norm LayerNorm, GELU MLP of width 3,072, learned positions, tied input/output embeddings) but not the optimizer settings for the small model. Every public reproduction (llm.c, build-nanogpt) takes those from the GPT-3 paper's "Small" model, which has the same shape. So does `baseline_124m.yaml`:

| Key | Value | Source / reason |
|---|---|---|
| `n_layer, n_head, n_embd, block_size` | 12, 12, 768, 1024 | GPT-2 124M (PUBLICLY DOCUMENTED) |
| `pos_emb, norm, mlp, bias, tie_embeddings` | learned, layernorm, gelu, true, true | GPT-2 architecture; the "nothing changed" baseline |
| `vocab_size` | 50304 | 50,257 padded up to a multiple of 128 (393 × 128) for faster matmuls; the 47 extra rows are never targets and learn to be ignored |
| `dropout` | 0.0 | Standard for single-epoch pretraining (llm.c, build-nanogpt) |
| `total_batch_tokens` | 524,288 (2^19) | GPT-3 Small: 0.5M-token batch; llm.c `-d 524288` |
| `micro_batch_size × seq_len` | 64 × 1024 | 64 sequences per GPU fit in 80 GB in bf16; on 8 GPUs that is the whole batch (accumulation 1) |
| `lr`, `min_lr_ratio`, `schedule` | 6e-4, 0.1, cosine | GPT-3 Small peak LR 6e-4, cosine to 10% (llm.c decays to 0 with `-q 0.0`; the difference is small) |
| `warmup_tokens` | 375,000,000 (715 steps) | GPT-3: linear warmup over 375M tokens; llm.c uses 700 steps |
| `optimizer`, `weight_decay`, `grad_clip` | adamw (β = 0.9, 0.95), 0.1, 1.0 | GPT-3 recipe; decay only on matrices (as in llm.c) |
| `dtype`, `compile` | bf16, true | Speed only; does not change what is learned |
| `train_tokens` | 10,000,000,000 | 10B FineWeb-Edu tokens, as in build-nanogpt; llm.c used 10B FineWeb tokens |
| `eval_every`, `eval_tokens` | 250 steps, 10.5M tokens | ~2% overhead; a smooth val curve |
| `hellaswag_every` | 1000 steps | Full 10,042-example validation set, split across GPUs, about 20 times per run |
| `ckpt_every`, `keep_last` | 1000 steps, 3 | One checkpoint every ~3–5 minutes on 8×H100; you lose at most that much on a crash |

What is **not** faithful, on purpose: the data (FineWeb-Edu instead of OpenAI's unreleased WebText) and the amount of it (10B tokens; OpenAI never published the GPT-2 token count). That is why you compare against the OpenAI weights on benchmarks, not on training loss.

### The numbers the config implies

$$\text{steps} = \left\lfloor \frac{10^{10}}{524{,}288} \right\rfloor = 19{,}073 \qquad \text{warmup} = \left\lfloor\frac{3.75\times10^{8}}{524{,}288}\right\rfloor = 715 \text{ steps}$$

$$\text{accumulation} = \frac{\text{total\_batch\_tokens}}{\text{micro} \times \text{seq\_len} \times \text{GPUs}} = \frac{524{,}288}{64 \times 1024 \times 8} = 1$$

On 2 GPUs with a micro-batch of 16 (a 24 GB card) the same batch needs $524{,}288 / (16 \times 1024 \times 2) = 16$ accumulation steps. The loss curve is the same; only the time per step changes. That is why the rehearsal on small GPUs tests the real recipe.

### Where to rent the 8×H100 node (checked 2026-11-13)

For data-parallel training you want **8× H100 SXM on one HGX board**: every GPU talks to every other over NVLink (900 GB/s). Each step all-reduces ~0.5 GB of fp32 gradients; over NVLink that takes milliseconds, over PCIe it eats a visible share of every step.

| Provider | 8×H100 SXM node | Per GPU-h | Notes |
|---|---|---|---|
| **RunPod Secure Cloud** | $35.12/h | $4.39 | Same template workflow as your 1-GPU boxes; per-second billing; availability varies by hour |
| Lambda | $31.92/h today, **$36.72/h from 2026-11-16** | $3.99 → $4.59 | Reliable 8× nodes; `shutdown` does not stop billing, terminate via dashboard or API |
| Vast.ai verified datacenter hosts | ~$21–24/h | ~$2.67–3.00 | Cheapest; check per-TB bandwidth price and reliability of the exact host |
| Nebius | ~$36/h | $4.50 | Self-serve; quotas for new accounts |
| 2-GPU rehearsal box (RunPod Community) | 2×RTX 4090 ~$0.68/h; 2×A100 SXM ~$2.78/h | | Any 2 GPUs with a CUDA 13 driver test the launch path |

**Recommended:** RunPod Secure Cloud 8×H100 SXM for the baseline (your template, secrets and stop commands already work there); Lambda or a verified Vast.ai host as fallback if no 8× pod is free when you want to run.

> [!NOTE]
> Prices from the RunPod and Lambda pricing pages and the Vast.ai offer list on 2026-11-13; Lambda's increase takes effect on 2026-11-16. Re-check the live price and 8-GPU availability right before you launch, and update `budget.md`.

> [!WARNING]
> An 8×H100 node costs **$21–37 per hour** whether it trains or sits idle, about $500–900 a day if forgotten. Never rent it to "look around": everything in this lesson is rehearsed on a 2-GPU box first.

## Do this

### 1. Read the config and check the arithmetic (5 min, laptop)

```bash
cd ~/mygpt && . .venv/bin/activate
python - <<'EOF'
from mygpt.config import load_config
c = load_config("configs/baseline_124m.yaml").train
for gpus, micro in ((8, 64), (2, 16)):
    accum = c.total_batch_tokens // (micro * c.seq_len * gpus)
    print(f"{gpus} GPUs, micro {micro}: accum {accum}, steps {c.train_tokens // c.total_batch_tokens}, "
          f"warmup {c.warmup_tokens // c.total_batch_tokens}")
EOF
```

Expected: `8 GPUs, micro 64: accum 1, steps 19073, warmup 715` and `2 GPUs, micro 16: accum 16, ...`.

### 2. Make the 8-GPU template (10 min)

1. RunPod → **Secrets → Create Secret** `wandb_api_key` with your W&B key.
2. **Templates** → your `mygpt-gpu` → **Duplicate**, name it `mygpt-8gpu`. Change **container disk to 150 GB** (venv ~10 GB, 20 GB of shards, three ~1.5 GB checkpoints, compile caches) and add one environment variable:

   ```text
   WANDB_API_KEY={{ RUNPOD_SECRET_wandb_api_key }}
   ```

3. Add the same line to `mygpt-gpu` so 1-GPU boxes log to W&B too.

### 3. Check 8×H100 availability and price (5 min)

**Pods → Deploy → Secure Cloud**, filter **H100 SXM**, set GPU count **8**. Write down whether it is offered right now and the hourly price. Do the same on Lambda's dashboard (instance type "8x H100 SXM"). Do not deploy. Add both to `budget.md`.

### 4. Rent the rehearsal box (10 min)

Deploy **2× RTX 4090** (Community Cloud, CUDA filter 13.0+) from `mygpt-8gpu`; if none is free, 2× A100 SXM. Then, as in 04.2, arm the hard stop first and set up:

```bash
nohup bash -c 'sleep 2h; runpodctl pod stop $RUNPOD_POD_ID' >/tmp/hardstop.log 2>&1 &
git clone https://github.com/<you>/<your-repo>.git ~/mygpt && bash ~/mygpt/tools/setup_box.sh
cd ~/mygpt && . .venv/bin/activate
nvidia-smi topo -m          # 4090s show PHB/NODE/SYS (PCIe); an H100 SXM node shows NV18 everywhere
```

### 5. Pull a slice of the shards and HellaSwag (3 min)

The rehearsal needs about 105M training tokens, so two shards are enough:

```bash
rclone copy r2:mygpt/tok/gpt2-10BT/ data/tok/gpt2-10BT \
  --include "meta.json" --include "val_*" --include "train_00000[0-1].bin" --multi-thread-streams 8 --progress
python -c "from mygpt.evaluate import load_hellaswag; print(len(load_hellaswag()), 'HellaSwag examples cached')"
```

Expect `10042 HellaSwag examples cached`. Downloading it once **before** `torchrun` checks that the box can reach GitHub and caches the file in `~/.cache/mygpt/`. (If it is missing, `mygpt.train` downloads it on rank 0 at startup, to a temporary file that is renamed when complete, while the other ranks wait.)

### 6. Launch the rehearsal in tmux (10 min)

```bash
tmux new -s train
mkdir -p runs
torchrun --standalone --nproc_per_node=2 -m mygpt.train --config configs/baseline_124m.yaml \
  train.micro_batch_size=16 train.train_tokens=52_428_800 train.warmup_tokens=10_485_760 \
  train.eval_every=50 train.hellaswag_every=50 train.ckpt_every=50 \
  train.out_dir=runs/rehearsal train.run_name=rehearsal \
  train.ckpt_remote=$MYGPT_REMOTE/ckpt/rehearsal train.wandb_project=my-gpt 2>&1 | tee runs/rehearsal.out
```

`--standalone` runs the rendezvous on this machine (single node); `--nproc_per_node` starts one process per GPU. The overrides shrink the run to 100 steps with a 20-step warmup but leave the model, batch size and data path untouched. Detach with <kbd>Ctrl</kbd>+<kbd>b</kbd> <kbd>d</kbd>; in a second window watch `nvidia-smi -l 2` (both GPUs near 100%).

### 7. Check the bucket, W&B and the summary (5 min)

```bash
rclone lsf r2:mygpt/ckpt/rehearsal/ --format "sp"     # ckpt_00000050.pt, ckpt_00000100.pt (~1.5 GB each), log.jsonl
python tools/run_summary.py runs/rehearsal --gpus 2 --target-tokens 10e9
```

Open wandb.ai → project `my-gpt` → run `rehearsal`: loss, lr, grad_norm, tokens_per_s, val_loss, hellaswag.

### 8. Write down the real launch command (5 min)

Commit this as `launch/baseline_8xh100.sh` in your repo; 05.3 runs it unchanged:

```bash
#!/usr/bin/env bash
# 05.3: the GPT-2 124M baseline on one 8xH100 SXM node. Run inside tmux from the repo root.
set -euo pipefail
mkdir -p runs
torchrun --standalone --nproc_per_node=8 -m mygpt.train --config configs/baseline_124m.yaml \
  train.ckpt_remote=$MYGPT_REMOTE/ckpt/baseline_124m train.wandb_project=my-gpt \
  2>&1 | tee -a runs/baseline_124m.out
```

If you go straight on to 05.2, keep this box: the drill uses the same setup. Otherwise terminate it now.

## How you know it worked

- The first line of the run reads `[train] params 124.5M | world 2 | micro 16x1024 | accum 16 | steps 100 | device cuda:0 | dtype bf16`.
- Step 1 loss is about 10.9–11.0 (a uniform guess over 50,257 tokens is ln 50,257 ≈ 10.8); by step 100 the training loss is around 6–7 and falling.
- Val loss and `hellaswag acc_norm` lines appear at steps 50 and 100. HellaSwag is about 0.25–0.26 at this point: chance level for four endings. That is correct; the point is that the evaluation runs on both GPUs and returns a number.
- `rclone lsf` shows two checkpoints of ~1.5 GB (124.5M fp32 weights plus two AdamW moment tensors) and `log.jsonl`; W&B shows the run with all metrics.
- `run_summary.py` reports a median node speed. On 2×4090 expect very roughly 100,000–200,000 tokens/s (ESTIMATE); the number only tells you the launch path works. Your real 8×H100 speed comes from 04.3 and is checked in the first minutes of 05.3.

## When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| Run hangs after `[train] params ...` with GPUs at 100% and no step lines, on 4090s | NCCL peer-to-peer over PCIe misbehaves on some consumer-GPU hosts | Stop with <kbd>Ctrl</kbd>+<kbd>c</kbd>, relaunch with `NCCL_P2P_DISABLE=1 torchrun ...` (rehearsal only; never needed on an SXM node) |
| `ValueError: total_batch_tokens ... not divisible by micro*seq*world` | Micro-batch × GPUs does not divide 524,288 | Use a power-of-two micro-batch (8, 16, 32, 64) |
| `CUDA out of memory` at step 1 | Micro-batch 64 does not fit in 24 GB | `train.micro_batch_size=16` (or 8); accumulation keeps the batch |
| `wandb: ERROR api_key not configured` or no run on wandb.ai | `WANDB_API_KEY` not in the template, or the secret name differs | `echo ${WANDB_API_KEY:0:4}`; fix the template and redeploy, or `export WANDB_API_KEY=...` for this session |
| `[ckpt] WARNING: ckpt_remote=... set but rclone is not installed` | `setup_box.sh` did not finish | Rerun `bash tools/setup_box.sh`; check `rclone lsf r2:mygpt/` |
| `URLError` or `FileNotFoundError` from HellaSwag | No internet to GitHub from the box | Run the step 5 pre-download (or copy `hellaswag_val.jsonl` into `~/.cache/mygpt/`), then relaunch (the run resumes from its checkpoint if it has one) |
| `torchrun: command not found` | venv not active in this tmux window | `cd ~/mygpt && . .venv/bin/activate` |

## Artifact

- RunPod template `mygpt-8gpu` (150 GB container disk, R2, HF and W&B secrets) and the `wandb_api_key` secret.
- `launch/baseline_8xh100.sh` and `tools/run_summary.py` committed; the rehearsal's checkpoints and log in `r2:mygpt/ckpt/rehearsal/` (05.2 uses them; about 3 GB, a few cents a month).
- `budget.md`: today's 8×H100 price and availability on RunPod and Lambda, and the ledger line:

  ```text
  | 2026-11-13 | 05.1 | RunPod 2x RTX 4090 Community, ~0.75 h at $0.68/h, launch rehearsal | ~$0.50 | $____ |
  ```
