---
id: "09.2"
module: 9
minutes: 20
practice_minutes: 210
prerequisites: ["09.1", "04.2"]
objectives:
  - Choose full fine-tuning over LoRA for a 350M model and set SFT hyperparameters from a published small-model recipe and a short learning-rate sweep.
  - Run scripts/sft.py on a 1×H100 with automatic stops, and turn the first minutes of logs into a measured time and cost.
  - Judge the SFT run from its logs (eval loss, token accuracy, gradient norm) and from side-by-side answers of the base and SFT models.
  - Store the SFT model in the bucket under hf/ and record what a 350M chat model can and cannot do.
volatility: implementation
sources:
  - title: "TRL 1.17 SFT Trainer docs (SFTConfig defaults, logged metrics, PEFT)"
    url: https://huggingface.co/docs/trl/sft_trainer
  - title: "alignment-handbook SmolLM2 recipes (config_smol.yaml for 135M/360M)"
    url: https://github.com/huggingface/alignment-handbook/tree/main/recipes/smollm2
  - title: "SmolLM2: When Smol Goes Big (paper)"
    url: https://arxiv.org/pdf/2502.02737
  - title: "SmolLM2-360M-Instruct model card (evaluation table)"
    url: https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct
  - title: "HuggingFaceTB/smol-smoltalk dataset card"
    url: https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk
  - title: "Trackio documentation (Transformers report_to integration)"
    url: https://huggingface.co/docs/trackio/index
  - title: "Vast.ai CLI commands (stop vs destroy instance)"
    url: https://docs.vast.ai/cli/commands
last_verified: "2026-11-13"
---

# 09.2 · Running SFT and checking the chat model

This is the step where your model starts answering. The run itself is small next to pretraining, two or three hours on one H100, but it has its own ways to fail quietly: a learning rate that does nothing, a template that trains the model never to stop, a loss that looks fine while the answers are nonsense. You run a 25-minute learning-rate sweep first, then the full two epochs, and you judge the result twice: from the training logs and by reading the base and SFT models' answers to the same prompts side by side.

## Goal

You have `hf/my-gpt-350m-sft/` (a full fine-tune of your base model on smol-smoltalk, with tokenizer and chat template) in your bucket, the training log, a learning-rate sweep result, and `reports/sft_probe.md` comparing the base and SFT models on 12 probe prompts. You know the run's measured tokens per second and cost, and you have written down, with examples, what your 350M chat model can and cannot do.

## What you need

- The data decision from [09.1](lesson-01.md) (`reports/sft_data.md`), with `tools/sft_data_stats.py` and `tools/probe_chat.py` committed.
- Your RunPod template `mygpt-gpu` from [04.2](../module-04/lesson-02.md), RunPod credit, and the base model in `r2:mygpt/hf/my-gpt-350m-base/`.
- Optional: a W&B key in the template (`WANDB_API_KEY`) to watch curves in the browser (`--report-to wandb`), or Trackio instead (`pip install trackio`, then `--report-to trackio`; free, no W&B account, dashboard local or on a free HF Space). The default `--report-to none` keeps only the console and trainer logs.
- A 1× H100 at RunPod or one of the alternatives in [09.1](lesson-01.md)'s box table.
- About 4 hours wall clock, of which ~3–3.5 h on the GPU (sweep ~0.5 h, full run ~2–3 h, checks ~0.3 h).
- **Expected cost: ~$8–10** on Community Cloud at $2.69/h (~$13–16 on Secure at $4.39/h).

> [!WARNING]
> The pod bills from Running to Terminate, whether or not training runs. Arm the hard stop (step 1) before anything else, and set it long enough for the whole lesson (5 hours), not for the run alone.

## Choices and costs

### Full fine-tune or LoRA

`sft.py` does both: the default is a full fine-tune, `--lora` trains rank-32 adapters on all linear layers (`r=32, lora_alpha=64`).

| | Full fine-tune (**recommended**) | LoRA (`--lora`) |
|---|---|---|
| Trainable parameters | all ~337M | ~15M (about 4.5%) |
| Training memory, weights + grads + AdamW state | ~5.4 GB in fp32 for 337M params (16 bytes per parameter) | ~1.4 GB frozen weights + ~0.2 GB for adapters |
| Compute per token | $6N$ | about $4N$ (no weight gradients for frozen layers), activations still full size |
| What `--out` contains | a complete model | only an adapter that points at the base folder by path |
| Learning rate | 5e-5 to 1e-3 (sweep below) | ~1e-4 (TRL docs) or higher; the default 5e-5 is too low |

At 350M parameters on an 80 GB GPU, LoRA saves memory you do not need and buys a ~30% speed-up at the cost of a weaker format change and an adapter folder that 10.1, 10.2 and the Hub upload would all have to handle specially. LoRA earns its place for 7B+ models on small GPUs, not here.

### Hyperparameters

| Setting | `sft.py` default | SmolLM2-360M recipe (alignment-handbook) | Use |
|---|---|---|---|
| Learning rate | 5e-5 | 1e-3 | **Sweep 5e-5, 2e-4, 1e-3** on 30k conversations, keep the lowest eval loss |
| Schedule / warmup | cosine, 3% | cosine, 10% | default |
| Batch | 16 × 4 accumulation = 64 conversations per step | 4 × 4 per GPU (128 global on 8 GPUs) | default; ~46k tokens per step at 1,024 |
| Max length | 1,024 | 8,192 | your `block_size` (09.1) |
| Epochs | 2 | 2 | default |
| Gradient checkpointing | on (TRL default) | on | **off** (`--set gradient_checkpointing=false`): a 350M model fits without it, and recomputing activations costs time |
| Precision | fp32 master weights, bf16 autocast | bf16 | default |

Why sweep instead of copying 1e-3: the SmolLM2 base model ended pretraining in a different state and was tuned with a 2× larger batch. Three short runs cost about $1.30 and replace a guess with a measurement.

> [!IMPORTANT]
> The weights must be trained in fp32 with bf16 autocast, not stored in bf16. With bf16 weights, an AdamW step of size 5e-5 changes only about 57% of the weights (bf16 rounds the rest away), and at the 5e-7 used for DPO in 10.1 almost none. The project scripts load the model with `dtype=torch.float32`; step 3 checks that before you spend money.

### Time and money (prices checked 2026-11-13)

| Part | GPU time (ESTIMATE) | Community $2.69/h | Secure $4.39/h |
|---|---|---|---|
| Setup, dataset download (~1 GB), template check | 0.3 h | $0.80 | $1.30 |
| LR sweep, 3 × 30k conversations × 1 epoch | 0.5 h | $1.35 | $2.20 |
| Full run, 460k conversations × 2 epochs | 2–3 h | $5.40–8.10 | $8.80–13.20 |
| Probe, chat checks, upload | 0.3 h | $0.80 | $1.30 |
| **Total** | **~3.1–4.1 h** | **~$8–11** | **~$14–18** |

> [!NOTE]
> These are estimates from 09.1's token count at an assumed 20% MFU. Step 5 replaces them with your measured tokens per second. Re-check the price on the deploy page and record what you pay in `budget.md`.

## Do this

### 1. Deploy, arm the stops, install (15 min)

Deploy a 1× H100 SXM pod from `mygpt-gpu` (04.2), then on the pod:

```bash
nohup bash -c 'sleep 5h; runpodctl pod stop $RUNPOD_POD_ID' >/tmp/hardstop.log 2>&1 &
git clone https://github.com/<you>/<your-repo>.git ~/mygpt && bash ~/mygpt/tools/setup_box.sh
cd ~/mygpt && . .venv/bin/activate && uv pip install -e ".[post,eval]"
tmux new -d -s guard "bash scripts/autoshutdown.sh --idle-minutes 20 --grace-minutes 15 2>&1 | tee /tmp/guard.log"
```

The idle guard stops the pod 20 minutes after the GPU goes idle, so a finished or crashed run stops billing even if you are asleep.

Not on RunPod? Copy `r2.env` to the box and source it before `setup_box.sh`, and give both the hard stop and `autoshutdown.sh` your provider's stop command (`--stop-cmd '...'`, list in 07.1 step 4). Without `--stop-cmd`, `autoshutdown.sh` falls back to `sudo shutdown -h now`, which does **not** stop billing on Vast.ai, Lambda or Hyperbolic.

### 2. Pull the base model and check the data is unchanged (5 min)

```bash
bash scripts/sync.sh pull-hf my-gpt-350m-base hf/my-gpt-350m-base
python -c "from huggingface_hub import HfApi; print(HfApi().dataset_info('HuggingFaceTB/smol-smoltalk').sha)"
```

Compare the hash with the one in `reports/sft_data.md`. Steps 4 and 5 pass the recorded hash as `--dataset-revision`, so the run trains on the data you measured even if the dataset changed since; if the hash differs, note it in `reports/sft_data.md`.

### 3. Check the precision before spending money (1 min)

```bash
grep -n "dtype=" scripts/sft.py scripts/dpo.py
```

Both `from_pretrained` lines must say `dtype=torch.float32`. If they say `torch.bfloat16 if bf16 else torch.float32`, your copy of the project predates the fix: pull the current `labs/common` scripts into your repo first.

### 4. Learning-rate sweep (30 min, ~$1.35)

```bash
mkdir -p logs
for lr in 5e-5 2e-4 1e-3; do
  python scripts/sft.py --model hf/my-gpt-350m-base --out runs/sft_lr$lr --lr $lr \
    --dataset-revision <sha from reports/sft_data.md> --max-samples 30000 --epochs 1 --max-length 1024 \
    --set gradient_checkpointing=false --set eval_steps=100 --set save_strategy=no \
    2>&1 | tee logs/sft_lr$lr.log
done
grep -h "eval_loss" logs/sft_lr*.log | tail -n 12
```

Each run is ~470 steps with an eval on 2,000 test conversations every 100 steps. Keep the learning rate with the lowest final `eval_loss`, unless its `grad_norm` was spiking or its loss jumped up at any point: then take the next lower one. Delete `runs/sft_lr*` afterwards; you only need the logs.

### 5. Full run (2–3 h)

```bash
tmux new -s sft
python scripts/sft.py --model hf/my-gpt-350m-base --out hf/my-gpt-350m-sft --lr <best_lr> \
  --dataset-revision <sha from reports/sft_data.md> --save-final-only \
  --max-length 1024 --set gradient_checkpointing=false --set run_name=sft_350m \
  --report-to wandb 2>&1 | tee logs/sft_full.log          # --report-to none without W&B
```

Detach with <kbd>Ctrl</kbd>+<kbd>b</kbd> then <kbd>d</kbd>. Every 10 steps the log prints a line like this one (real format, from a tiny test model on 2026-11-13; your values differ):

```text
{'loss': '10.41', 'grad_norm': '1.465', 'learning_rate': '0.0003455', 'entropy': '10.8', 'mean_token_accuracy': '0.07757', 'num_tokens': '1.656e+04', 'epoch': '0.625'}
```

After ~10 minutes, measure the real speed: note `num_tokens` and the minutes since the first step, then

$$\text{tokens/s} = \frac{\Delta\,\text{num\_tokens}}{\Delta t}, \qquad \text{MFU} = \frac{6 \times 337\text{M} \times \text{tokens/s}}{989 \times 10^{12}}$$

For example, 98,000 tokens/s gives an MFU of 20%, and 670M trained-on tokens take 670M / 98,000 ≈ 1.9 h. Put the measured hours and dollars into `budget.md` now, not at the end.

What healthy looks like over the run (ESTIMATE, a plausible pattern rather than a measured curve for your model):

- `loss` (assistant tokens only) starts around the base model's validation loss plus a bit for the unfamiliar tags, drops quickly in the first few hundred steps as the model learns the format, then decreases slowly.
- `eval_loss` (every 500 steps) follows it down and flattens in the second epoch. At the start of epoch 2, train `loss` drops by a visible step because the model has seen those conversations; if `eval_loss` starts rising at the same time, the second epoch is over-fitting.
- `mean_token_accuracy` rises from the base model's level and flattens; `grad_norm` stays in a stable band after warmup, without spikes.

`sft.py` saves a checkpoint every 500 steps in `hf/my-gpt-350m-sft/checkpoint-*` (keeps 2). If the process crashes while the pod is still up, rerun the same command with `--resume` added: it continues from the newest checkpoint. If the pod itself is gone, the checkpoints went with its disk and the run restarts from step 0; at this size that costs a few dollars. `--save-final-only` deletes the checkpoints (optimizer state, ~4 GB each) once the final model is saved, so the folder holds only what you push.

### 6. Store the model right away (5 min)

The idle guard stops the pod 20 minutes after training ends, and a stopped pod's container disk is erased. Push the model before you do anything else:

```bash
ls -la hf/my-gpt-350m-sft                        # config.json, model.safetensors, tokenizer files, chat_template.jinja; no checkpoint-*
bash scripts/sync.sh push-hf my-gpt-350m-sft hf/my-gpt-350m-sft
bash scripts/sync.sh push logs ckpt/sft_350m/logs    # sweep and full-run logs
rclone ls r2:mygpt/hf/my-gpt-350m-sft | head
```

### 7. Check the model by reading it (20 min)

```bash
python tools/probe_chat.py --models hf/my-gpt-350m-base hf/my-gpt-350m-sft --out reports/sft_probe.md
bash scripts/sync.sh push reports ckpt/sft_350m/reports      # so you can read it on your laptop
python scripts/chat.py --model hf/my-gpt-350m-sft --chat --prompt "Explain photosynthesis simply."
python scripts/chat.py --model hf/my-gpt-350m-sft --chat        # interactive; type exit to quit
python scripts/chat.py --model hf/my-gpt-350m-base --prompt "The history of the printing press"   # base: plain completion
```

Read `reports/sft_probe.md` probe by probe (on your laptop: `bash scripts/sync.sh pull ckpt/sft_350m/reports reports`; the idle guard may stop the pod while you read) and write two lists at its end: what improved and what did not. Expect roughly this for a 350M model trained on 20B tokens:

| Can do after SFT | Cannot do (do not chase it with more SFT) |
|---|---|
| Answer in the assistant role instead of continuing your text | Reliable facts: capitals, authors and dates are often confidently wrong |
| Stop with `<\|endoftext\|>` after one answer (base: never) | Arithmetic beyond small numbers; multi-step reasoning |
| Follow simple format requests: numbered list, one sentence, "yes or no" | Non-trivial code that runs |
| Rewrite and summarize a short text in a fluent way | Remember details across several turns |
| A friendly, consistent tone | Refuse harmful requests consistently (smol-smoltalk has little safety data) |

For calibration: SmolLM2-360M-Instruct, trained on the same data after 4 trillion pretraining tokens, scores 7.4% on GSM8K and 3.66 on MT-Bench (model card). Your model should not be expected to beat that.

### 8. Commit and stop paying (5 min)

```bash
git add reports/sft_probe.md && git commit -m "SFT probe report" && git push
```

If you continue with [10.1](../module-10/lesson-01.md) right away, keep the pod: DPO starts from `hf/my-gpt-350m-sft` on this box. Otherwise terminate it in the RunPod console (Vast.ai: `vastai destroy instance <ID>`; Lambda, Hyperbolic: terminate in the dashboard).

> [!CAUTION]
> Terminating deletes the container disk. Check that `rclone ls` shows `model.safetensors` under `hf/my-gpt-350m-sft/` before you terminate.

## How you know it worked

- `grep` in step 3 shows `dtype=torch.float32` in both scripts.
- The sweep produced three final `eval_loss` values and you can say which learning rate won and by how much.
- The full run ends with `saved SFT model to hf/my-gpt-350m-sft`; `eval_loss` at the end is clearly below its first value and did not rise during epoch 2 by more than a few hundredths.
- Your measured tokens per second, GPU-hours and dollars are in `budget.md`, within about 2× of the 09.1 estimate.
- In `reports/sft_probe.md`, the SFT model stopped by itself on most probes (typically 10–12 of 12) and the base model on few; the SFT answers are in the assistant role and follow the list / one-sentence / yes-or-no instructions most of the time.
- `r2:mygpt/hf/my-gpt-350m-sft/` contains `config.json`, `model.safetensors` (~1.35 GB in fp32), `tokenizer.json`, `tokenizer_config.json` and `chat_template.jinja`.

## When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| Loss barely moves; answers look like the base model's | bf16 weights eat the updates, or learning rate too low | Step 3; rerun the sweep with the fp32 scripts |
| Loss falls, but every answer runs until `--max-new-tokens` | EOT not trained (custom `--chat-template` without it inside `{% generation %}`), or the model folder has the wrong `eos_token` | Recheck the mask printout from 09.1 on `hf/my-gpt-350m-sft`; never pass `--chat-template` |
| Answers repeat the user's question or write the next `<\|user\|>` turn | `--assistant-only-loss false` was set, so user text was trained | Rerun with the default |
| `loss` spikes or becomes `nan` | Learning rate too high for this model | Take the next lower rate from the sweep and restart |
| `CUDA out of memory` | Long batch with gradient checkpointing off | `--batch-size 8 --grad-accum 8` (same 64 conversations per step), or drop `gradient_checkpointing=false` |
| Run much slower than estimated (MFU under 10%) | CPU-bound tokenization, or a bad host | Check `nvidia-smi` utilization; `gpu_check.py` from 04.2; for a slow host, push nothing, terminate and redeploy |
| `ValueError` about a non-conversational dataset | A local JSONL without a `messages` column | Convert to `{"messages": [{"role": ..., "content": ...}, ...]}` per line |
| `eval_loss` rises through epoch 2 | Over-fitting a small model to 460k conversations | Rerun with `--epochs 1` (half the cost) and compare both models with `probe_chat.py` |

## Artifact

- `r2:mygpt/hf/my-gpt-350m-sft/` (full model + tokenizer + chat template) and the logs and probe report under `r2:mygpt/ckpt/sft_350m/`.
- In git: `reports/sft_probe.md` with your "improved / not improved" lists, the winning learning rate and measured speed added to `reports/sft_data.md`.
- `budget.md` lines (replace with your measured numbers):

  ```text
  | 2026-11-13 | 09.2 | RunPod 1x H100 SXM Community, LR sweep 0.5 h at $2.69/h | ~$1.35 | $____ |
  | 2026-11-13 | 09.2 | RunPod 1x H100 SXM Community, SFT 2 epochs + checks ~2.6 h at $2.69/h | ~$7.00 | $____ |
  ```
