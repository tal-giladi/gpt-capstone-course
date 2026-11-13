---
id: "10.1"
module: 10
minutes: 20
practice_minutes: 120
prerequisites: ["09.2"]
objectives:
  - Choose an open preference dataset by license, format and size, and check its pairs through your tokenizer (length bias, truncation).
  - Size a DPO run on one H100, including the memory and compute of the frozen reference model.
  - Run scripts/dpo.py with two settings of beta and learning rate, and read rewards, margins and accuracies to tell learning from over-optimization.
  - Store the DPO models in the bucket and compare them with the SFT model on the same probe prompts.
volatility: implementation
sources:
  - title: "TRL 1.17 DPO Trainer docs (loss, logged metrics, beta, precompute_ref_log_probs)"
    url: https://huggingface.co/docs/trl/dpo_trainer
  - title: "trl-lib/ultrafeedback_binarized dataset"
    url: https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized
  - title: "HuggingFaceH4/ultrafeedback_binarized dataset card (MIT, binarization, TruthfulQA fix)"
    url: https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized
  - title: "argilla/ultrafeedback-binarized-preferences-cleaned dataset card"
    url: https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned
  - title: "allenai/llama-3.1-tulu-3-8b-preference-mixture dataset card"
    url: https://huggingface.co/datasets/allenai/llama-3.1-tulu-3-8b-preference-mixture
  - title: "HuggingFaceTB/smoltalk2 dataset card (preference subset)"
    url: https://huggingface.co/datasets/HuggingFaceTB/smoltalk2
  - title: "alignment-handbook SmolLM2 recipes (DPO config_smol.yaml)"
    url: https://github.com/huggingface/alignment-handbook/tree/main/recipes/smollm2
  - title: "SmolLM2: When Smol Goes Big (paper, DPO on UltraFeedback)"
    url: https://arxiv.org/pdf/2502.02737
last_verified: "2026-11-13"
---

# 10.1 · Preference data and the DPO run

SFT taught your model the format; preference tuning nudges which of its possible answers it prefers. Direct Preference Optimization (DPO) does that with pairs of answers to the same prompt, one marked better, and a frozen copy of the SFT model that keeps the new model from drifting too far. It is the cheapest step of the whole project, under an hour on one H100, which is why you can afford to train two variants. It is also the easiest to fool yourself with: the training metrics always look like progress. This lesson runs DPO, teaches you to read those metrics honestly, and leaves the real verdict to the judge in 10.2.

## Goal

You have two DPO models in your bucket: `hf/my-gpt-350m-dpo/` (the SmolLM2-360M recipe: beta 0.5, learning rate 1e-6, 2 epochs) and `hf/my-gpt-350m-dpo-b01/` (the script defaults: beta 0.1, 5e-7, 1 epoch), both trained on UltraFeedback pairs from your SFT model. You have their training logs with a short table of the final eval metrics, the length statistics of the preference data through your tokenizer, and a probe report comparing SFT and both DPO models on the same prompts.

## What you need

- `hf/my-gpt-350m-sft/` from [09.2](../module-09/lesson-02.md) in your bucket, and the Module 9 tools (`sft_data_stats.py`, `probe_chat.py`) in `tools/`.
- A 1× H100 pod from your `mygpt-gpu` template (the one from 09.2 if it is still running), with `pip install -e ".[post,eval]"`.
- About 2 hours wall clock, ~1–1.5 h on the GPU.
- **Expected cost: ~$3–4** on Community Cloud at $2.69/h (~$5–7 on Secure at $4.39/h). The data is free.

> [!WARNING]
> Same rule as every GPU lesson: arm the hard stop and the idle guard first, push each model to the bucket as soon as it is saved, and terminate the pod when you are done (or keep it only if you go straight on to 10.2).

## Choices and costs

### The preference data (checked 2026-11-13)

| Dataset | Size | License | Format for `dpo.py` | Fit |
|---|---|---|---|---|
| **`trl-lib/ultrafeedback_binarized`** | 62.1k train / 1k test pairs | Card is empty; it repackages the UltraFeedback binarization, which HuggingFaceH4 publishes under MIT | Ready: conversational `chosen` / `rejected`, scores 1–10 | **Recommended.** `dpo.py` default, the data SmolLM2-360M-Instruct was aligned with |
| `HuggingFaceH4/ultrafeedback_binarized` | 61,135 `train_prefs` / 2,000 `test_prefs` | MIT | Has a string `prompt` column next to message lists; `dpo.py` fails on that mix ("'list' object has no attribute 'endswith'", reproduced 2026-11-13) | Same data; needs the conversion in step 4 |
| `argilla/ultrafeedback-binarized-preferences-cleaned` | 60.9k | MIT | Same string-`prompt` issue | Cleaner: pairs chosen by average rating, TruthfulQA prompts removed. Use it if you report TruthfulQA |
| `allenai/llama-3.1-tulu-3-8b-preference-mixture` | 273k | ODC-BY; some parts non-commercial | Conversational pairs plus a string `prompt` | On-policy data for an 8B model; much larger than a 350M model can use |
| `HuggingFaceTB/smoltalk2` (preference) | 447k (from the Tulu 3 mixture), `think` / `no_think` | Apache-2.0 for new data; sources keep theirs | Needs conversion | Built for SmolLM3-3B with reasoning traces |

How UltraFeedback pairs are made matters for what DPO will learn: four model answers per prompt were scored by GPT-4; the best becomes `chosen`, a random one of the other three `rejected` (H4 card). So the pairs reward what GPT-4 liked in 2023, including, on average, longer answers. Measured through the chat template on 3,000 pairs (GPT-2 tokenizer, 2026-11-13):

| Statistic | Value |
|---|---|
| Prompt tokens, mean / p90 | 188 / 453 |
| Chosen / rejected answer tokens, mean | 309 / 271 |
| Pairs where chosen is longer | 55.5% (by 38 tokens on average) |
| Pairs longer than 1,024 tokens (prompt + answer) | 10.6% |
| Tokens processed per pair (both sequences, capped at 1,024) | ~905 |

The length skew is mild, but DPO amplifies whatever separates chosen from rejected. Watch answer length in 10.2.

### Two runs instead of one

| Setting | Run A: SmolLM2-360M recipe | Run B: `dpo.py` defaults |
|---|---|---|
| beta | 0.5 (stays closer to the SFT model) | 0.1 |
| Learning rate | 1e-6 | 5e-7 |
| Epochs | 2 | 1 |
| Batch | 8 × 8 accumulation = 64 pairs (recipe: 128 on 8 GPUs) | 64 pairs |
| Steps | ~1,940 | ~970 |
| Max length | 1,024 | 1,024 |

$\beta$ sets how expensive drifting from the reference model is: the implicit reward of an answer is $\beta \log \frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$, so with a larger $\beta$ the same reward needs a smaller change in log-probability. There is no way to tell from the training loss which run makes the better chat model; that is exactly what 10.2's judge is for. Two runs cost about $1.50 more than one.

### Memory and compute of the reference model

DPO needs the log-probabilities of every chosen and rejected answer under the frozen SFT model. TRL loads a second copy of `--model` for that, in the same dtype as the policy.

| Item | Size for 337M params |
|---|---|
| Policy: fp32 weights + gradients + AdamW state (16 B/param) | ~5.4 GB |
| Reference: fp32 weights, forward only | ~1.35 GB |
| Activations for 8 pairs × 2 sequences × up to 1,024 tokens | the bulk; tens of GB without gradient checkpointing |

Total compute per token is about $6N$ for the policy plus $2N$ for the reference forward pass, so $8N$: $8 \times 337\text{M} \times 56\text{M tokens per epoch} \approx 1.5 \times 10^{17}$ FLOPs, about 0.2 GPU-hours per epoch at an assumed 20% MFU (ESTIMATE). `precompute_ref_log_probs=True` would run the reference once over the data and drop it from memory; that matters for 7B models (a 14 GB bf16 reference), not for 1.35 GB. Leave it off.

> [!IMPORTANT]
> As in 09.2, the scripts must train fp32 weights with bf16 autocast. DPO learning rates are 50–100× smaller than SFT's: with bf16-stored weights, a step of 5e-7 changes only about 1% of the weights, and the run would show rewards stuck near zero.

### Time and money (prices checked 2026-11-13)

| Part | GPU time (ESTIMATE) | Community $2.69/h | Secure $4.39/h |
|---|---|---|---|
| Setup or reuse of the 09.2 pod, data download (~0.3 GB) | 0.2 h | $0.55 | $0.90 |
| Smoke test (20 steps) | 0.1 h | $0.27 | $0.44 |
| Run A, 2 epochs, with evals | 0.5–0.8 h | $1.35–2.15 | $2.20–3.50 |
| Run B, 1 epoch, with evals | 0.3–0.4 h | $0.80–1.10 | $1.30–1.75 |
| Probe report, uploads | 0.2 h | $0.55 | $0.90 |
| **Total** | **~1.3–1.7 h** | **~$3.50–4.60** | **~$5.75–7.50** |

> [!NOTE]
> Re-check the H100 price on the RunPod deploy page and put the measured hours into `budget.md` after each run.

## Do this

### 1. Box, stops, models (10 min)

On a fresh pod (skip what is already done if you kept the 09.2 pod):

```bash
nohup bash -c 'sleep 3h; runpodctl pod stop $RUNPOD_POD_ID' >/tmp/hardstop.log 2>&1 &
git clone https://github.com/<you>/<your-repo>.git ~/mygpt && bash ~/mygpt/tools/setup_box.sh
cd ~/mygpt && . .venv/bin/activate && uv pip install -e ".[post,eval]"
tmux new -d -s guard "bash scripts/autoshutdown.sh --idle-minutes 20 --grace-minutes 15 2>&1 | tee /tmp/guard.log"
bash scripts/sync.sh pull-hf my-gpt-350m-sft hf/my-gpt-350m-sft
grep -n "dtype=" scripts/dpo.py        # must show dtype=torch.float32
mkdir -p logs reports
```

### 2. Look at the pairs through your tokenizer (5 min)

```bash
python tools/sft_data_stats.py --model hf/my-gpt-350m-sft --dataset trl-lib/ultrafeedback_binarized \
  --kind dpo --n 3000 --epochs 2 --json reports/dpo_data_stats.json
```

Compare with the table above: `share_chosen_longer`, `chosen_minus_rejected_mean` and `share_over_max_length` should be close. Then read five pairs and decide for yourself whether `chosen` is better:

```python
from datasets import load_dataset
ds = load_dataset("trl-lib/ultrafeedback_binarized", split="test").shuffle(seed=3).select(range(5))
for r in ds:
    print("PROMPT:", r["chosen"][0]["content"][:400])
    print(f"--- CHOSEN ({r['score_chosen']}):", r["chosen"][-1]["content"][:500])
    print(f"--- REJECTED ({r['score_rejected']}):", r["rejected"][-1]["content"][:500], "\n" + "=" * 80)
```

### 3. Smoke test: does anything move? (5 min, ~$0.25)

```bash
python scripts/dpo.py --model hf/my-gpt-350m-sft --out runs/dpo_smoke --beta 0.5 --lr 1e-6 \
  --max-samples 2000 --eval-split '' --set max_steps=20 --set logging_steps=2 \
  --set save_strategy=no --set gradient_checkpointing=false 2>&1 | tee logs/dpo_smoke.log
```

The first logged line has `'loss': '0.6931'` and every reward at `'0'`: at step 0 the policy equals the reference, so the loss is exactly $\ln 2$. Here is that line from a tiny test model (2026-11-13; your logps are less negative, because your model is not random):

```text
{'loss': '0.6931', 'grad_norm': '15.12', 'learning_rate': '0', 'entropy': '10.8', 'num_tokens': '2719', 'logits/chosen': '-0.2058', 'logits/rejected': '-0.2042', 'mean_token_accuracy': '0.02814', 'rewards/chosen': '0', 'rewards/rejected': '0', 'rewards/accuracies': '0', 'rewards/margins': '0', 'logps/chosen': '-3446', 'logps/rejected': '-2818', 'epoch': '0.06667'}
```

By step 20, `rewards/chosen` and `rewards/rejected` must be non-zero (small, well under 1) and `loss` slightly different from 0.6931. If every reward is still `'0'` after warmup, stop: the update is not reaching the weights (step 1's `grep`, or a learning rate of 0 from a typo). `logps/*` are summed over the answer's tokens, which is why they are in the hundreds or thousands.

### 4. Optional: use a different preference dataset

Skip this unless you chose an alternative. The H4 and Argilla versions need their `chosen`/`rejected` columns only (the message lists already include the user turn):

```python
import json, os
from datasets import load_dataset
os.makedirs("data", exist_ok=True)
ds = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned", split="train")
with open("data/prefs.jsonl", "w", encoding="utf-8") as f:
    for r in ds:
        f.write(json.dumps({"chosen": r["chosen"], "rejected": r["rejected"]}) + "\n")
```

Then pass `--dataset data/prefs.jsonl` below (a local file has no eval split, so you lose the eval metrics; split off 1,000 lines yourself if you want them).

### 5. Run A: the SmolLM2 recipe (30–50 min)

```bash
tmux new -s dpo
python scripts/dpo.py --model hf/my-gpt-350m-sft --out hf/my-gpt-350m-dpo --beta 0.5 --lr 1e-6 --epochs 2 \
  --set gradient_checkpointing=false --set run_name=dpo_b05 --report-to wandb 2>&1 | tee logs/dpo_b05.log
```

Watch these, in this order of importance (every 500 steps the `eval_` versions on 1,000 held-out pairs are the honest ones):

| Metric | Healthy (ESTIMATE, typical shape) | Warning sign |
|---|---|---|
| `rewards/accuracies` | rises from 0.5 to roughly 0.6–0.75 | eval stays at ~0.5 (nothing learned), or train near 1.0 while eval is flat (memorizing) |
| `rewards/margins` | grows steadily above 0 | negative or flat |
| `rewards/chosen`, `rewards/rejected` | both often drift below 0, rejected faster | chosen falling as fast as rejected |
| `logps/chosen` | falls a little from its starting value | falls by a large fraction: the model is unlearning even the good answers (over-optimization; answers get long, repetitive or odd) |
| `loss` | from 0.693 down to roughly 0.5–0.65 | below ~0.3 on a small model usually means memorized pairs |
| `grad_norm` | stable band | spikes, `nan` |

The TRL docs say the same thing in one sentence: DPO usually wins by pushing down the rejected answers rather than pushing up the chosen ones. A falling `rewards/chosen` is therefore not by itself a problem; a collapsing `logps/chosen` is.

When the run ends, push it at once and summarize the evals:

```bash
bash scripts/sync.sh push-hf my-gpt-350m-dpo hf/my-gpt-350m-dpo
python - <<'EOF'
import ast
for line in open("logs/dpo_b05.log", encoding="utf-8"):
    if "'eval_loss'" in line:
        d = ast.literal_eval(line[line.index("{"):].strip())
        print({k: d.get(k) for k in ("epoch", "eval_loss", "eval_rewards/accuracies",
                                     "eval_rewards/margins", "eval_logps/chosen", "eval_logps/rejected")})
EOF
```

### 6. Run B: the script defaults (15–25 min)

```bash
python scripts/dpo.py --model hf/my-gpt-350m-sft --out hf/my-gpt-350m-dpo-b01 \
  --set gradient_checkpointing=false --set run_name=dpo_b01 --report-to wandb 2>&1 | tee logs/dpo_b01.log
bash scripts/sync.sh push-hf my-gpt-350m-dpo-b01 hf/my-gpt-350m-dpo-b01
```

Run the same summary snippet on `logs/dpo_b01.log` and paste both tables into `reports/dpo_runs.md`, with the measured GPU time of each run.

### 7. Read the three models side by side (15 min)

```bash
python tools/probe_chat.py --models hf/my-gpt-350m-sft hf/my-gpt-350m-dpo hf/my-gpt-350m-dpo-b01 \
  --out reports/dpo_probe.md
bash scripts/sync.sh push logs ckpt/dpo_350m/logs && bash scripts/sync.sh push reports ckpt/dpo_350m/reports
```

Expect differences to be subtle: slightly longer, more structured or more polite answers, sometimes a better refusal of the lock-picking probe. If one DPO model is much longer or repeats itself, note it; 10.2 measures whether the judge prefers it or just its length.

### 8. Clean up

```bash
rm -rf hf/my-gpt-350m-dpo*/checkpoint-* runs/dpo_smoke
git add reports/dpo_runs.md reports/dpo_probe.md reports/dpo_data_stats.json && git commit -m "DPO runs" && git push
```

Keep the pod if you start 10.2 now (it generates answers and runs lm-eval on the GPU); otherwise check the bucket and terminate.

## How you know it worked

- The smoke test started at loss 0.6931 with zero rewards and showed non-zero rewards by step 20.
- Both runs end with `saved DPO model to ...`, and the bucket has `model.safetensors` under `hf/my-gpt-350m-dpo/` and `hf/my-gpt-350m-dpo-b01/`.
- Final `eval_rewards/accuracies` is above 0.55 for at least one run and `eval_rewards/margins` is positive; `eval_logps/chosen` did not collapse (it fell by a small fraction of its starting value, not by half).
- `reports/dpo_runs.md` has both eval tables and the measured GPU hours; `reports/dpo_probe.md` has three answers per probe and your notes on length and tone.

## When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| All rewards stay `0` and loss stays 0.6931 | Weights not updated: bf16 master weights, or learning rate 0 | `grep -n "dtype=" scripts/dpo.py`; check the `learning_rate` column is non-zero after warmup |
| `AttributeError: 'list' object has no attribute 'endswith'` | Dataset has a string `prompt` column next to message lists (H4, Argilla, Tulu) | Convert to `chosen`/`rejected` only (step 4) |
| `Dropping fully truncated examples` removes many rows | Prompts longer than `--max-length` | Expected for a few percent; if it is more, you passed a smaller `--max-length` |
| `eval_rewards/accuracies` ~0.5 throughout | Learning rate too small for this beta, or the SFT model cannot tell answers apart at all | Compare with run B; try `--lr 2e-6` with beta 0.5 as a third run (~$1) |
| Train accuracies near 1.0, eval flat, answers get long and repetitive | Over-optimization on 62k pairs | Prefer the 1-epoch or higher-beta model; never pick a DPO model by its training metrics |
| `CUDA out of memory` | Long pairs, two sequences each, no gradient checkpointing | `--batch-size 4 --grad-accum 16` (still 64 pairs per step) |
| Pod stopped mid-run, model lost | Idle guard during a long dataset tokenization on CPU, or the hard stop | Raise `--grace-minutes`; push after every run; rerun the lost one (~$1–2) |

## Artifact

- `r2:mygpt/hf/my-gpt-350m-dpo/` and `r2:mygpt/hf/my-gpt-350m-dpo-b01/` (full models with tokenizer and chat template); logs and reports under `r2:mygpt/ckpt/dpo_350m/`.
- In git: `reports/dpo_data_stats.json`, `reports/dpo_runs.md` (settings, final eval metrics, GPU time per run), `reports/dpo_probe.md`.
- `budget.md` line:

  ```text
  | 2026-11-13 | 10.1 | RunPod 1x H100 SXM Community, DPO smoke + 2 runs + probes ~1.5 h at $2.69/h | ~$4.00 | $____ |
  ```
