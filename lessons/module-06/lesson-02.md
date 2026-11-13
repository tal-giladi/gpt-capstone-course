---
id: "06.2"
module: 6
minutes: 20
practice_minutes: 60
prerequisites: ["06.1"]
objectives:
  - Map each architecture switch in mygpt to the change it makes and to its cost in parameters and FLOPs.
  - Run single-change ablations and the two exportable bundles with the queue, and read the table against the noise threshold.
  - Apply the export rule so the chosen architecture maps to GPT2LMHeadModel, LlamaForCausalLM or Qwen3ForCausalLM.
  - Choose the main-run architecture and write the decision with its evidence.
volatility: implementation
sources:
  - title: "Touvron et al., LLaMA: Open and Efficient Foundation Language Models (pre-norm RMSNorm, SwiGLU, RoPE)"
    url: https://arxiv.org/abs/2302.13971
  - title: "Qwen3 Technical Report (QK-Norm added, QKV bias removed)"
    url: https://arxiv.org/html/2505.09388v1
  - title: "Wortsman et al., Small-scale proxies for large-scale Transformer training instabilities (qk-layernorm)"
    url: https://arxiv.org/abs/2309.14322
  - title: "Keller Jordan et al., modded-nanogpt (rotary, QK-norm, untied head in the speedrun records)"
    url: https://github.com/KellerJordan/modded-nanogpt
  - title: "Radford et al., Language Models are Unsupervised Multitask Learners (GPT-2)"
    url: https://cdn.openai.com/better-language-models/language-models.pdf
last_verified: "2026-11-13"
---

# 06.2 · Architecture ablations

GPT-2's architecture is from 2019. The models that replaced it (Llama, Qwen and most open models since) changed a handful of parts: rotary position embeddings instead of learned ones, RMSNorm instead of LayerNorm, a SwiGLU MLP instead of GELU, no bias terms, and in Qwen3 a normalisation on queries and keys. `mygpt` has a switch for each. In this lesson you turn them on one at a time against the control, then as the two bundles that a standard Hugging Face class can load, and pick your architecture from the numbers rather than from fashion.

## Goal

`results/ablations.md` contains the six single-change runs and the two exportable bundles (`abl_llama`, `abl_qwen3`) against the control, each labelled better / worse / within noise, with parameter counts and tokens/s. Below the table you have written which architecture the main run uses and why, and it passes the export rule.

## What you need

- The two controls, noise threshold and tools from [06.1](lesson-01.md); the ablation box (keep it from 06.1 or rent a new one the same way).
- About 6 hours of wall clock for 8 runs, almost all unattended; ~30 minutes of your time.
- **Expected cost: $13–15** (8 runs × ~35 min on 1×H100 SXM Community at $2.69/h, plus setup).

## Choices and costs

### The switches

All runs start from `configs/ablation_base.yaml` (GPT-2 architecture, 32k vocabulary, 111.0M parameters, 775 MFLOP per token for training) and change only what the row says. Parameter and FLOP numbers below come from `mygpt.model.GPT` at sequence length 1024.

| Run | Override | What changes | Params | Training FLOPs/token | Used in |
|---|---|---|---|---|---|
| `abl_rope` | `model.pos_emb=rope` | Rotary position embedding inside attention replaces the learned 1024×768 position table | 110.2M | ~775M | Llama, Qwen, speedrun |
| `abl_rmsnorm` | `model.norm=rmsnorm` | RMSNorm (no mean subtraction, no bias) replaces LayerNorm | ~111.0M | ~775M | Llama, Qwen |
| `abl_swiglu` | `model.mlp=swiglu` | Gated SiLU MLP with hidden 8/3 × 768 = 2,048 instead of GELU with 3,072; same parameters | 111.0M | ~775M | Llama, Qwen, PaLM |
| `abl_qknorm` | `model.qk_norm=true` | Normalises queries and keys per head before the dot product | ~111.0M | ~775M | Qwen3, speedrun |
| `abl_nobias` | `model.bias=false` | Removes all bias vectors from linear layers and norms | ~110.9M | ~775M | Llama, Qwen3 |
| `abl_untied` | `model.tie_embeddings=false` | Separate output matrix instead of reusing the input embedding | 136.2M (+25.2M) | 775M (logged MFU reads ~926M, see below) | speedrun, most large models |
| `abl_llama` | rope + rmsnorm + swiglu + `bias=false` | The Llama bundle | 110.1M | ~774M | exports to `LlamaForCausalLM` |
| `abl_qwen3` | the Llama bundle + `qk_norm=true` | The Qwen3 bundle | 110.1M | ~774M | exports to `Qwen3ForCausalLM` |

Untying adds parameters but no matmul work (the input embedding is a table lookup). The training log's MFU uses 6 × parameters, which counts the second matrix as if it were multiplied; compare **tokens/s**, not MFU, for that run.

### The export rule

Your final model has to load with `AutoModelForCausalLM` so that lm-eval, TRL, the Hub and the demo work without custom code. `mygpt.export_hf` maps exactly three combinations and refuses everything else as an "ablation-only combination":

| Combination | Exports to |
|---|---|
| All GPT-2 settings (learned, layernorm, gelu, bias, no qk_norm) | `GPT2LMHeadModel` |
| `rope` + `rmsnorm` + `swiglu` + `bias=false` | `LlamaForCausalLM` |
| the same + `qk_norm=true` | `Qwen3ForCausalLM` |

`tie_embeddings` may be true or false in all three. So the single-change runs tell you **where** a gain comes from, but the decision is between three architectures: GPT-2, Llama or Qwen3, each tied or untied. RoPE alone, for example, cannot be your final model.

### What to expect (and what not to believe)

PUBLICLY DOCUMENTED: Llama adopted pre-norm RMSNorm, SwiGLU and RoPE; Qwen3 removed the QKV bias and added QK-Norm for training stability; Wortsman et al. show QK-layernorm mostly widens the range of learning rates that train stably; the speedrun's early records credit rotary embeddings, QK-norm and an untied head. ESTIMATE for your 1B-token runs: RoPE and SwiGLU each give small gains (in the order of 0.005–0.02 bpb), RMSNorm and no-bias are about neutral in loss, QK-norm is about neutral at the control's LR (its value shows up when you raise the LR in 06.3), and the bundles are at least as good as their best single part. Effects this size are close to your threshold; that is normal, and it is why the bundles and the threshold matter more than any single row.

> [!NOTE]
> Run cost uses the RunPod Community H100 SXM price checked on 2026-11-13 ($2.69/h). Check the live price, and update the per-run cost in `budget.md` if it changed.

## Do this

### 1. Activate the 06.2 block (2 min)

In `tools/ablations.txt`, remove the leading `# ` from the eight lines of the two 06.2 blocks (the comment headers can stay). Check what will run:

```bash
DRY_RUN=1 bash tools/run_ablations.sh tools/ablations.txt
```

The controls show `DONE, skipping`; eight new commands are printed, each with exactly the overrides you expect.

### 2. Run the queue (5 min, then ~5 hours unattended)

On the ablation box (if it is new: hard stop, `setup_box.sh`, idle guard and shard pull as in 06.1 steps 2–3; make the hard stop `sleep 7h` for this block):

```bash
cd ~/mygpt && git pull && . .venv/bin/activate
tmux new -s queue
WANDB_PROJECT=my-gpt-ablations bash tools/run_ablations.sh tools/ablations.txt 2>&1 | tee -a runs/queue.log
```

Check the first minute of each new run when you can: the `params` in the header must match the table above (110.2M for RoPE, 136.2M for untied, 110.1M for the bundles). A wrong parameter count means a wrong override. You may split the block over two sessions: terminate, rent again later, rerun the same command.

### 3. Build and read the table (15 min)

```bash
python tools/ablation_table.py "runs/abl_*" --queue tools/ablations.txt --price-per-gpu-hour 2.69 | tee results/ablations.md
```

In W&B, group the project by run and overlay `val_bpb` for the control and the bundles: the curves should separate early and stay separated. Then answer, in a few lines under the table:

1. Which single changes are better beyond the threshold, which are within noise, which are worse?
2. Is `abl_qwen3` or `abl_llama` better than the control beyond the threshold? Is the difference between the two bundles beyond the threshold?
3. Is `abl_untied` worth 25M extra parameters? At the main-run size the extra output matrix is 32,768 × 1,024 = 33.5M parameters (+10% on 337M), more memory in SFT and a larger download.

### 4. Choose the architecture (10 min)

Use this order:

| Situation | Choice |
|---|---|
| A bundle beats the control beyond the threshold | That bundle. If both do and they are within noise of each other, **Qwen3**: same loss, and QK-norm gives LR headroom you can use in 06.3 and stability insurance for a 20B-token run |
| Both bundles within noise of the control | **Qwen3** anyway is defensible (standard, stable, exportable), but write down that your data did not show a gain |
| A bundle is worse beyond the threshold | Stop and check the runs (headers, data, a resumed run with a changed setting) before believing it; then GPT-2 architecture |
| Untied better beyond the threshold | Add `tie_embeddings: false` if you accept +10% parameters at 350M; otherwise keep tied |

Write the choice as one line in `results/ablations.md`, for example: `Architecture for the main run: Qwen3 bundle, tied (abl_qwen3 -0.0xx bpb vs control, threshold 0.00x).`

### 5. Keep going or stop

Go straight on to 06.3 on the same box, or terminate it (all logs and `DONE` markers are in the bucket).

## How you know it worked

- `runs/queue.log` shows eight new `ok` runs; none `FAILED`.
- Every run in `results/ablations.md` was compared at step 1,907 with a verdict; parameter counts in the headers matched the table.
- The tokens/s column is within a few percent across runs, except where the change really costs compute (none of these should cost more than ~5%).
- Your decision line names an exportable combination: GPT-2, Llama or Qwen3, tied or untied.

## When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| A run shows the same val bpb as the control to 4 decimals | The override did not apply (typo in the key, or the line still commented) | `grep params runs/<name>/train.out`; the queue prints each command, check it; fix the line and rerun under a new name |
| `KeyError: unknown ModelConfig keys` | Misspelled key, for example `model.qknorm` | Use the exact keys: `pos_emb`, `norm`, `mlp`, `qk_norm`, `bias`, `tie_embeddings` |
| `abl_untied` is ~15% slower than the others | Extra memory traffic for the second 25M matrix, or MFU misread | Compare tokens/s; a few percent slower is expected, much more points to the host |
| A bundle diverges (NaN) at the control's LR | Unusual at 6e-4, but possible with a bad host or data | Look at grad norms before the NaN; rerun once; if it repeats, report it and use the other bundle |
| You want "RoPE + GELU" because it scored best | Not exportable | Treat it as evidence that RoPE helps; choose the exportable bundle that contains RoPE |

## Artifact

- `results/ablations.md` (version 2) committed: control, six single changes, two bundles, verdicts, and your architecture decision with its numbers.
- `tools/ablations.txt` with the 06.2 block active, committed (so the queue history is in git).
- `budget.md` ledger line:

  ```text
  | 2026-11-13 | 06.2 | RunPod 1x H100 SXM Community, ~5 h at $2.69/h, 8 architecture ablations | ~$14 | $____ |
  ```
