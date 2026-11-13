---
id: "06.4"
module: 6
minutes: 25
practice_minutes: 60
prerequisites: ["05.3", "06.3"]
objectives:
  - Compute the main model's parameters, training FLOPs and Chinchilla-optimal alternative, and justify training past 20 tokens per parameter.
  - Transfer batch size and learning rates from the 124M ablations to the ~350M model and state which parts are evidence and which are practice.
  - Check the main run's cost from measured speeds against the money left under the $500 cap and apply a go / shrink rule.
  - Freeze the recipe in a committed, exportable config with a short decision record.
volatility: implementation
sources:
  - title: "Hoffmann et al., Training Compute-Optimal Large Language Models (Chinchilla, arXiv 2203.15556)"
    url: https://arxiv.org/abs/2203.15556
  - title: "Brown et al., Language Models are Few-Shot Learners (GPT-3 model sizes, batch sizes and learning rates)"
    url: https://papers.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html
  - title: "Hu et al., MiniCPM (WSD scheduler; compute-optimal data-model ratio far above Chinchilla, arXiv 2404.06395)"
    url: https://arxiv.org/abs/2404.06395
  - title: "Karpathy, build-nanogpt play.ipynb (OpenAI GPT-2 350M HellaSwag 0.375224, GPT-3 350M 0.436)"
    url: https://raw.githubusercontent.com/karpathy/build-nanogpt/master/play.ipynb
  - title: "Wen, Hall, Ma, Liang: Fantastic Pretraining Optimizers and Where to Find Them (arXiv 2509.02046)"
    url: https://arxiv.org/abs/2509.02046v1
  - title: "RunPod GPU pricing"
    url: https://www.runpod.io/pricing
  - title: "Lambda pricing"
    url: https://lambda.com/pricing
  - title: "Vast.ai pricing"
    url: https://vast.ai/pricing
  - title: "Nebius prices"
    url: https://nebius.com/prices
last_verified: "2026-11-13"
---

# 06.4 · Sizing the main run and freezing the recipe

The main run is the biggest single cost in the course, roughly $170–200 on one 8×H100 node for about five hours, and it cannot be changed once it starts without paying again. So before Module 7, you settle every number in one config file: the size of the model, how many tokens it sees, the batch and learning rates carried over from your 124M experiments, and the architecture and optimizer your ablations chose. Then you check the price against what is left of your $500, prove the config exports to a standard Hugging Face class, and freeze it with a short written decision.

## Goal

`configs/main_350m.yaml` in your repo holds your recipe (architecture and optimizer from 06.2–06.3, ~337M parameters, ~20B tokens, batch and LRs transferred from 124M, `ckpt_remote` and W&B set), passes the export check on a tiny model with the same switches, and is tagged `recipe-v1`. A decision record `decisions/0001-main-run-recipe.md` states the choices and evidence, and `budget.md` shows the main-run estimate from your measured speeds next to the money left, with a go / shrink decision.

## What you need

- `results/ablations.md` with your architecture and optimizer decisions ([06.2](lesson-02.md), [06.3](lesson-03.md)).
- Measured speeds: `main_350m` per-GPU tokens/s from [04.3](../module-04/lesson-03.md) and the 8-GPU scaling efficiency from [05.3](../module-05/lesson-03.md); the ledger in `budget.md` with every actual so far.
- The token report for `tok/bpe32k-100BT-30` from [03.3](../module-03/lesson-03.md) (train tokens available).
- About 1.5 hours on your laptop. Optional: a 1×H100 for an LR-transfer check (~1.7 h).
- **Expected cost: $0** (laptop only), or **$4–5** with the optional LR check.

## Choices and costs

### How big, how long

`main_350m.yaml` is GPT-2-medium shaped: 24 layers, width 1,024, 16 heads, context 1,024, your 32,768-token vocabulary. With the Llama/Qwen3 bundle (SwiGLU hidden 2,752) and tied embeddings:

$$N = 24 \times (4 \cdot 1024^2 + 3 \cdot 1024 \cdot 2752) + 32{,}768 \times 1024 \approx 303.6\text{M} + 33.6\text{M} = 337.2\text{M}$$

(370.7M untied.) Training compute for $D = 2\times10^{10}$ tokens:

$$C \approx 6ND = 6 \times 3.372\times10^{8} \times 2\times10^{10} = 4.05\times10^{19}\ \text{FLOP}$$

Chinchilla's result is roughly 20 training tokens per parameter for the lowest loss at a fixed compute budget. For this model that is $20 \times 337\text{M} = 6.7$B tokens; you plan 20B, about 59 tokens per parameter. For the same compute, the Chinchilla-optimal model would be $N = \sqrt{C/120} \approx 580$M parameters on ~11.6B tokens.

| Option | Params | Tokens | Tokens/param | Why (not) |
|---|---|---|---|---|
| Chinchilla-optimal for 350M | 337M | 6.7B | 20 | Cheapest way to reach this model's "compute-optimal" loss, but the model keeps improving well past it and the data is there |
| Chinchilla-optimal for the same money | ~580M | ~11.6B | 20 | Slightly lower loss per dollar, but a bigger model to fine-tune, serve on a free CPU Space, and no same-size OpenAI model to beat |
| **Chosen: over-trained 350M** | 337M | 20B | 59 | Fixed size to beat GPT-2 medium (355M) on equal terms; cheaper SFT, DPO, evals and demo; one epoch of data you already have |
| Smaller fallback | ~240M (16 layers) | 20B | 83 | If the budget check below fails |

Over-training a smaller model is standard practice when the model will be used after training (REASONABLE INDUSTRY PRACTICE; MiniCPM reports a compute-optimal data-to-model ratio far above Chinchilla's when using WSD). Your target, from Karpathy's evaluation of the OpenAI weights: GPT-2 medium (350M) HellaSwag **0.375**, GPT-3 Medium (350M, 300B tokens) **0.436**.

Check the data: 20B tokens must fit in one epoch of `tok/bpe32k-100BT-30`. Your token report says how many train tokens it has (Module 3 projected 21–24B). If it has fewer than ~20.5B, set `train_tokens` to about 97% of what is there rather than repeating data.

### Batch and learning rates from 124M to 350M

| Setting | Transfer rule | Evidence level |
|---|---|---|
| Batch 524,288 tokens | Keep. GPT-3 used 0.5M tokens for both Small (125M) and Medium (350M); 20B / 524,288 = 38,146 steps | PUBLICLY DOCUMENTED (GPT-3 table) |
| AdamW `lr` (AdamW recipe) | Halve your best 124M LR. GPT-3: Small 6.0e-4, Medium 3.0e-4. If your sweep picked 1.2e-3 at 124M, use ~6e-4 | PUBLICLY DOCUMENTED pattern; your own number is an ESTIMATE |
| `muon_lr` (Muon recipe) | Keep your best 124M Muon LR. Muon's orthogonalised, shape-scaled update has about the same size at any width | REASONABLE PRACTICE (speedrun and Moonlight use one Muon LR across sizes); not tested here at 350M |
| AdamW `lr` for embeddings/head with Muon | Keep or lower slightly (e.g. 6e-4 → 4e-4) | REASONABLE PRACTICE |
| Warmup | ~250M tokens (~477 steps), as in the template | REASONABLE PRACTICE: long enough for Adam's moment estimates, short relative to 38k steps |
| Schedule | Your 06.3 choice; WSD with `decay_frac: 0.2`, `min_lr_ratio: 0.0` in the template | Your ablation |
| `micro_batch_size` | 32 on 8×H100 (accumulation 2), as measured in 04.3 | Your benchmark |

LR is the setting most likely to be off after a 3× size change. The optional step 5 tests two LRs at 350M for 0.5B tokens (~$4.50). It cannot find the optimum, but it catches a choice that is clearly too high (diverges or lags early) before you spend $180.

### The cost check (prices checked 2026-11-13)

$$t_{\text{train}} = \frac{2\times10^{10}}{8 \cdot r_{\text{main}} \cdot s}$$

With 04.3's example $r_{\text{main}} = 175{,}000$ tokens/s and a measured $s = 0.93$: $2\times10^{10} / (8 \times 175{,}000 \times 0.93) = 15{,}361$ s = 4.3 h of training. `budget_calc.py` adds 25% for compile, evals, checkpoints and one restart.

| Provider (8×H100 SXM node) | Price/h | Main run at ~5.3 billed hours |
|---|---|---|
| RunPod Secure Cloud | $35.12 | ~$187 |
| Lambda (from 2026-11-16) | $36.72 | ~$196 |
| Vast.ai verified datacenter host | ~$21–24 | ~$111–127 (check bandwidth fees and host reliability) |
| Nebius (HGX 8-GPU VM, $4.50/GPU-h) | ~$36 | ~$191 |

Then compare with what is left. Example: actuals through Module 6 of $105 leave $395. Still to come besides the main run: evaluation ~$12, SFT + DPO ~$25, storage ~$7, a dev box ~$15, so ~$59. That leaves $336 for the main run and contingency.

| Money left for main run + contingency, vs main-run estimate $M$ | Decision |
|---|---|
| ≥ 1.25 × $M$ | **Go** with ~337M × 20B |
| 1.10–1.25 × $M$ | Cut `train_tokens` to 15B (cost scales with D), or use a cheaper verified 8× host |
| < 1.10 × $M$ | Shrink to ~240M (`n_layer: 16`) on 20B tokens, or the plan's cheaper path (124–200M on 10B) |

> [!NOTE]
> Re-check the 8×H100 price on the provider pages before you freeze the decision, and again on launch day in 07.1. Lambda's price rises on 2026-11-16.

> [!WARNING]
> The main run is ~$180–200 in one go. Never launch it from an unfrozen config ("I'll just change the LR on the box"): every change after this lesson goes through a new commit and tag, so you always know what you paid for.

## Do this

### 1. Edit the config (15 min, laptop)

Open `configs/main_350m.yaml` and set:

- `model`: your switches from 06.2 (`pos_emb`, `norm`, `mlp`, `qk_norm`, `bias`, `tie_embeddings`); keep `n_layer: 24`, `n_head: 16`, `n_embd: 1024`, `block_size: 1024`, `vocab_size: 32768`.
- `train`: `optimizer`, `muon_lr`, `lr`, `weight_decay` (AdamW), `muon_weight_decay` (Muon), `schedule`, `decay_frac`, `min_lr_ratio` from 06.3 with the transfer rules above; `train_tokens` from the data check; `ckpt_remote: r2:mygpt/ckpt/main_350m`; `wandb_project: my-gpt`; `run_name: main_350m`.

Replace the header comment with three lines: what this run is, the date, and "see decisions/0001-main-run-recipe.md".

### 2. Count parameters and FLOPs (5 min)

```bash
python - <<'EOF'
import torch
from mygpt.config import load_config
from mygpt.model import GPT
c = load_config("configs/main_350m.yaml")
with torch.device("meta"):
    m = GPT(c.model)
n, t = m.num_params(), c.train
print(f"params {n/1e6:.1f}M | tokens {t.train_tokens/1e9:.1f}B | {t.train_tokens/n:.0f} tokens/param | "
      f"steps {t.train_tokens // t.total_batch_tokens} | 6ND {6*n*t.train_tokens:.2e} FLOP")
EOF
```

Expect `params 337.2M | tokens 20.0B | 59 tokens/param | steps 38146 | 6ND 4.05e+19 FLOP` for the tied Qwen3/Llama bundle (370.7M untied).

### 3. Prove the recipe exports (5 min)

Train a tiny CPU model with exactly the frozen model switches and export it:

```bash
SW=$(python -c "
from mygpt.config import load_config
m = load_config('configs/main_350m.yaml').model
print(' '.join(f'model.{k}={getattr(m, k)}' for k in ('pos_emb', 'norm', 'mlp', 'qk_norm', 'bias', 'tie_embeddings')))")
echo $SW
python -m mygpt.train --config configs/smoke_cpu.yaml $SW train.out_dir=runs/export_check
python -m mygpt.export_hf --ckpt runs/export_check --out runs/export_check_hf
```

The export prints a JSON block with `"architecture": "Qwen3ForCausalLM"` (or `LlamaForCausalLM` / `GPT2LMHeadModel`) and `"max_abs_logit_diff_fp32"` far below 1e-3. A `ValueError: ablation-only combination` means the config is not exportable: fix it now, not after the main run.

### 4. Run the cost check (10 min)

```bash
python tools/budget_calc.py --tps-baseline <05.3 per-GPU tok/s> --tps-ablation <04.3 ablation tok/s> \
  --tps-main <04.3 main tok/s> --price-1gpu 2.69 --price-8gpu <today's 8x per-GPU price> --scaling <05.3 s> \
  | tee reports/budget-main-run.md
```

Take the main-run row, add your ledger's actuals and the remaining fixed items, and apply the decision table. Write the result into `budget.md`, for example: `Main run: Qwen3-bundle 337M x 20B, est. $187 (RunPod Secure, 5.3 h). Left after it: $149 (80% of estimate). GO.`

### 5. Optional: a two-point LR check at 350M (~1.7 h, ~$4.50)

On a 1×H100 SXM Community box (setup as in 06.1, first 20 shards; a Vast.ai or Verda spot H100 from 06.1's table works the same), run the frozen config for 0.5B tokens at your chosen LR and at 2× that LR:

```bash
python -m mygpt.train --config configs/main_350m.yaml train.train_tokens=500_000_000 \
  train.warmup_tokens=100_000_000 train.micro_batch_size=32 train.hellaswag_every=0 train.eval_every=250 \
  train.ckpt_remote="" train.wandb_project=my-gpt-ablations train.out_dir=runs/lr350_a train.run_name=lr350_a
# second run: the same plus train.muon_lr=<2x> (Muon recipe) or train.lr=<2x> (AdamW recipe), out_dir/run_name lr350_b
python tools/ablation_table.py runs/lr350_a runs/lr350_b --control lr350_a --noise <06.1 seed spread>
```

If 2× is clearly better, raise the LR by ~1.5×; if it diverges or is clearly worse, keep yours. Terminate the box afterwards.

### 6. Write the decision record (15 min)

Create `decisions/0001-main-run-recipe.md`:

```markdown
# 0001 Main-run recipe

Date: <date>   Status: accepted   Config: configs/main_350m.yaml @ tag recipe-v1

### Context
Baseline GPT-2 124M (10B tokens): val bpb <x>, HellaSwag <x> (results.md).
Ablations at 124M-class, 1B tokens, 1xH100 (results/ablations.md): seed spread <x> bpb, threshold <x>.

### Decision
- Architecture: <Qwen3 | Llama | GPT-2> bundle, <tied | untied>, 24x1024, 16 heads, ctx 1024, vocab 32768 (own BPE). Exports to <class>.
- Optimizer: <Muon muon_lr x + AdamW lr x | AdamW lr x>, weight_decay <x> (AdamW), muon_weight_decay <x>, grad_clip 1.0.
- Schedule: <WSD decay_frac 0.2 to 0 | cosine to 10%>, warmup 250M tokens.
- Size: 337M params, <20B> tokens (59 tokens/param, ~3x Chinchilla), batch 524,288, 38,146 steps.

### Evidence
<one line per chosen change: delta vs control, verdict; LR transfer rule and the optional 350M check>

### Rejected
<changes that were within noise or worse, and why they are not in the recipe>

### Cost
Estimate $<x> on <provider> at $<x>/h, <x> h; money left after it $<x> (<x>% of estimate).
```

### 7. Freeze it (2 min)

```bash
git add configs/main_350m.yaml decisions/ results/ reports/budget-main-run.md budget.md
git commit -m "Freeze main-run recipe v1"
git tag recipe-v1 && git push && git push --tags
```

## How you know it worked

- The parameter script prints ~337M (or ~371M untied), 38,146 steps and ~4e19 FLOP.
- The export check names a standard class with a logit difference below 1e-3.
- `budget.md` has the main-run estimate from your measured speeds, the money left, and a GO / shrink line that follows the decision table.
- `git show recipe-v1 --stat` lists the config, the decision record and the ablation results; `train_tokens` fits inside your train split.

## When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `ValueError: ablation-only combination` at the export check | A switch outside the three exportable bundles (e.g. `bias: true` with RoPE) | Make the config match one bundle exactly; rerun step 3 |
| Parameter count far from 337M (or 371M untied) | Wrong `n_embd`, `n_layer` or `vocab_size` | Compare with the template; the switches move the count by about 1M at most, the shape must not change |
| Data check: fewer train tokens than `train_tokens` | Fewer files tokenized or decontamination dropped documents | Set `train_tokens` to ~97% of the train tokens in `meta.json`; one epoch, no repeats |
| Cost check says shrink, but you want 350M | Prices rose or the run was slower than planned | Cut tokens to 15B first (still 44 tokens/param); a cheaper verified 8× host second; the model size last |
| LR check: both 350M runs diverge | LR from 124M too high for 350M, or Muon with weight decay turned on | Halve the LR and recheck one run; check `muon_weight_decay` is 0.0 |
| Someone edits the config on the box later | The recipe was not frozen | Launch only from a commit at or after `recipe-v1`; record the commit in `budget.md` at launch (07.1) |

## Artifact

- `configs/main_350m.yaml` (your recipe), `decisions/0001-main-run-recipe.md`, `reports/budget-main-run.md`, and the final `results/ablations.md`, all in git at tag `recipe-v1`.
- `budget.md`: main-run estimate, money left, decision line, and (only if you ran step 5) the ledger line:

  ```text
  | 2026-11-13 | 06.4 | RunPod 1x H100 SXM Community, ~1.7 h at $2.69/h, 350M LR check (2 x 0.5B tokens) | ~$4.50 | $____ |
  ```
