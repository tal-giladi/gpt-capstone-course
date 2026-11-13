---
id: "08.2"
module: 8
minutes: 30
practice_minutes: 75
prerequisites: ["08.1", "03.3"]
objectives:
  - Evaluate OpenAI GPT-2 124M and 355M under exactly your harness settings and put all four models in one table.
  - Measure bits per byte of all four models on identical held-out text, and explain what that number does and does not show.
  - State the contamination status of every task for your models and for OpenAI's, from your Module 3 reports and an optional extra scan.
  - Decide which differences are real with standard errors and paired bootstrap intervals, and word each claim accordingly.
volatility: implementation
sources:
  - title: "Radford et al., Language Models are Unsupervised Multitask Learners (GPT-2; WebText, 8-gram overlap analysis)"
    url: https://cdn.openai.com/better-language-models/language-models.pdf
  - title: "openai-community/gpt2 on the Hugging Face Hub"
    url: https://huggingface.co/openai-community/gpt2
  - title: "openai-community/gpt2-medium on the Hugging Face Hub"
    url: https://huggingface.co/openai-community/gpt2-medium
  - title: "Karpathy, build-nanogpt play.ipynb (OpenAI GPT-2 124M val loss 3.2924 on FineWeb-Edu; HellaSwag 0.2945 / 0.3752)"
    url: https://raw.githubusercontent.com/karpathy/build-nanogpt/master/play.ipynb
  - title: "lm-evaluation-harness quickstart (stderr reported by bootstrap)"
    url: https://lm-evaluation-harness.readthedocs.io/getting_started/quickstart/
  - title: "Brown et al., Language Models are Few-Shot Learners (GPT-3; 13-gram contamination analysis, appendix C)"
    url: https://papers.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html
  - title: "Dror et al., The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing (ACL 2018; bootstrap and paired tests)"
    url: https://aclanthology.org/P18-1128/
last_verified: "2026-11-13"
---

# 08.2 · Comparing against GPT-2 and your baseline

The question the whole course has been building towards is simple to ask: is your model better than OpenAI's GPT-2 of the same size? Answering it honestly takes three things. The reference models must be measured by you, with your settings, not copied from a paper. The language-modelling number must be computed on the same text for every model, which across different tokenizers means bits per byte, not loss. And every difference needs an error bar and a note on what could have leaked into training. This lesson does all three on the box from 08.1, and ends with a table in which every "better" is earned.

## Goal

One table, `reports/eval_table.md`, with your 350M model, your 124M baseline, OpenAI GPT-2 124M and GPT-2 medium (355M) on six tasks under identical settings, plus paired bootstrap intervals for the two comparisons that matter (yours vs GPT-2 medium, baseline vs GPT-2). `reports/bpb.md` holds all four models' bits per byte on the identical common val text, and `reports/contamination.md` states per task what was checked and what was found. The GPU box is terminated.

## What you need

- The 1×H100 pod from [08.1](lesson-01.md) still running, with `evals/my-gpt-350m-base/` and `evals/baseline-124m/`.
- The common val sets `tok/common-val-gpt2/` and `tok/common-val-bpe32k/` and `TOKEN_REPORT.md` from [03.3](../module-03/lesson-03.md); the main checkpoint in `r2:mygpt/ckpt/main_350m/` from [07.3](../module-07/lesson-03.md).
- About 1.5 hours, ~1 hour of it on the pod.
- **Expected cost: ~$3** (about 1 h at $2.69/h, checked 2026-11-13). Optional extra contamination scan: ~$1 on a Hetzner CPU box.

## Choices and costs

### Three comparisons, three different questions

| Comparison | What differs | What it can show |
|---|---|---|
| Your baseline vs OpenAI GPT-2 124M | Training data (FineWeb-Edu 10B vs WebText) only; same architecture and tokenizer | Whether your reproduction is faithful (05.3's goal) |
| Your 350M vs your baseline | Size, recipe, tokenizer, 2× tokens | How much the whole package bought; not which part (that was Module 6) |
| **Your 350M vs OpenAI GPT-2 medium (355M)** | Data, tokens, tokenizer, recipe; same size class | The course's headline claim: a better 350M model from your recipe and data, at a known cost |

None of them isolates a single cause, and the report says so.

### Bits per byte on identical text

Per-token loss depends on the tokenizer: a tokenizer with longer tokens has fewer, harder predictions. Bits per byte converts the loss over a text into bits per UTF-8 byte of that text:

$$\text{bpb} = \frac{\text{loss (nats/token)}}{\ln 2} \cdot \frac{\text{tokens}}{\text{bytes}}$$

The common val set from 03.3 is the same documents tokenized twice, held out from both of your training sets. GPT-2 models read `common-val-gpt2`, your 350M reads `common-val-bpe32k`; each is evaluated on the whole split, so all four see the same bytes.

| Model | Expected bpb on FineWeb-Edu val | Basis |
|---|---|---|
| OpenAI GPT-2 124M | ~1.04 | Karpathy's 3.2924 nats on FineWeb-Edu ÷ ln 2 ÷ ~4.57 bytes/token (ESTIMATE) |
| OpenAI GPT-2 medium | ~0.95–0.99 | ESTIMATE |
| Your baseline 124M | ~0.97 | 05.3 |
| Your 350M | ~0.84–0.89 | 07.2 |

> [!IMPORTANT]
> Your models were trained on FineWeb-Edu and this text is FineWeb-Edu; OpenAI's were trained on WebText, a different mix. A lower bpb here shows that your model predicts **this kind of text** better. It is a fair comparison between your own models and a home-field advantage against OpenAI's. Say exactly that in the report; the benchmarks carry the cross-model claim.

### How big must a difference be?

A benchmark score is a mean over $n$ items, so its standard error is about $\sqrt{p(1-p)/n}$:

| Task | $n$ | SE at ~35% (points) | Unpaired 95% half-width of a difference |
|---|---|---|---|
| HellaSwag | 10,042 | 0.48 | ~1.3 |
| LAMBADA | 5,153 | 0.66 | ~1.8 |
| ARC-Easy | 2,376 | 0.98 | ~2.7 |
| PIQA (at ~65%) | 1,838 | 1.11 | ~3.1 |
| WinoGrande (at ~52%) | 1,267 | 1.40 | ~3.9 |
| ARC-Challenge (at ~23%) | 1,172 | 1.23 | ~3.4 |

Two models answer the **same** items, and their errors are correlated (hard items are hard for both). A paired bootstrap uses that: it resamples items, recomputes the difference of the two models on each resample 10,000 times, and takes the 2.5th and 97.5th percentiles. Its interval is usually noticeably narrower than the unpaired one. `tools/eval_table.py --compare` does this from the `samples_*.jsonl` files `--log_samples` wrote. A difference whose 95% interval excludes 0 may be called better; one that does not is "no measurable difference", however the point estimates look.

These intervals cover only the finite benchmark. They do not cover training noise: a second training seed of your 350M model would land somewhat differently (06.1 measured that spread at 124M on val bpb). With one run per model you cannot measure it here; say so.

### Contamination: what you know and what you do not

| Task | Your two models | OpenAI GPT-2 |
|---|---|---|
| HellaSwag, ARC-Easy, ARC-Challenge, PIQA | Checked in 03.3: training documents sharing a 13-gram with any item were removed from both shard sets (counts in `TOKEN_REPORT.md`) | Unknown. The GPT-2 paper ran an 8-gram overlap analysis of WebText against several test sets; WebText itself is unreleased |
| WinoGrande, LAMBADA | Not checked in Module 3. Optional step 5 scans for them | Unknown (LAMBADA passages come from novels; the GPT-2 paper reports some overlap for it) |

Removing documents after a 13-gram match removes the obvious leaks; paraphrased or translated copies stay. That is why the report states the method, not "clean".

## Do this

### 1. Evaluate OpenAI's GPT-2 with your settings (15 min, pod)

```bash
cd ~/mygpt && . .venv/bin/activate
TASKS=hellaswag,arc_easy,arc_challenge,piqa,winogrande,lambada_openai
for m in gpt2 gpt2-medium; do
  lm_eval --model hf --model_args pretrained=openai-community/$m,dtype=bfloat16 --tasks $TASKS --num_fewshot 0 \
    --batch_size 32 --seed 1234 --log_samples --output_path evals/$m 2>&1 | tee logs/lmeval-$m.log
done
```

Same command as 08.1 with only the model changed. The results files record `model_sha`, the exact revision of the weights; it goes into the report.

### 2. Pull the common val sets and the main checkpoint (5 min)

```bash
bash scripts/sync.sh pull-tok common-val-gpt2   data/tok/common-val-gpt2
bash scripts/sync.sh pull-tok common-val-bpe32k data/tok/common-val-bpe32k
bash scripts/sync.sh pull-ckpt main_350m runs/main_350m          # final checkpoint, ~2.9 GB
python -c "import json;[print(d, json.load(open(f'data/tok/{d}/meta.json'))['val_docs']) for d in ('common-val-gpt2','common-val-bpe32k')]"
```

Both sets must have the same number of documents.

### 3. Bits per byte for all four models (20 min)

The two OpenAI models through `tools/hf_bpb.py`, your two models natively through `mygpt.evaluate`, each over the whole split:

```bash
mkdir -p reports/bpb
for m in gpt2 gpt2-medium; do
  python tools/hf_bpb.py --model openai-community/$m --data-dir data/tok/common-val-gpt2 --eval-tokens all \
    --json reports/bpb/$m.json
done
python tools/hf_bpb.py --model hf/baseline-124m --data-dir data/tok/common-val-gpt2 --eval-tokens all \
  --json reports/bpb/baseline-124m-hf.json
python tools/hf_bpb.py --model hf/my-gpt-350m-base --data-dir data/tok/common-val-bpe32k --eval-tokens all \
  --json reports/bpb/my-gpt-350m-base-hf.json
```

`mygpt.evaluate --eval-tokens all` (or `-1`) covers the same full windows natively:

```bash
python -m mygpt.evaluate --ckpt runs/baseline_124m --data-dir data/tok/common-val-gpt2 \
  --eval-tokens all | tee reports/bpb/baseline-124m-native.json
python -m mygpt.evaluate --ckpt runs/main_350m --data-dir data/tok/common-val-bpe32k \
  --eval-tokens all | tee reports/bpb/my-gpt-350m-base-native.json
grep '"eval_tokens"' reports/bpb/*.json
```

By default `mygpt.evaluate` reads the training micro-batch per forward pass (64 sequences for the baseline, 32 for the main run), which fits the 80 GB H100; on a 24 GB card add `--batch 16`, which changes the speed, not the windows. The last line must show the same `eval_tokens` in the native and HF file of each model. Native and HF numbers for the same model must agree within ~0.002 bpb (bf16 rounding); a bigger gap means the export is not the model you trained. Then write `reports/bpb.md`: one row per model with val loss, bytes per token and bpb, and the home-field note from above.

### 4. The four-model table with intervals (10 min)

```bash
python tools/eval_table.py evals/my-gpt-350m-base evals/baseline-124m evals/gpt2 evals/gpt2-medium \
  --names mine-350m,baseline-124m,gpt2,gpt2-medium \
  --compare mine-350m:gpt2-medium --compare baseline-124m:gpt2 --compare mine-350m:baseline-124m \
  --out reports/eval_table.md
```

Read the comparison blocks row by row. For each task write one sentence in your notes: "+x.x points [lo, hi]: better" or "no measurable difference". Sanity checks: the GPT-2 124M HellaSwag acc_norm is ~31.1 (the lm-eval quickstart reports 0.3108), GPT-2 medium is higher than GPT-2 on every task except perhaps WinoGrande and ARC-Challenge, and your baseline-vs-GPT-2 block shows mostly "no measurable difference" or small gains.

### 5. Contamination statement (15 min, optional scan ~2 h unattended)

Copy the HellaSwag/ARC/PIQA numbers (items with an overlap, documents removed, share of tokens) for both shard sets from `TOKEN_REPORT.md` into `reports/contamination.md`. For WinoGrande and LAMBADA, either write "not checked", or scan for them on a cheap CPU box as in 03.3 (Hetzner CPX62, ~€0.18/h), on the **filtered** training shards:

```bash
python scripts/decontam.py --shards-or-parquet data/tok/bpe32k-100BT-30 --tokenizer data/tok/bpe32k-100BT-30 \
  --benchmarks winogrande,lambada_openai --out reports/decontam-wg-lambada-bpe32k.json
python scripts/decontam.py --shards-or-parquet data/tok/gpt2-10BT --tokenizer gpt2 \
  --benchmarks winogrande,lambada_openai --out reports/decontam-wg-lambada-gpt2.json
```

`decontam.py` downloads the splits lm-eval scores (WinoGrande `winogrande_xl` validation with the blank filled by the correct option, LAMBADA-OpenAI test passages). On a box without Hub access, write them elsewhere with `python tools/bench_texts.py --out reports/bench_wg_lambada.jsonl` and pass `--benchmark-file reports/bench_wg_lambada.jsonl` instead of `--benchmarks`.

Expect few or no WinoGrande hits and a handful of LAMBADA items matching common phrases. Read two or three flagged examples before you judge; with this scan you only report, you do not retrain.

### 6. Save and terminate (5 min)

```bash
bash scripts/sync.sh push evals ckpt/main_350m/evals
git add reports/ logs/lmeval-*.log evals/*/*/results_*.json && git commit -m "Eval: GPT-2 references, bpb, contamination" && git push
```

RunPod console → the pod → **Terminate**.

## How you know it worked

- `reports/eval_table.md` has four model columns, no settings warnings, item counts identical to 08.1, and three comparison blocks marked "paired bootstrap".
- OpenAI GPT-2 124M lands at HellaSwag acc_norm ~31.1 and LAMBADA acc ~33 in your table (your bf16 run may differ from published fp32 runs by a few tenths of a point).
- `reports/bpb.md` lists four bpb values from the same text, native and HF numbers for your models within ~0.002 of each other, and the home-field note.
- `reports/contamination.md` states, per task and per model, what was checked and with what method.

## When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `tokenizer mismatch` from `hf_bpb.py` | GPT-2 model pointed at `common-val-bpe32k`, or your model at `common-val-gpt2` | Each model reads the set made with its own tokenizer |
| bpb of an OpenAI model far above ~1.1 | Evaluated only a few windows, or `--seq-len` above 1,024 | Use `--eval-tokens all`; GPT-2's context is 1,024 |
| Native and HF bpb differ by more than ~0.005 | Different windows (`eval_tokens` differs between the two JSON files, e.g. one run without `--eval-tokens all`), or a broken export | Re-run both with `--eval-tokens all`; if it persists, recheck the export's logit test |
| `CUDA out of memory` in `mygpt.evaluate` | It uses the training micro-batch by default (64 for the baseline) | Add `--batch 16` (or 8); the windows and the result stay the same |
| `eval_table.py` warns about different `limit` or `n-shot` | One run used a test setting | Rerun that model with the settings in `reports/eval_settings.md`; never mix |
| A comparison says "unpaired, from stderr" | One of the runs lacks `--log_samples` | Rerun it with `--log_samples`; the paired interval is the one to report |
| Your 350M loses to GPT-2 medium on HellaSwag | Fewer tokens than needed, or a recipe problem | Report it as it is, with the interval; the bpb and curves show where the model stands. An honest loss is a valid result |

## Artifact

- In git: `reports/eval_table.md`, `reports/bpb.md` and `reports/bpb/*.json`, `reports/contamination.md` (plus `decontam-wg-lambada-*.json` if scanned), `evals/*/*/results_*.json`, `logs/lmeval-*.log`.
- In the bucket: `ckpt/main_350m/evals/` with samples for all four models.
- `budget.md`:

  ```text
  | 2026-11-18 | 08.2 | RunPod 1x H100 SXM Community, ~1 h at $2.69/h, GPT-2 references + bpb on common val | ~$3 | $____ |
  ```
