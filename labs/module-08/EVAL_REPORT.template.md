# Evaluation report: my-gpt-350m-base

Copy to `EVAL_REPORT.md` in your project repo (lesson 08.3). Replace every `<...>`. Every number
in this file must be traceable to a file in `evals/` or `reports/` that is committed or in the
bucket. Module 11's technical report and model card copy from here.

## 1. What was evaluated

| Model | What it is | Params | Training tokens | Tokenizer | Source |
|---|---|---|---|---|---|
| my-gpt-350m-base | Your main run, step <38146> | <337M> | <20.0B> FineWeb-Edu | own BPE 32,768 | r2:mygpt/hf/my-gpt-350m-base/ |
| baseline-124m | Your GPT-2 124M reproduction | 124M | 10B FineWeb-Edu | GPT-2 | r2:mygpt/hf/baseline-124m/ |
| gpt2 | OpenAI GPT-2 small (2019) | 124M | not published (WebText) | GPT-2 | openai-community/gpt2 @ <sha> |
| gpt2-medium | OpenAI GPT-2 medium (2019) | 355M | not published (WebText) | GPT-2 | openai-community/gpt2-medium @ <sha> |

## 2. Settings

lm-eval-harness <0.4.13>, `--model hf`, `dtype=bfloat16`, 0-shot, no `--limit`, `--batch_size 32`,
`--log_samples`, seed 1234, on <1x H100 SXM, RunPod Community>, <date>. No chat template.
Bits per byte: the whole common val set (<n> documents, <bytes> bytes) tokenized by each tokenizer;
`mygpt.evaluate` for native checkpoints, `tools/hf_bpb.py` for Hugging Face models; context 1,024.

## 3. Results

### Benchmarks (points, ± standard error)

<paste reports/eval_table.md>

### Bits per byte on identical text (lower is better)

| Model | Val loss (nats/token) | Bytes per token | Val bpb |
|---|---|---|---|
| my-gpt-350m-base | | | |
| baseline-124m | | | |
| gpt2 | | | |
| gpt2-medium | | | |

### Training curves

![Validation bits per byte vs training tokens](reports/figures/val_bpb.svg)
![In-training HellaSwag vs training tokens](reports/figures/hellaswag.svg)

## 4. Claims

Each claim with its evidence (table row, interval) and its strength: "clearly better" only when
the 95% interval of the difference excludes 0.

1. <my-gpt-350m-base vs gpt2-medium on HellaSwag: +x.x points [lo, hi], paired bootstrap>
2. <...>

## 5. Caveats

- Contamination: <HellaSwag, ARC-Easy, ARC-Challenge, PIQA: documents sharing a 13-gram with any
  item were removed from both training sets (TOKEN_REPORT.md: n docs, x%). WinoGrande and
  LAMBADA: <checked with tools/bench_texts.py: n items overlap | not checked>. OpenAI GPT-2:
  WebText contents unknown.>
- Bits per byte is measured on FineWeb-Edu text, the distribution your models were trained on and
  OpenAI's were not; it favours your models and is not a general claim about language modelling.
- Single training run per model: no seed variance for the models themselves; the intervals cover
  only the finite benchmark sizes.
- 0-shot log-likelihood scoring of base models; says nothing about chat quality (Modules 9-10).
- <bf16 vs fp32, anything you changed, anything that failed>

## 6. Honest-claims checklist

- [ ] Every model ran with the same harness version, tasks, shots, dtype and no `--limit`.
- [ ] Every number has a file behind it (`evals/*/results_*.json`, `reports/bpb-*.json`).
- [ ] Differences called "better" have a 95% interval that excludes 0; others say "no measurable difference".
- [ ] The metric for each task (acc or acc_norm) is named, and the same for all models.
- [ ] Published numbers quoted from other papers are not mixed into the same table as harness runs.
- [ ] Contamination status is stated per task.
- [ ] Training tokens and compute are stated next to every comparison with OpenAI's models.
- [ ] The eval cost is in budget.md.
