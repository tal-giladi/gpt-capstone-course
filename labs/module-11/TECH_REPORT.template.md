# <Title: e.g. "my-gpt-350m: training a GPT-2-medium-class model from scratch for $___">

<Your Name> · <date> · Code: <GitHub repo URL> (tag `v1.0`) · Model: <HF URL> · Demo: <Space URL>

<!-- Technical report template (lesson 11.3). Every number must come from a file in your repo:
     cite it in brackets, e.g. [reports/eval_report.md]. Label every ESTIMATE as such. -->

## Abstract

<4-6 sentences: what you built (size, tokens, tokenizer), what it cost, the main result against
OpenAI GPT-2 and your own baseline (one or two numbers with standard errors), what post-training
changed (win rate with CI), and the one lesson most worth reading about.>

## 1. Goal and constraints

- Target: beat OpenAI GPT-2 of the same size on HellaSwag and validation loss.
- Budget: hard cap $500, planned at $4.00 per GPU-hour [budget.md].
- Constraint: the final model must export to a standard Hugging Face class.

## 2. Data

| | Value | Source |
|---|---|---|
| Dataset | FineWeb-Edu (ODC-By 1.0, Common Crawl ToU) | [DATA_CARD.md] |
| Subsets | `sample/10BT` (baseline), first <30> files of `sample/100BT` (main) | [manifests/] |
| Revision | `<sha>` | [manifests/*.json] |
| Size | <28.5> GB + <64.6> GB parquet | |
| Decontamination | <n> docs dropped (13-gram vs HellaSwag, ARC, PIQA) | [reports/decontam-*.json] |
| Validation | held-out split + common val set for bpb across tokenizers | [TOKEN_REPORT.md] |

## 3. Tokenizer

<Vocab 32,768 BPE trained on <n> docs; bytes per token vs GPT-2's on the same text; why it is
worth it (or not). Table from reports/compare_tokenizers.json.>

## 4. Architecture and ablations

| Run | Change vs baseline | Tokens | Val bpb | Δ | HellaSwag | Kept? |
|---|---|---|---|---|---|---|
| abl_base | GPT-2 124M | 1B | | | | |
| abl_rope | + RoPE | 1B | | | | |
| ... | | | | | | |

<Final recipe and why; seed-noise estimate; what you would test next. From results/ablations.md and
decisions/0001-main-run-recipe.md.>

## 5. Pretraining

<Config summary, hardware/provider/price, tokens/s and MFU measured, wall time, loss curve figure
(PNG in assets/), incidents (spikes, rollbacks, restarts) with what you did. From
reports/main-run-log.md and log.jsonl.>

## 6. Evaluation

<Main table copied from reports/eval_report.md: OpenAI GPT-2 124M and 355M, your baseline, your
base model; bpb on the common val set; contamination note. State which differences exceed 2
standard errors.>

## 7. Alignment (SFT and DPO)

<Data, epochs, learning rates chosen and why; probe examples (two good, two bad); win rate with CI,
position and length bias; regression table. From reports/sft_data.md, sft_probe.md,
alignment_results.md, winrate.json.>

## 8. Cost: planned vs actual

<Paste the output of `python tools/ledger_summary.py budget.md`, then a by-category table.>

| Category | Planned | Actual | GPU-hours | Note |
|---|---|---|---|---|
| Data + tokenization (CPU) | $10 | | | |
| Dev / debugging GPU | $40 | | | |
| Baseline 124M | $50 | | | |
| Ablations | $60 | | | |
| Main run | $160 | | | |
| Evaluation | $12 | | | |
| SFT + DPO | $25 | | | |
| Judge API | $5 | | | |
| Storage | $8 | | | |
| Hosting + demo | $0 | | | |
| **Total** | **$370 (+$130 contingency)** | | | |

Energy and CO2 (ESTIMATE): <GPU-hours x 0.7 kW x 1.2 PUE x grid intensity>.

## 9. What went wrong and what I learned

<5-8 bullets, each: what happened, what it cost (time or $), what you would do differently.>

## 10. Reproducibility

| Artifact | Where | Pin |
|---|---|---|
| Code | <GitHub URL> | tag `v1.0`, commit `<sha>` |
| Configs | `configs/main_350m.yaml`, `configs/baseline_124m.yaml` | same commit |
| Environment | `pyproject.toml` pins, `labs/module-04/setup_box.sh`, driver <version> | |
| Data | FineWeb-Edu revision `<sha>`, file list in `manifests/` | |
| Tokenizer | `tokenizers/bpe32k/tokenizer.json` (sha256 `<...>`) | |
| SFT / DPO data | revisions `<sha>`, `<sha>` | |
| Checkpoints | HF `<you>/my-gpt-350m-base`, `<you>/my-gpt-350m` tag `v1.0` | commit `<sha>` |
| Evaluation | lm-eval 0.4.13, commands in `reports/eval_report.md` | |

Commands to reproduce the main run, the evaluation and the post-training, in order:

```bash
<copy the exact commands from your run logs>
```

## References

- Penedo et al., *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale*, 2024. https://arxiv.org/abs/2406.17557
- Radford et al., *Language Models are Unsupervised Multitask Learners*, 2019. https://cdn.openai.com/better-language-models/language-models.pdf
- <the papers behind every recipe change you kept, the DPO paper, lm-eval-harness, the datasets>
