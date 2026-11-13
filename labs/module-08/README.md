# Module 8 lab files — evaluate honestly

Helpers for lessons 08.1–08.3. Copy them into the `tools/` folder of your project repo and commit
them. The code they work with (`mygpt/evaluate.py`, `mygpt/export_hf.py`, `scripts/decontam.py`,
lm-eval-harness from the `[eval]` extra) is in `labs/common/`.

| File | Used in | What it does |
|---|---|---|
| `eval_table.py` | 08.1–08.3 | Reads lm-eval `--output_path` folders (newest `results_*.json` in each) and prints one markdown table: per task the chosen metric (acc_norm for HellaSwag, ARC, PIQA; acc for WinoGrande and LAMBADA, plus LAMBADA perplexity) in points ± standard error, and an average over tasks all models have. `--compare A:B` adds A − B per task with a 95% interval: a paired bootstrap over the same items when both runs used `--log_samples`, else an unpaired interval from the standard errors. Warns when runs differ in shots, `--limit`, lm-eval or task versions. |
| `hf_bpb.py` | 08.2 | Bits per byte of any Hugging Face causal LM (OpenAI GPT-2, your exports) on mygpt val shards, with exactly the windows `mygpt.evaluate` uses; `--eval-tokens all` covers the whole split and prints the `--eval-tokens` value for `mygpt.evaluate` that covers the same windows. Refuses to run if the shards were made with another tokenizer. |
| `bench_texts.py` | 08.2 | Writes the WinoGrande (validation, 1,267) and LAMBADA-OpenAI (test, 5,153) items as a `--benchmark-file` for `scripts/decontam.py`, for the two tasks Module 3 did not check. |
| `plot_curves.py` | 08.3 | SVG plots of val bpb, in-training HellaSwag, train loss, tokens/s and LR against training tokens for several runs' `log.jsonl`, with optional dashed reference lines. Stdlib only. |
| `EVAL_REPORT.template.md` | 08.3 | The evaluation report skeleton with the honest-claims checklist. |

Examples:

```bash
python tools/eval_table.py evals/my-gpt-350m-base evals/baseline-124m evals/gpt2 evals/gpt2-medium \
  --names mine-350m,baseline-124m,gpt2,gpt2-medium --compare mine-350m:gpt2-medium --out reports/eval_table.md
python tools/hf_bpb.py --model openai-community/gpt2 --data-dir data/tok/common-val-gpt2 --eval-tokens all --json reports/bpb-gpt2.json
python tools/bench_texts.py --out reports/bench_wg_lambada.jsonl
python tools/plot_curves.py runs/main_350m runs/baseline_124m --labels "my 350M,baseline 124M" --out-dir reports/figures
```

Requirements: `eval_table.py` needs Python 3.10+ and numpy (for the bootstrap); `plot_curves.py`
only the standard library; `hf_bpb.py` and `bench_texts.py` the project venv
(`pip install -e ".[post,eval]"`).

Tested on 2026-11-13 on Windows (CPU, lm-eval 0.4.13, transformers 5.22.2): `hf_bpb.py` on a
smoke-run export matched `mygpt.evaluate` on the same windows exactly (val_loss 4.27541 both),
and ran on `openai-community/gpt2` with GPT-2 shards (tokenizer check passed); `eval_table.py` on
lm-eval result folders of `openai-community/gpt2` with and without `--log_samples` (paired and
unpaired comparisons, settings warnings); `bench_texts.py` + `decontam.py --benchmark-file` on a
small GPT-2 shard set; `plot_curves.py` on a smoke-run log and a synthetic 6,000-step log.
