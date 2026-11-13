---
id: "08.3"
module: 8
minutes: 20
practice_minutes: 60
prerequisites: ["08.2"]
objectives:
  - Assemble an evaluation report in which every number traces to a committed file, with tables, curve plots, settings and costs.
  - Turn measured differences into claims whose wording matches their evidence, and run the honest-claims checklist on your own draft.
  - Present compute next to quality, so that a reader can judge your model against GPT-2 for what it cost.
volatility: concept
sources:
  - title: "Mitchell et al., Model Cards for Model Reporting (FAT* 2019)"
    url: https://arxiv.org/abs/1810.03993
  - title: "Biderman et al., Lessons from the Trenches on Reproducible Evaluation of Language Models (lm-eval-harness, arXiv 2405.14782)"
    url: https://arxiv.org/abs/2405.14782
  - title: "Hoffmann et al., Training Compute-Optimal Large Language Models (6ND compute estimate)"
    url: https://arxiv.org/abs/2203.15556
  - title: "Karpathy, build-nanogpt play.ipynb (OpenAI GPT-2 and GPT-3 HellaSwag reference lines)"
    url: https://raw.githubusercontent.com/karpathy/build-nanogpt/master/play.ipynb
last_verified: "2026-11-13"
---

# 08.3 · Writing the evaluation report

Numbers do not speak for themselves; the report decides what they are allowed to say. In this lesson you turn Module 8's tables into `EVAL_REPORT.md`: what was evaluated and how, the results with error bars, the training curves, the cost, and a short list of claims, each worded as strongly as its evidence allows and no stronger. It is the document Module 11 copies into the model card and the technical report, and the one a sceptical reader will check first. It costs nothing but an hour on your laptop.

## Goal

`EVAL_REPORT.md` at the root of your project repo, built from the template: four-model benchmark table with standard errors, paired comparisons, bits per byte on identical text, SVG plots of val bpb and HellaSwag against training tokens for your main run and baseline (with the GPT-2 reference lines), a compute-and-cost table, a numbered list of claims with their evidence, the caveats, and the honest-claims checklist ticked. Every number in it points to a file in `evals/` or `reports/`.

## What you need

- `reports/eval_table.md`, `reports/bpb.md`, `reports/bpb/*.json`, `reports/contamination.md` from [08.2](lesson-02.md); `reports/eval_settings.md` from [08.1](lesson-01.md).
- The `log.jsonl` of the main run and the baseline (in the bucket under `ckpt/main_350m/` and `ckpt/baseline_124m/`), `reports/main-run-log.md` from Module 7, `budget.md`.
- `tools/plot_curves.py` and `tools/EVAL_REPORT.template.md` from the [Module 8 lab files](../../labs/module-08/).
- About 1.5 hours on your laptop. **Expected cost: $0.**

## Choices and costs

### What goes in, what stays out

| Content | In the report? | Why |
|---|---|---|
| Your four-model harness table, with metric names and ± stderr | Yes, the core | Measured by you, identical settings |
| Paired intervals for the three comparisons | Yes | The basis for every "better" |
| Bits per byte on the common val set, with the home-field note | Yes | Fair between your models; context for OpenAI's |
| In-training HellaSwag curve with Karpathy's GPT-2 lines (0.2945, 0.3752) | Yes, as a plot | Same scoring rule as your in-training eval, so the lines belong on that plot |
| Published numbers from other setups (GPT-3 paper 0-shot table, Open LLM Leaderboard few-shot) | Only in a separate, labelled "context" sentence | Different harness, shots or scoring; never in your table |
| `--limit` runs, CPU rehearsals | No | Not full benchmarks |
| Cherry-picked best checkpoint by benchmark | No | Report the final checkpoint you froze; choosing by test score is tuning on the test set |

### Claim strength must match the evidence

| Evidence | Allowed wording | Not allowed |
|---|---|---|
| Paired 95% interval excludes 0 | "scores x.x points higher than GPT-2 medium on HellaSwag (95% CI [lo, hi])" | "dramatically better", "outperforms GPT-2 medium" (without saying on what) |
| Interval includes 0 | "no measurable difference on ARC-Challenge" | "slightly better", "on par or better" |
| Better on 4 of 6 tasks, others within noise | "better on HellaSwag, ARC-Easy, PIQA and LAMBADA; no measurable difference on WinoGrande and ARC-Challenge" | "better across the board" |
| Lower bpb on FineWeb-Edu val | "predicts held-out FineWeb-Edu text better" | "a better language model than GPT-2" |
| One training run | "this run" | "this recipe reliably gives" |

### Compute next to quality

$$C \approx 6ND \quad\Rightarrow\quad C_{\text{main}} \approx 6 \times 3.37\times10^{8} \times 2\times10^{10} = 4.0\times10^{19}\ \text{FLOP}$$

| Model | Params | Tokens | Training compute | Cost |
|---|---|---|---|---|
| Your 350M | 337M | 20B | ~4.0e19 FLOP (6ND) | $ from `budget.md` |
| Your baseline | 124M | 10B | ~7.5e18 FLOP | $ from `budget.md` |
| OpenAI GPT-2 medium | 355M | not published | not published | not published |
| OpenAI GPT-2 124M | 124M | not published | not published | not published |

Leave the unknowns as "not published"; do not fill them with guesses from blog posts. The point of the table is that a reader sees your result together with exactly what it took.

## Do this

### 1. Get the logs and make the plots (10 min)

```bash
mkdir -p runs/main_350m runs/baseline_124m
rclone copy r2:mygpt/ckpt/main_350m/log.jsonl runs/main_350m/
rclone copy r2:mygpt/ckpt/baseline_124m/log.jsonl runs/baseline_124m/
python tools/plot_curves.py runs/main_350m runs/baseline_124m --labels "my 350M (own BPE),baseline 124M (GPT-2 tokenizer)" \
  --out-dir reports/figures --smooth 5 \
  --hline "hellaswag:OpenAI GPT-2 124M=0.2945" --hline "hellaswag:OpenAI GPT-2 medium=0.3752"
```

Expect `wrote val_bpb.svg, hellaswag.svg, train_loss.svg, tokens_per_s.svg, lr.svg`. Open `reports/figures/val_bpb.svg` in a browser: your 350M curve should run below the baseline's at every token count and show the WSD drop over the last 20%; `lr.svg` shows warmup, the flat phase and the decay (and any change you made in 07.2). The train-loss plot compares runs with different tokenizers, so it is for your eyes only: put val bpb in the report.

### 2. Start the report from the template (10 min)

```bash
cp tools/EVAL_REPORT.template.md EVAL_REPORT.md
```

Fill sections 1 and 2 from `reports/eval_settings.md`, the `model_sha` values in `evals/gpt2*/*/results_*.json`, `hf/my-gpt-350m-base/mygpt_export.json` (step, tokens, params) and the common val counts in `TOKEN_REPORT.md`. The image links in the template are relative to the repo root (`reports/figures/...`); fix them if you keep the report elsewhere.

### 3. Paste the measured results (15 min)

Section 3: paste `reports/eval_table.md` (table and comparison blocks) and the four rows of `reports/bpb.md`. Add the compute-and-cost table above with your real dollars: main run from 07.3's reconciled line, evaluation from the 08.1 and 08.2 lines.

### 4. Write the claims (20 min)

At most five, numbered, each one sentence plus its evidence, worded by the table above. A typical honest set:

1. On HellaSwag (0-shot, acc_norm), my 350M model scores **x.x** vs GPT-2 medium's **y.y**, a difference of +d.d points (paired bootstrap 95% CI [lo, hi]).
2. It is better than GPT-2 medium on ARC-Easy, PIQA and LAMBADA (intervals in table 3) and shows no measurable difference on WinoGrande and ARC-Challenge, where all four models are near chance.
3. My GPT-2 124M reproduction matches OpenAI's GPT-2 124M within noise on k of 6 tasks (baseline vs gpt2 block).
4. On held-out FineWeb-Edu text my 350M model reaches 0.xx bits per byte vs 0.yy for my baseline; OpenAI's models score 0.zz and 0.ww on the same text, which is outside their training distribution.
5. Training cost: $N for ~4.0e19 FLOP on one 8×H100 node in h hours.

If a claim you hoped for is not supported, write the supported version. "Matches GPT-2 medium within noise at a training cost of $N" is a result.

### 5. Caveats and checklist (10 min)

Fill section 5 from `reports/contamination.md` and the run log (any rollback, skipped batches or shortened training). Then go through section 6 line by line against your files and tick only what is true. A box you cannot tick is either fixed now (rerun, remove the number) or written down as a caveat.

### 6. Commit and back up (5 min)

```bash
git add EVAL_REPORT.md reports/figures/ && git commit -m "Evaluation report" && git push
bash scripts/sync.sh push EVAL_REPORT.md ckpt/main_350m/EVAL_REPORT.md
```

## How you know it worked

- `EVAL_REPORT.md` renders on GitHub with both plots, the four-model table and the comparison blocks.
- Picking any number in it at random, you can name the file it came from in under a minute.
- Every "better" in the claims has an interval that excludes 0 next to it; every other comparison says "no measurable difference".
- The bpb row carries the home-field note, the compute table has "not published" where OpenAI did not publish, and no `--limit` or few-shot number appears in your table.
- All eight checklist boxes are ticked, or the unticked one has a caveat explaining why.

## When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `plot_curves.py`: the 350M val bpb curve has a gap or a jump | Rolled-back steps or a resume; the last record per step wins, so a rollback shows as a step change | Expected if 07.2 had a rollback; mention it in the caption |
| HellaSwag reference lines sit far above or below your curves | Mixing scoring rules: the lines are Karpathy's per-token scoring, which matches the in-training eval, not lm-eval acc_norm | Keep the lines on the in-training plot only; use the lm-eval table for the cross-model claim |
| A claim feels too weak to write | The evidence is weak | Write it anyway, or drop it; never upgrade the wording |
| Report numbers differ from `reports/eval_table.md` | Copied by hand and mistyped | Paste the generated table instead of retyping |
| Plots do not show on GitHub | Image path wrong relative to the report | Links relative to the report's folder, e.g. `reports/figures/val_bpb.svg` from the repo root |

## Artifact

- In git: `EVAL_REPORT.md` and `reports/figures/*.svg`, next to Module 8's `reports/` and `evals/` files. This is the module's artifact; Module 11 builds the model card and technical report from it.
- In the bucket: `ckpt/main_350m/EVAL_REPORT.md`.
- `budget.md`: no new spending; add a line with the total evaluation cost of Module 8 for the final ledger:

  ```text
  | 2026-11-18 | 08.3 | Evaluation total (08.1 + 08.2), report written on laptop | ~$5 | $____ |
  ```
