---
id: "10.2"
module: 10
minutes: 25
practice_minutes: 120
prerequisites: ["10.1", "08.1"]
objectives:
  - Generate answers once on the GPU and run a paid LLM judge from your laptop, both answer orders per prompt, within a dollar budget you estimated beforehand.
  - Pick a judge model by price and purpose (a cheap judge for selection, a strong one for the reported number) using current API prices.
  - Report a win rate with a confidence interval, position bias and length bias, and say whether the difference is real.
  - Check with lm-eval that SFT and DPO did not damage the base model's skills, and write the results note.
volatility: implementation
sources:
  - title: "Anthropic API pricing (standard and batch, per model)"
    url: https://platform.claude.com/docs/en/about-claude/pricing
  - title: "OpenAI API pricing"
    url: https://developers.openai.com/api/docs/pricing
  - title: "HuggingFaceH4/no_robots dataset card (test split, CC BY-NC 4.0)"
    url: https://huggingface.co/datasets/HuggingFaceH4/no_robots
  - title: "TRL 1.17 DPO Trainer docs"
    url: https://huggingface.co/docs/trl/dpo_trainer
  - title: "SmolLM2-360M-Instruct model card (MT-Bench, IFEval reference numbers)"
    url: https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct
last_verified: "2026-11-13"
---

# 10.2 · Measuring alignment with a judge

DPO's training metrics always improve; whether the model got better is a different question. The practical answer is a pairwise judgment: the same prompts go to the SFT model and to a DPO model, and a strong LLM decides which answer is better, twice per prompt with the order swapped so that its preference for the first or second slot cancels out. You run that comparison the cheap way (answers generated once on the GPU, judging from your laptop for a few dollars), report the result with a confidence interval instead of a bare percentage, and check with lm-eval that alignment did not quietly cost the model the skills it learned in pretraining.

## Goal

You have `judge/final/winrate.json` and `judge/final/notes.md` for the DPO model you ship versus your SFT model (200 prompts, both orders, judged by Claude Opus 5.5), a cheaper selection comparison of both DPO variants, a regression table of base vs SFT vs DPO on HellaSwag, ARC-Easy, ARC-Challenge and PIQA, and `reports/alignment_results.md`: which model goes to Module 11 and why, in numbers.

## What you need

- `hf/my-gpt-350m-sft/`, `hf/my-gpt-350m-dpo/` and `hf/my-gpt-350m-dpo-b01/` in your bucket from [10.1](lesson-01.md), and `hf/my-gpt-350m-base/` from Module 8.
- An LLM API account from 01.2 (Anthropic or OpenAI) with a **monthly spend limit of $10** and about $5–10 of prepaid credit.
- The [Module 10 lab files](../../labs/module-10/) in `tools/`, committed.
- A 1× H100 pod for ~1.2 h (answer generation and lm-eval), then your laptop.
- **Expected cost: ~$7–8**: ~$3.25 GPU (Community $2.69/h) plus ~$4.10 of judge API calls.

> [!CAUTION]
> The API key can spend money until the limit stops it. Create it now, put it only in `~/.config/mygpt/secrets.env` on your laptop (mode 600), and never on the pod template, in the repo, or in a notebook you share. Set the spend limit **before** the first call. If a key leaks, revoke it in the console first, then investigate.

## Choices and costs

### Which judge (prices checked 2026-11-13)

Each prompt costs two judge calls (both orders). For 200 prompts with answers of up to 384 tokens, `judge_tools.py cost` estimates ~270k input and ~145k output tokens. The output includes hidden reasoning (both Claude Opus 5.5 and the small OpenAI models think before answering, and thinking is billed as output); 300 reasoning tokens per call is an assumption, so treat the dollar column as an ESTIMATE.

| Judge | $ per 1M tokens, input / output | 200 prompts × 2 orders | Batch API (−50%) | Use |
|---|---|---|---|---|
| **Claude Opus 5.5** (`claude-opus-5-5`, script default) | $4.00 / $20.00 | **~$4.00** | ~$2.00 | **The reported number** |
| Claude Sonnet 5.5 (`claude-sonnet-5-5`) | $2.00 / $10.00 | ~$2.00 | ~$1.00 | Alternative strong judge |
| **Claude Haiku 5.5** (`claude-haiku-5-5`) | $0.10 / $0.50 (prompts up to 100k tokens) | **~$0.10** | ~$0.05 | **Selection between DPO variants**, second opinion |
| OpenAI `gpt-6.1-sol` | $2.00 / $10.00 | ~$2.00 | ~$1.00 | If your account is OpenAI |
| OpenAI `gpt-6-luna` | $0.10 / $0.50 | ~$0.10 | ~$0.05 | Cheap OpenAI judge |
| OpenAI `gpt-5-nano` | $0.05 / $0.40 | ~$0.07 | ~$0.04 | Cheapest listed |

`judge_winrate.py` calls the standard (non-batch) endpoints, so plan with the standard column. With Anthropic it sends `effort: low` (a verdict needs little reasoning) and the server-side refusal fallback; for Haiku 5.5 a declined request simply stays declined and is counted as `invalid`. `summary.json` splits `invalid` into `invalid_declined` (the judge refused) and `invalid_no_verdict` (a reply without `[[A]]`/`[[B]]`/`[[TIE]]`), and records the judge's real `judge_input_tokens`, `judge_output_tokens` and the `estimated_cost_usd` they imply at the list prices above.

> [!NOTE]
> API prices change more often than GPU prices. Re-check both pricing pages, rerun `judge_tools.py cost` on your real answers, and write the estimate into `budget.md` before the first paid call.

### The plan: select cheaply, report strongly

| Comparison | Judge | Cost | Why |
|---|---|---|---|
| DPO (beta 0.5) vs SFT, and DPO-b01 (beta 0.1) vs SFT | Haiku 5.5 | ~$0.20 | Pick the DPO model to ship |
| Chosen DPO vs SFT | Opus 5.5 | ~$4.00 | The number in your report and model card |

Selecting with one judge and reporting with another limits, but does not remove, the bias of choosing the winner on the same 200 prompts; your notes say so.

### Prompts and sample size

The prompts default to the 500-prompt test split of `HuggingFaceH4/no_robots` (human-written, ten categories from brainstorming to coding; CC BY-NC 4.0, fine for evaluation). With $n$ prompts and win rate $p$, the standard error is about $\sqrt{p(1-p)/n}$: at $n = 200$ and $p \approx 0.5$ that is 0.035, so the 95% interval is roughly ±7 points. A 55% result on 200 prompts is therefore not distinguishable from 50%; telling 55% from 50% reliably would need about $1.96^2 \times 0.25 / 0.05^2 \approx 385$ prompts. Ties (verdicts that flip with the order) shrink the variance somewhat. 200 prompts is enough to detect a clear improvement and cheap enough to run twice.

### Regression check

| Task (lm-eval 0.4.13) | Items | Standard error at ~50% accuracy |
|---|---|---|
| HellaSwag (acc_norm) | 10,042 | ~0.5 points |
| ARC-Easy (acc_norm) | 2,376 | ~1.0 point |
| ARC-Challenge (acc_norm) | 1,172 | ~1.4 points |
| PIQA (acc_norm) | 1,838 | ~1.2 points |

Same settings as Module 8 for all three models (log-likelihood scoring, no chat template), so the only difference is the post-training. `eval_delta.py` flags a drop larger than two combined standard errors. Small SFT models often lose a point or two on these; a drop of five points or more means post-training overwrote base skills.

### Time and money

| Part | Where | Time | Cost (checked 2026-11-13) |
|---|---|---|---|
| Answer generation, 3 models × 200 prompts | H100 | ~0.6 h (ESTIMATE, greedy, batch 1) | $1.60 |
| lm-eval, 3 models × 4 tasks | H100 | ~0.4 h | $1.10 |
| Setup and uploads | H100 | ~0.2 h | $0.55 |
| Haiku 5.5 selection, 2 × 200 prompts | laptop + API | ~15 min | ~$0.20 |
| Opus 5.5 final, 200 prompts | laptop + API | ~20 min | ~$4.00 |
| **Total** | | | **~$7.50** |

## Do this

### 1. Generate all answers on the GPU (45 min)

On the pod (stops armed and environment installed as in 10.1; pull the models you do not have yet):

```bash
for m in base sft dpo dpo-b01; do bash scripts/sync.sh pull-hf my-gpt-350m-$m hf/my-gpt-350m-$m; done
# SFT (A) vs DPO beta 0.5 (B): --dry-run generates both answer sets and calls no API
python scripts/judge_winrate.py --model-a hf/my-gpt-350m-sft --model-b hf/my-gpt-350m-dpo \
  --n 200 --dry-run --out judge/gen_dpo 2>&1 | tee logs/gen_dpo.log
python tools/judge_tools.py answers judge/gen_dpo/judgments.jsonl --out judge/ans_dpo
# reuse the SFT answers; generate only DPO-b01's
python scripts/judge_winrate.py --answers-a judge/ans_dpo/answers_a.jsonl --model-b hf/my-gpt-350m-dpo-b01 \
  --n 200 --dry-run --out judge/gen_b01 2>&1 | tee logs/gen_b01.log
python tools/judge_tools.py answers judge/gen_b01/judgments.jsonl --out judge/ans_b01
```

The dry run prints one complete judge request between `---- example judge request (dry run, nothing sent) ----` lines; read it, because that is exactly what the judge will see. Its coin-flip "summary" is meaningless. `answers` reports how many answers are empty; more than a handful means a generation problem (see below).

### 2. Run the regression evals (25 min)

```bash
for m in base sft dpo dpo-b01; do
  lm_eval --model hf --model_args pretrained=hf/my-gpt-350m-$m,dtype=bfloat16 \
    --tasks hellaswag,arc_easy,arc_challenge,piqa --batch_size 32 --output_path evals/$m
done
python tools/eval_delta.py evals/base evals/sft evals/dpo evals/dpo-b01 \
  --names base,sft,dpo,dpo-b01 --out evals/regression.md
bash scripts/sync.sh push judge ckpt/dpo_350m/judge && bash scripts/sync.sh push evals ckpt/dpo_350m/evals
```

`eval_delta.py` prints a table like this (format from a test run on tiny models; your numbers are real accuracies):

```text
| task | metric | base | sft | dpo | dpo-b01 | Δ sft | Δ dpo | Δ dpo-b01 |
|---|---|---|---|---|---|---|---|---|
| arc_easy | acc_norm | 0.250 ± 0.164 | 0.250 ± 0.164 | 0.250 ± 0.164 | ... | +0.000 | +0.000 | ... |

No change beyond 2 standard errors against the first model.
```

When the push has finished, terminate the pod. Everything else runs on your laptop.

### 3. Estimate the bill from the real answers (laptop, 5 min)

```bash
bash scripts/sync.sh pull ckpt/dpo_350m/judge judge
python tools/judge_tools.py cost judge/gen_dpo/judgments.jsonl
```

The table it prints is for one comparison of 200 prompts. Write the Opus 5.5 line plus twice the Haiku 5.5 line into `budget.md` as the estimate.

### 4. Select the DPO model with the cheap judge (15 min, ~$0.20)

```bash
set -a; . ~/.config/mygpt/secrets.env; set +a          # ANTHROPIC_API_KEY (or OPENAI_API_KEY)
python scripts/judge_winrate.py --answers-a judge/ans_dpo/answers_a.jsonl --answers-b judge/ans_dpo/answers_b.jsonl \
  --n 200 --judge-model claude-haiku-5-5 --out judge/haiku_dpo
python scripts/judge_winrate.py --answers-a judge/ans_b01/answers_a.jsonl --answers-b judge/ans_b01/answers_b.jsonl \
  --n 200 --judge-model claude-haiku-5-5 --out judge/haiku_b01
python tools/judge_tools.py report judge/haiku_dpo/judgments.jsonl --out judge/haiku_dpo
python tools/judge_tools.py report judge/haiku_b01/judgments.jsonl --out judge/haiku_b01
```

With OpenAI instead: `--judge openai --judge-model gpt-6-luna`. Keep `--n 200` and the default `--seed 0` exactly as in step 1, otherwise the prompt list does not match the answer files. Ship the DPO variant with the higher win rate; if both intervals overlap heavily, prefer the one with fewer invalid verdicts and shorter answers (the gentler change).

### 5. Measure the chosen model with the strong judge (20 min, ~$4)

```bash
python scripts/judge_winrate.py --answers-a judge/ans_dpo/answers_a.jsonl --answers-b judge/ans_dpo/answers_b.jsonl \
  --n 200 --out judge/final                     # default judge: claude-opus-5-5
python tools/judge_tools.py report judge/final/judgments.jsonl --out judge/final
cat judge/final/notes.md
```

(Use `judge/ans_b01/` instead if DPO-b01 won step 4.) `judge/final/summary.json` and the two Haiku summaries hold `judge_input_tokens`, `judge_output_tokens` and `estimated_cost_usd`, computed from the token counts the API returned; add the three costs and write the sum into `budget.md`. When the bill appears, check it against the provider console's usage page.

### 6. Interpret the numbers (15 min)

`judge/final/notes.md` reads like this example (from a synthetic test file, not a real model):

```text
- B wins 46, A wins 43, ties 111 (a tie includes order-dependent verdicts).
- B win rate (ties = 0.5): 50.7%, 95% bootstrap CI 46.2%-55.5% -> no detectable difference.
- Among decided prompts: B 51.7% (Wilson 95% CI 41.5%-61.8%), sign test p = 0.832.
- Judge order consistency: 44.5%; picks the first-shown answer 44.8% of the time (50% = no position bias).
- Mean answer length: A 102 words, B 127 words; 72% of B's wins are prompts where B was longer.
```

Read it line by line:

- **Win rate and CI.** If the whole interval is above 50%, DPO is better on these prompts by this judge; if it contains 50%, you cannot claim an improvement, and you say so. Both are legitimate results for a 350M model.
- **Ties.** `judge_winrate.py` only counts a win when both orders agree. Many ties mean the judge's verdict depends on the order, often because both answers are similarly weak.
- **Position bias.** A first-position pick rate far from 50% (below 40% or above 60%) means a single-order evaluation would have been biased; the swap is what saved you. Report the number.
- **Length bias.** If DPO's answers are clearly longer and most of its wins are prompts where it was longer, part of the "improvement" may be the judge liking length (the judge prompt asks it not to, which reduces but does not remove the effect). Read five of those wins in `judgments.jsonl` and decide.
- **Second opinion.** Compare with the Haiku 5.5 result for the same pair. Two judges agreeing on direction is stronger evidence than either alone.

### 7. Write the results note and commit (15 min)

Create `reports/alignment_results.md`:

```markdown
# Alignment results (10.2)

| Comparison (200 no_robots test prompts, both orders) | Judge | B win rate | 95% CI | B/A/ties | first-pos rate |
|---|---|---|---|---|---|
| DPO (beta 0.5) vs SFT | claude-haiku-5-5 | | | | |
| DPO-b01 (beta 0.1) vs SFT | claude-haiku-5-5 | | | | |
| <shipped DPO> vs SFT | claude-opus-5-5 | | | | |

Regression (lm-eval 0.4.13, acc_norm): paste evals/regression.md

Decision: ship <model> because ... Caveats: selection and final judgment use the same prompts;
judge length bias (B answers __ words vs __); 350M model, numbers are not comparable to MT-Bench.
Cost: GPU $__, judge API $__.
```

```bash
cp judge/final/winrate.json reports/winrate.json
git add reports/alignment_results.md reports/winrate.json evals/regression.md tools/ && git commit -m "Alignment results" && git push
```

## How you know it worked

- `judge/final/summary.json` shows `n_judged` close to 200 and `invalid` under ~5; every prompt in `judgments.jsonl` has two verdicts.
- `reports/winrate.json` has a win rate, a bootstrap CI, a Wilson CI, the sign-test p-value, the first-position pick rate and mean answer lengths; your note states plainly whether the CI excludes 50%.
- The Haiku and Opus results for the same pair point in the same direction, or your note explains why not.
- `evals/regression.md` covers four models and four tasks; no SFT/DPO drop exceeds 2 standard errors, or you have explained the one that does.
- The summed `estimated_cost_usd` from the three `summary.json` files is within about 2× of `judge_tools.py cost`, the console's usage page agrees with it, and both lines are in `budget.md`.

## When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| Many empty answers in `answers` output | The model emitted EOT immediately, or the prompt plus 384 new tokens ran past the 1,024-token context | Check a few with `chat.py`; for long prompts, rerun generation with `--max-new-tokens 256` |
| `invalid` is large (the script prints a WARNING) | The judge refused (`invalid_declined`) or did not end with `[[A]]`, `[[B]]` or `[[TIE]]` (`invalid_no_verdict`) | Open `judgments.jsonl`: invalid rows carry `invalid_reason` and the raw `judge_replies`; refusals on unsafe prompts are expected; above ~10% try another judge model |
| `prompts` list shorter than 200 when using answer files | Different `--n` or `--seed` than at generation | Rerun with exactly the same `--n` and `--seed` |
| `AuthenticationError` / 401 | Key not exported in this shell, or revoked | `set -a; . ~/.config/mygpt/secrets.env; set +a`; check `echo ${ANTHROPIC_API_KEY:0:7}` |
| `RateLimitError` / 429 or slow progress | New-account rate limits | The SDK retries automatically; wait, or run the two Haiku comparisons one after the other |
| Bill far above the estimate | Long answers (rambling to 384 tokens) or more reasoning tokens than assumed | Divide `judge_output_tokens` by `judge_calls` in `summary.json`; rerun `cost` with `--thinking-tokens` set to that minus ~60 visible tokens |
| Win rate ~50% with CI containing 50% | DPO changed little, or the judge cannot separate two weak answers | Report it honestly; look at categories in `judgments.jsonl`; a bigger difference needs a better base model, not a different statistic |
| lm-eval drops 5+ points after SFT | Too high a learning rate or too many epochs overwrote base skills | Compare with a 1-epoch SFT model; consider the lower learning rate from the 09.2 sweep |
| `ci95` in `summary.json` is wider than the bootstrap CI from `judge_tools.py report` | `summary.json` uses a Wilson interval with ties as 0.5 (it treats ties like coin flips); the bootstrap resamples the real 0 / 0.5 / 1 scores | Report the bootstrap CI and the Wilson CI over decided prompts from `judge_tools.py report`, as in step 7 |

## Artifact

- `reports/winrate.json` (shipped DPO model vs SFT, Opus 5.5 judge) and `reports/alignment_results.md` (selection, final result with CI and biases, regression table, decision, caveats, cost), committed; judgments and evals under `r2:mygpt/ckpt/dpo_350m/`.
- Your aligned model for Module 11: `r2:mygpt/hf/my-gpt-350m-dpo/` (or `-dpo-b01`), named in the results note.
- `budget.md` lines:

  ```text
  | 2026-11-13 | 10.2 | RunPod 1x H100 SXM Community, answer generation + lm-eval ~1.2 h at $2.69/h | ~$3.25 | $____ |
  | 2026-11-13 | 10.2 | Judge API: Haiku 5.5 selection (2 x 200 prompts) + Opus 5.5 final (200 prompts) | ~$4.20 | $____ |
  ```
