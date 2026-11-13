---
id: "08.1"
module: 8
minutes: 25
practice_minutes: 75
prerequisites: ["07.3", "05.3"]
objectives:
  - Choose the benchmark tasks, shots and metrics (acc vs acc_norm) that measure something real for a ~350M base model, and fix them as one harness configuration.
  - Run lm-eval-harness 0.4.13 on your main model and your exported baseline on a cheap GPU box, with per-item samples saved.
  - Explain why the same model gets three different HellaSwag numbers (in-training eval, lm-eval acc, lm-eval acc_norm) and why --limit results cannot be reported.
volatility: implementation
sources:
  - title: "lm-evaluation-harness on PyPI (0.4.13, 2026-08-31; HF backend in the hf extra; run/ls/validate CLI)"
    url: https://pypi.org/project/lm-eval/
  - title: "lm-evaluation-harness quickstart (GPT-2 HellaSwag acc 0.2891, acc_norm 0.3108)"
    url: https://lm-evaluation-harness.readthedocs.io/getting_started/quickstart/
  - title: "EleutherAI lm-evaluation-harness repository"
    url: https://github.com/EleutherAI/lm-evaluation-harness
  - title: "Zellers et al., HellaSwag: Can a Machine Really Finish Your Sentence?"
    url: https://arxiv.org/abs/1905.07830
  - title: "openai-community/gpt2 model card (Open LLM Leaderboard few-shot results)"
    url: https://huggingface.co/openai-community/gpt2
  - title: "RunPod GPU pricing"
    url: https://www.runpod.io/pricing
  - title: "Lambda instance pricing (1x A10 $1.29, 1x H100 SXM $4.89 from 2026-11-16)"
    url: https://lambda.com/service/gpu-cloud/pricing
  - title: "Hyperbolic GPU marketplace (H100 SXM from $3.19/GPU-h)"
    url: https://www.hyperbolic.ai/marketplace
  - title: "Vast.ai pricing 2026 (H100 SXM from $1.73, median $2.40; third-party snapshot)"
    url: https://www.spheron.network/blog/vastai-pricing-2026/
  - title: "Free cloud GPUs in 2026 (Kaggle ~30 GPU-h/week, T4/P100; third-party)"
    url: https://gpuperhour.com/blog/free-cloud-gpus-and-credits
last_verified: "2026-11-13"
---

# 08.1 · Setting up lm-eval-harness

Your training log already holds a HellaSwag number, but it was measured by your own code with your own scoring rule, and nobody else can reproduce it. Reportable numbers come from a standard harness that anyone can rerun on your published model: EleutherAI's lm-eval-harness, the tool behind most small-model papers and leaderboards. Using it well is mostly about decisions made before the first run: which tasks a 350M model can do above chance, how many examples to show it, which of the two accuracy columns to read, and which settings must be identical for every model you will compare. This lesson fixes those decisions, tries them on your laptop for free, and then runs them on your two models on a cheap rented GPU.

## Goal

`evals/my-gpt-350m-base/` and `evals/baseline-124m/` hold complete lm-eval 0.4.13 results (results JSON plus per-item samples) for six tasks, 0-shot, bf16, no `--limit`, for your main model and your exported GPT-2 124M baseline. `reports/eval_settings.md` states the configuration every later run must use, and `tools/eval_table.py` prints both models in one table. The box stays up for 08.2.

## What you need

- `r2:mygpt/hf/my-gpt-350m-base/` from [07.3](../module-07/lesson-03.md) and `r2:mygpt/ckpt/baseline_124m/` (final checkpoint and log) from [05.3](../module-05/lesson-03.md).
- The `mygpt-gpu` RunPod template from [04.2](../module-04/lesson-02.md) and RunPod credit (or credit at one of the alternatives below).
- The [Module 8 lab files](../../labs/module-08/) copied into `tools/` and committed.
- On your laptop: the project venv with `pip install -e ".[eval]"` (the extra installs `lm-eval[hf]`, so lm-eval brings its Hugging Face backend: transformers, accelerate, peft).
- About 1.5 hours, ~45 minutes of it on the GPU box.
- **Expected cost: ~$2** (about 0.7 h of a 1×H100 SXM Community pod at $2.69/h, checked 2026-11-13). 08.2 continues on the same pod for ~$3 more.

## Choices and costs

### Which tasks a 350M model can do

| Task (lm-eval name) | Items scored | Chance | GPT-2 124M | Keep? |
|---|---|---|---|---|
| `hellaswag` | 10,042 (validation) | 25% | 31.1% acc_norm | **Yes**: the course's headline task, clearly above chance at this size |
| `arc_easy` | 2,376 (test) | 25% | 39.5% acc_norm | **Yes**: grade-school science; rises clearly with training |
| `arc_challenge` | 1,172 (test) | 25% | 22.7% acc_norm | Yes, but expect noise around chance; report it, claim nothing from it |
| `piqa` | 1,838 (validation) | 50% | 62.5% acc_norm | **Yes**: physical commonsense, works from ~100M parameters |
| `winogrande` | 1,267 (validation) | 50% | 51.6% acc | Yes, as a check: near chance for all models of this size |
| `lambada_openai` | 5,153 (test) | ~0% | 32.6% acc, perplexity 40.1 | **Yes**: last-word prediction; measures long-range context use |
| `mmlu`, `gsm8k` | | 25%, 0% | at chance, ~1% | **No**: at chance below ~1B parameters; a number that says nothing |

GPT-2 124M column: HellaSwag from the lm-eval quickstart (PUBLICLY DOCUMENTED); the other five measured for this course on 2026-11-13 with lm-eval 0.4.13 on `openai-community/gpt2`, fp32 on a laptop CPU, full test sets, 0-shot (41 minutes for those five tasks). You measure all of them yourself in bf16 in 08.2. Items: the lm-eval 0.4.13 splits.

### 0-shot or few-shot

| Setting | What it costs | Effect at 350M | Verdict |
|---|---|---|---|
| **0-shot** | 1× | The standard for small base models (GPT-2/3 papers' 0-shot tables, Pythia, SmolLM) | **Use it for everything** |
| 5–25-shot (Open LLM Leaderboard v1 style) | 5–25× longer prompts, ~5–20× the GPU time | Small models gain little and the prompt often exceeds 1,024 tokens of context | Do not mix: the gpt2 model card's leaderboard numbers (HellaSwag 10-shot 31.53, ARC 25-shot 22.01, WinoGrande 5-shot 50.43) are not comparable with your 0-shot table |

### acc or acc_norm

For multiple-choice tasks lm-eval scores each answer by its total log-likelihood given the question. `acc` picks the answer with the highest total, which favours short answers (fewer tokens, fewer chances to lose probability). `acc_norm` divides by the answer's length **in characters** first. That makes it tokenizer-independent, which matters here: your model and GPT-2 split the same answer into different numbers of tokens.

| Model | Scoring | HellaSwag |
|---|---|---|
| OpenAI GPT-2 124M | lm-eval `acc` | 0.2891 |
| OpenAI GPT-2 124M | lm-eval `acc_norm` (per character) | 0.3108 |
| OpenAI GPT-2 124M | build-nanogpt / `mygpt.evaluate` (mean loss per token, Karpathy's number) | 0.2945 |

Same weights, three numbers. Rule for this course: **acc_norm** for HellaSwag, ARC-Easy, ARC-Challenge and PIQA; **acc** for WinoGrande (its two options differ only in the filled-in word, scored on the rest of the sentence) and LAMBADA (one target word); state the metric next to every number. Your in-training HellaSwag stays a training signal; the lm-eval number is what you report.

### Settings that must be identical for every model

`lm_eval` 0.4.13, the same six tasks, `--num_fewshot 0`, `dtype=bfloat16`, `--batch_size 32`, no `--limit`, no chat template, `--log_samples` (per-item results for paired statistics in 08.2), `--seed 1234`. Module 10 reuses exactly these settings for SFT and DPO models.

> [!WARNING]
> `--limit N` evaluates only the first N items of each task. Use it to test the pipeline, never to report. In this course's test run, `--limit 100` gave OpenAI GPT-2 a HellaSwag acc_norm of 0.44 ± 0.05; on all 10,042 items it is 0.31.

### Where to run it (checked 2026-11-13)

| Box | Price | Fits | Verdict |
|---|---|---|---|
| Your laptop CPU | $0 | Pipeline tests with `--limit` (minutes) | The full suite takes many hours per model on a CPU: test only |
| RunPod Community RTX 4090 (24 GB) | ~$0.34/h | lm-eval for all four models; 08.2's `mygpt.evaluate` with `--batch 16` | Cheap, but slower, and you must remember `--batch 16`: by default `mygpt.evaluate` uses the training micro-batch (64 × 1,024 with a 50k vocabulary for the baseline), which needs more than 24 GB |
| **RunPod Community 1×H100 SXM (80 GB)** | $2.69/h | Everything in 08.1 and 08.2 | **Recommended**: ~1.5 h for both lessons, ~$4–5 |
| Vast.ai 1×H100 SXM (verified host) | from ~$1.73/h, median ~$2.40/h (marketplace) | Everything in 08.1 and 08.2 | Cheapest H100; check the host's reliability and CUDA 13 driver; stop with `vastai stop instance <ID>`, finish with `destroy` |
| Lambda 1×A10 (24 GB) / 1×H100 SXM | $1.29/h / $4.89/h (prices from 2026-11-16) | A10: like the RTX 4090 (`--batch 16`); H100: everything | Reliable; terminate in the dashboard, a `shutdown` does not stop billing |
| Hyperbolic 1×H100 SXM | from $3.19/h | Everything | Marketplace; availability per supplier |
| Kaggle / Google Colab free notebook | $0 (Kaggle: ~30 GPU-h/week on T4 or P100, 16 GB) | Pipeline tests only | No native bf16 on T4/P100, so you cannot use the identical settings below; never report numbers from it |
| 8×H100 | $35/h | | Never for evaluation |

> [!NOTE]
> Prices from the RunPod, Lambda and Hyperbolic pricing pages on 2026-11-13; Vast.ai figures are marketplace snapshots from third-party trackers (the live grid did not load); the Kaggle quota is the commonly reported figure, check yours in Kaggle's settings. lm-eval version from PyPI on the same day (0.4.13, released 2026-08-31; 0.5.0 exists only as a dev pre-release). Re-check both, and keep the version pinned in `pyproject.toml` for the whole module.

## Do this

### 1. Write down the configuration (5 min, laptop)

Create `reports/eval_settings.md`:

```markdown
# Evaluation settings (fixed for Modules 8 and 10)
lm-eval 0.4.13, --model hf, model_args dtype=bfloat16, --num_fewshot 0, --batch_size 32, --seed 1234,
--log_samples, no --limit, no chat template.
Tasks: hellaswag, arc_easy, arc_challenge, piqa, winogrande, lambada_openai.
Metrics: acc_norm for hellaswag, arc_easy, arc_challenge, piqa; acc for winogrande, lambada_openai (+ perplexity).
```

### 2. Rehearse on your laptop (10 min, $0)

```bash
pip show lm-eval | head -2          # Version: 0.4.13
lm_eval --model hf --model_args pretrained=openai-community/gpt2,dtype=float32 --device cpu \
  --tasks hellaswag,arc_easy --batch_size 16 --limit 100 --output_path evals/_rehearsal
```

It downloads GPT-2 and the two datasets, prints a table with `acc`, `acc_norm` and `Stderr` columns, and writes `evals/_rehearsal/openai-community__gpt2/results_<timestamp>.json`. The numbers are meaningless (100 items) and the stderr column says so: ±0.05. On Windows, set `PYTHONUTF8=1` first, or printing the table fails with a `UnicodeEncodeError` (the results file is written either way). The newer subcommand form `lm-eval run --model hf ...` is equivalent.

### 3. Rent the box and pull the models (15 min)

RunPod → **Community Cloud** → **H100 SXM**, 1 GPU, template `mygpt-gpu` → Deploy. Then:

```bash
nohup bash -c 'sleep 4h; runpodctl pod stop $RUNPOD_POD_ID' >/tmp/hardstop.log 2>&1 &
git clone https://github.com/<you>/<your-repo>.git ~/mygpt && bash ~/mygpt/tools/setup_box.sh
cd ~/mygpt && . .venv/bin/activate && pip install -e ".[eval]"
tmux new -d -s guard "bash scripts/autoshutdown.sh --idle-minutes 30 --stop-cmd 'runpodctl pod stop $RUNPOD_POD_ID'"
bash scripts/sync.sh pull-hf my-gpt-350m-base hf/my-gpt-350m-base
bash scripts/sync.sh pull-ckpt baseline_124m runs/baseline_124m          # final checkpoint + log, ~1.5 GB
```

The idle guard uses 30 minutes here because the GPU is idle while datasets download and while you read results.

On Vast.ai, Lambda or Hyperbolic: copy `~/.config/mygpt/r2.env` to the box, run `set -a; . ~/r2.env; set +a` before `setup_box.sh`, and replace both `runpodctl pod stop $RUNPOD_POD_ID` commands with that provider's stop command (07.1 step 4 lists them). On a 24 GB GPU add `--batch 16` to every `mygpt.evaluate` call in 08.2.

### 4. Export the baseline the same way as the main model (3 min)

```bash
python -m mygpt.export_hf --ckpt runs/baseline_124m --out hf/baseline-124m --tokenizer gpt2
bash scripts/sync.sh push-hf baseline-124m hf/baseline-124m
```

Expect `"architecture": "GPT2LMHeadModel"`, a logit difference below 1e-3 and `"step": 19073`. Both your models now go through the identical Hugging Face path that OpenAI's GPT-2 will in 08.2.

### 5. Run the suite on both models (25 min)

```bash
mkdir -p evals logs
TASKS=hellaswag,arc_easy,arc_challenge,piqa,winogrande,lambada_openai
for m in my-gpt-350m-base baseline-124m; do
  lm_eval --model hf --model_args pretrained=hf/$m,dtype=bfloat16 --tasks $TASKS --num_fewshot 0 \
    --batch_size 32 --seed 1234 --log_samples --output_path evals/$m 2>&1 | tee logs/lmeval-$m.log
done
```

Each run ends with a table and `total_evaluation_time_seconds` in the results JSON; note both times for the cost line (ESTIMATE: ~5–10 minutes per model on an H100).

### 6. Read the results (10 min)

```bash
python tools/eval_table.py evals/my-gpt-350m-base evals/baseline-124m --names mine-350m,baseline-124m \
  --compare mine-350m:baseline-124m --out reports/eval_table_own.md
python tools/run_summary.py runs/baseline_124m | grep hellaswag       # the in-training number, for comparison
```

Check: no `WARNING` lines (identical settings), item counts 10,042 / 2,376 / 1,172 / 1,838 / 1,267 / 5,153, every accuracy at or above chance, and your 350M clearly above your 124M on HellaSwag, ARC-Easy, PIQA and LAMBADA. Keep the pod running and go on to [08.2](lesson-02.md); if you stop here, copy `evals/` off first (step 7) and terminate.

### 7. Save the results (5 min)

```bash
bash scripts/sync.sh push evals ckpt/main_350m/evals
git add reports/eval_settings.md reports/eval_table_own.md logs/lmeval-*.log evals/*/*/results_*.json
git commit -m "lm-eval: main model and baseline" && git push
```

Only the small `results_*.json` files go into git; the `samples_*.jsonl` files (tens of MB) stay in the bucket.

## How you know it worked

- `reports/eval_table_own.md` lists six tasks (plus LAMBADA perplexity) for both models with the item counts above, and no settings warnings.
- The baseline lands near OpenAI's GPT-2 124M: HellaSwag acc_norm around 0.30–0.32, PIQA around 0.62, LAMBADA acc around 0.3 (ESTIMATES; your baseline saw different data than OpenAI's).
- Your 350M model is clearly better than your baseline on HellaSwag (several points, interval excluding 0).
- The baseline's lm-eval HellaSwag acc_norm differs from its in-training number by about one to two points, because of the scoring rule and lm-eval's text preprocessing, not because something broke.

## When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: transformers` / `accelerate` from lm-eval | Since its December 2025 packaging change, lm-eval installs the HF backend only with its `[hf]` extra; your checkout predates the `[eval]` extra that asks for it, or lm-eval was installed by hand | Pull the course code and run `pip install -e ".[eval]"` again (the extra is `lm-eval[hf]>=0.4.13,<0.5`) |
| `UnicodeEncodeError` at the end on Windows | The table contains arrows the console code page cannot print | `set PYTHONUTF8=1` (PowerShell: `$env:PYTHONUTF8=1`); the results JSON was already written |
| Your model scores exactly chance on every task | Wrong tokenizer in the export, or the model loaded with random weights | `python scripts/chat.py --model hf/my-gpt-350m-base --prompt "The capital of France is"`; check `mygpt_export.json` |
| `Token indices sequence length is longer than the specified maximum` | A few LAMBADA or HellaSwag items exceed 1,024 tokens | lm-eval truncates from the left; harmless for these tasks |
| CUDA out of memory | Batch 32 with long items on a smaller GPU | `--batch_size 16` for all models (it does not change scores); write it into `eval_settings.md` |
| Numbers differ slightly between two runs of the same model | bf16 kernels or batch composition | Differences of ~0.1 point are normal; anything larger means different settings: compare the `config` block of both results files |
| Dataset download fails (`429` or timeout) | Unauthenticated Hub requests are rate-limited | `export HF_TOKEN=...` (read token is enough) and rerun |

## Artifact

- In git: `reports/eval_settings.md`, `reports/eval_table_own.md`, `evals/*/*/results_*.json`, `logs/lmeval-*.log`.
- In the bucket: `hf/baseline-124m/` and `ckpt/main_350m/evals/` (results plus samples).
- `budget.md`:

  ```text
  | 2026-11-18 | 08.1 | RunPod 1x H100 SXM Community, ~0.7 h at $2.69/h, lm-eval on main model + baseline | ~$2 | $____ |
  ```
