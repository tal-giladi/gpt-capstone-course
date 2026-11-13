# Lesson spec (binding for every agent writing this course)

Read first, in this order:
1. `C:\Users\TalGiladi\OneDrive\repos\tals-academy\docs\new-course-instructions.md` (Academy rules — win over everything).
2. `curriculum/course-plan.md` (what the course is, budget, decisions).
3. This file.
4. `curriculum/code-contract.md` (the lab code every lesson drives).

## What kind of course this is

A **guided real-world capstone**, not a theory course. The learner already finished an
LLM Research Engineer course (they know transformers, BPE, AdamW, DDP, scaling laws, SFT, DPO,
evals). Do NOT re-teach theory. Each lesson is a step the learner performs on real
infrastructure with real money: which company/website, what it costs today, exact commands,
how to know it worked, what goes wrong. Short "why" paragraphs are fine when they drive a
decision.

Running project: train **their own ~350M-parameter GPT-2-class model** on ~20B tokens of
FineWeb-Edu with **their own BPE tokenizer**, after a GPT-2 124M baseline (GPT-2 tokenizer,
10B tokens) and a set of ablations that choose the improvements. Then SFT, DPO, eval, publish.
Hard budget cap **$500** total (expected $350–450). Every lesson that spends money states the
expected cost and adds a line to the learner's `budget.md` ledger.

## Current facts and prices

- Today is 2026-11-13. Before writing a lesson, **web-search current primary sources**
  (provider pricing pages, Hugging Face dataset cards, official docs, release notes). Use the
  WebSearch tool (mode "standard"; "extended" for prices; WebFetch if you have it). Cite only
  URLs you actually saw.
- Put every price with its date: "RunPod H100 SXM: ~$2.99/GPU-h (checked 2026-11-13)".
- Prices drift: every price table ends with a `> [!NOTE]` telling the learner to re-check the
  live page and update `budget.md`.
- Plan-level reference prices (checked 2026-11-13): H100 SXM on-demand RunPod ~$2.99–3.99,
  Lambda $3.99/GPU-h on 8× nodes ($4.29 single), Hyperbolic $3.19, Nebius $4.50, Vast.ai
  marketplace ~$1.3–3.6 (variable). Budget planning rate: **$4.00/GPU-h**. Cloudflare R2
  $0.015/GB-mo, zero egress; Backblaze B2 $0.006/GB-mo, free egress up to 3× stored.
  FineWeb-Edu: ODC-BY license; `sample/10BT` ≈ 28.5 GB parquet; `sample/100BT` exists.
- Distinguish: PUBLICLY DOCUMENTED (cite) vs REASONABLE PRACTICE vs ESTIMATE (show the math).

## Lesson file format (Academy rules, summarized — the instructions file is authoritative)

Front-matter (YAML): `id: "MM.L"` (module two digits, lesson number in module, e.g. "04.2"),
`module`, `minutes`, `practice_minutes`, `prerequisites` (earlier lesson ids), `objectives`
(2–4 measurable), `volatility: implementation` (almost all lessons here), `sources` (title +
url, real URLs you saw in search results), `last_verified: "2026-11-13"`.

Then exactly one H1: `# MM.L · Title`, then one plain intro paragraph (no list/heading/bold).

### Fixed `##` sections, in this order, in EVERY lesson

1. `## Goal` — what you will have at the end (the artifact), 2–4 sentences.
2. `## What you need` — accounts, prior artifacts, time, **expected cost** for this lesson.
3. `## Choices and costs` — the real options (companies/sites/tools), a table with live prices
   and trade-offs, and the recommended choice for this project. If there is genuinely no
   choice, say what is used and what it costs (often $0).
4. `## Do this` — numbered steps, commands in fenced code blocks (```bash / ```python /
   ```yaml). Use the lab code from `labs/` (see code contract). Never call progress tools.
5. `## How you know it worked` — concrete checks (expected output, numbers, file sizes,
   loss ranges, screenshots to look for).
6. `## When it goes wrong` — common failures and recovery (symptom → cause → fix).
7. `## Artifact` — exactly what to save/commit/upload, where, and the `budget.md` line.

Use `###` inside sections. No other `##` sections. No "Quiz"/"Knowledge check" section.
Allowed: GFM tables, task lists, fenced code, KaTeX `$…$`, mermaid, GitHub alerts
(`> [!NOTE]`, `> [!TIP]`, `> [!IMPORTANT]`, `> [!WARNING]`, `> [!CAUTION]`),
`<details><summary>` only. No other HTML. Anything that spends money, deletes data, or exposes
secrets gets `> [!WARNING]` or `> [!CAUTION]` (e.g. "a forgotten 8×H100 node costs ~$30/hour").

Links: relative only. Lesson → lesson: `../module-04/lesson-02.md`. Lab files:
`../../labs/common/...` or `../../labs/module-NN/`. External sites: full https URLs.
No docsify links, no links to the course's own GitHub.

## Quizzes

- `lessons/module-NN/lesson-MM.quiz.yaml`: 4 questions (3–5 allowed). Module quiz
  `assessments/module-NN-quiz.quiz.yaml`: 9 questions (8–10) + `assessments/module-NN-quiz.md`
  short intro page (what it covers, pass mark 70%, links to the lessons; no questions/answers).
- Exactly 4 options, one correct, `correct` 0–3 spread evenly with no pattern, similar-length
  options (correct one NOT the longest), no all/none of the above, `explanation` 1–3 sentences,
  ids q1..qN, `>-` block scalars for text with `:` `#` quotes or backslashes.
- Questions are realistic decisions: "Your 8×H100 node shows 40% lower tokens/s than your
  1-GPU benchmark predicted. What do you check first?" — not recall of wording.
- Parse every YAML after writing: `py -c "import yaml,sys;[yaml.safe_load(open(f,encoding='utf-8')) for f in sys.argv[1:]]" <files>`.

## Process (Academy section 9)

- Finish one lesson fully (md + quiz.yaml) before the next. After each lesson append
  `lesson-MM done` to `curriculum/status/module-NN.log`; after the module quiz, `module done`.
- Do not edit shared files (`_sidebar.md`, `glossary.md`, `README.md`, `labs/common/`,
  `BUILD_PROGRESS.md`). Glossary terms go to `curriculum/glossary-inbox/module-NN.md`
  (`- **Term** — definition`). Code changes you need in `labs/common/` go to
  `curriculum/code-requests/module-NN.md`. Module-specific helper files go in `labs/module-NN/`
  (always with a `README.md`).
- English, UTF-8, LF line endings. Do not commit; the main session commits.
- Write files with LF: after writing, the main session normalizes, so don't worry about CRLF warnings.

## Module/lesson map (titles are final; H1 must match)

Module 1 — Plan, accounts and guardrails
- 01.1 · The project, the budget and the spending rules
- 01.2 · Accounts, API tokens and secrets
- 01.3 · Your project repo and a CPU dry run

Module 2 — Data: find it, license it, store it
- 02.1 · Choosing pretraining data
- 02.2 · Object storage for datasets and checkpoints
- 02.3 · Downloading FineWeb-Edu into your bucket

Module 3 — Your tokenizer and token shards
- 03.1 · Training your own BPE tokenizer
- 03.2 · Tokenizing 20B tokens on a rented CPU box
- 03.3 · Validation split, decontamination and the token report

Module 4 — GPUs: pick, rent, benchmark
- 04.1 · Where to rent GPUs and what they cost
- 04.2 · Your first H100 box
- 04.3 · Benchmarking throughput and fixing the budget

Module 5 — Baseline: reproduce GPT-2 124M
- 05.1 · The baseline config and a multi-GPU launch
- 05.2 · Checkpoints, resume and surviving a dead node
- 05.3 · Running the 10B-token baseline

Module 6 — Make it yours: ablations
- 06.1 · Designing a fair ablation
- 06.2 · Architecture ablations
- 06.3 · Optimizer and schedule ablations
- 06.4 · Sizing the main run and freezing the recipe

Module 7 — The main run
- 07.1 · Pre-flight checklist and launch
- 07.2 · Babysitting a long run
- 07.3 · Finishing the run and shutting everything down

Module 8 — Evaluate honestly
- 08.1 · Setting up lm-eval-harness
- 08.2 · Comparing against GPT-2 and your baseline
- 08.3 · Writing the evaluation report

Module 9 — Instruction tuning (SFT)
- 09.1 · Choosing SFT data and a chat template
- 09.2 · Running SFT and checking the chat model

Module 10 — Preference tuning (DPO)
- 10.1 · Preference data and the DPO run
- 10.2 · Measuring alignment with a judge

Module 11 — Ship it
- 11.1 · Publishing the model on Hugging Face
- 11.2 · A live demo Space
- 11.3 · The technical report and final cost ledger

Sidebar running numbers 01..32 in this order (01.1 → 01, …, 11.3 → 32).

## Shared conventions (fixed by modules 2 and 4 — all lessons follow them)

- **Price updates (2026-11-13):** RunPod Secure H100 SXM $4.39, Community $2.69; Lambda 8×H100
  $3.99/GPU-h rising to $4.59 on 2026-11-16 (single $4.89); Vast.ai H100 SXM from ~$1.87 (median
  $2.40); Backblaze B2 now $6.95/TB-mo. Planning rate stays $4.00/GPU-h until 04.3 replaces it
  with the learner's measured numbers. PyTorch PyPI wheels = CUDA 13.0 → NVIDIA driver ≥ 580.
  `hf_transfer` is gone (huggingface_hub ≥ 1.0); use `HF_XET_HIGH_PERFORMANCE=1`.
- **Data plan:** FineWeb-Edu `sample/10BT` (baseline, GPT-2 tokenizer) + first 30 files of
  `sample/100BT` (~64.6 GB, ~22.5B GPT-2 tokens; own tokenizer, ablations + main run).
- **Bucket:** rclone remote `r2`, root `MYGPT_REMOTE=r2:mygpt`, configured only by
  `RCLONE_CONFIG_R2_*` env vars. Prefixes: `raw/fineweb-edu/<subset>/`, `manifests/`,
  `tokenizer/`, `tok/<name>/`, `ckpt/<run_name>/`, `hf/`, `tmp/`. Shard sets:
  `tok/gpt2-10BT/` (baseline) and `tok/bpe32k-100BT-30/` (own 32,768-vocab tokenizer); locally
  `data/tok/<same name>`. Use `rclone copy`, never `sync`, unless a lesson says why.
- **Learner's project repo:** a copy of `labs/common` pushed to the learner's own GitHub repo;
  module lab helpers copied into its `tools/` folder. `budget.md` at its root, ledger rows
  `| date | lesson | item | estimate | actual |`.
- **Box setup:** `labs/module-04/setup_box.sh`; auto-shutdown
  `bash scripts/autoshutdown.sh --idle-minutes 5 --stop-cmd '...'` (OS shutdown does NOT stop
  billing on RunPod or Lambda — terminate via provider CLI/API).
