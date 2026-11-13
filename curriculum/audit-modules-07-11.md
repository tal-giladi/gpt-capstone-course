# Audit: Modules 7–11 (resources, alternatives, artifact chain)

Date: 2026-11-13. Scope: `lessons/module-07..11`, their quizzes, `assessments/module-07..11-quiz*`,
`labs/module-07..11` (read only, no lab code changed). Dry run after the changes:
`npm run check-course` → 0 problems (32 lessons, 43 quiz files, 229 questions).

Prices added below were checked on 2026-11-13 and cited in each lesson's `sources`. Third-party
figures (not from the provider's own page) are labelled as such in the lesson.

## Artifact chain

| Module | Produces (`## Artifact`) | Consumed by (`## What you need`) | Status |
|---|---|---|---|
| 6 (06.4) | `configs/main_350m.yaml` @ tag `recipe-v1`, `decisions/0001-main-run-recipe.md`, `results/ablations.md`, budget estimate | 07.1 (config, tag, estimate); 11.3 (ablations, decision) | OK |
| 7 (07.3) | `r2:mygpt/hf/my-gpt-350m-base/`, `ckpt/main_350m/{ckpt_00038146.pt, ckpt_00030000.pt, log.jsonl}`, `reports/main-run-log.md`, budget line | 08.1 (`hf/my-gpt-350m-base/`), 08.3 (`log.jsonl`, run log), 09.1, 10.2, 11.1 | OK; 09.1 and 10.2 said "from Module 8" → fixed to 07.3 |
| 8 (08.3) | `EVAL_REPORT.md`, `reports/eval_table.md`, `reports/bpb.md`, `reports/contamination.md`, `evals/` | 11.1 (`EVAL_REPORT.md`), 11.3 | OK (Module 9 does not need it; 09.1 now says it is used again in Module 11) |
| 9 (09.2) | `r2:mygpt/hf/my-gpt-350m-sft/`, `reports/sft_data.md`, `reports/sft_probe.md` | 10.1 (`hf/my-gpt-350m-sft/`), 11.1, 11.3 | OK |
| 10 (10.1, 10.2) | `hf/my-gpt-350m-dpo/`, `hf/my-gpt-350m-dpo-b01/`, `reports/winrate.json`, `reports/alignment_results.md` | 10.2 (DPO models), 11.1 (chosen DPO model, `winrate.json`, `alignment_results.md`) | OK |
| 11 | Hub repos `<you>/my-gpt-350m` (+`sft` branch, tag `v1.0`), `<you>/my-gpt-350m-base`; Space `<you>/my-gpt-350m-chat`; `REPORT.md`, `reports/cost_table.md`, tag `report-v1.0` | Final portfolio (11.2 consumes 11.1's repo/tag; 11.3 consumes `reports/demo.md`) | OK |

Cross-module inputs from Modules 1–6 checked by name: `DATA_CARD.md` (02.1/02.3), `TOKEN_REPORT.md`,
`tok/common-val-*` (03.3), `reports/compare_tokenizers.json` (03.1), `ckpt/baseline_124m/ckpt_00019073.pt`
+ `log.jsonl` + `results.md` (05.3), templates `mygpt-gpu` (04.2) and `mygpt-8gpu` (05.1). All match.

## Per lesson

### 07.1 · Pre-flight checklist and launch
- Resources: 8×H100 node; experiment tracking; phone alerts; provider credit.
- Alternatives added: 8×H100 → Hyperbolic ($3.19/GPU-h, ~$25.50/h), Nebius ($4.50/GPU-h, ~$36/h), Prime Intellect (aggregator, $1.50–4/GPU-h third-party); now 6 on-demand options plus spot/reserved. Tracking → new table: W&B Free (5 GB, Pro $60/mo) recommended, no tracker (`wandb_project: ""`, preflight WARN), Trackio (free; pretraining needs a code request). Alerts → self-hosted ntfy via `--ntfy-server`.
- Do this: provider table for the hard stop / idle guard / final terminate (Vast.ai `stop`/`destroy instance`, Lambda Cloud API, others), and how to bring `r2.env` without a template. Replaced the inaccurate "stop call you tested in 04.2" (04.2 tests only RunPod).
- Quiz: q5 added (non-RunPod launch: both stop guards + sourcing `r2.env`).

### 07.2 · Babysitting a long run
- Resources: the running node (07.1), W&B, ntfy. No new paid resource.
- Change: node death may resume at any provider from 07.1's table (bucket holds the state); watcher price and ledger note.

### 07.3 · Finishing the run and shutting everything down
- Resources: bucket storage for checkpoints and HF export; optional second copy.
- Alternatives added for the second copy: second bucket at another company (B2 $6.95/TB-mo), laptop/external disk (besides private Hub repo, bucket-only). Note that a non-R2 main bucket works via `MYGPT_REMOTE`.
- Do this: terminate instructions for Vast.ai (`destroy`, not `stop`), Hyperbolic/Nebius/Prime Intellect; billing-check table row added.

### 08.1 · Setting up lm-eval-harness
- Resources: eval GPU.
- Alternatives added: Vast.ai 1×H100 (from ~$1.73, median ~$2.40; third-party snapshot), Lambda 1×A10 $1.29 / 1×H100 $4.89, Hyperbolic $3.19, Kaggle/Colab free (pipeline tests only: no bf16 on T4/P100).
- Do this: non-RunPod setup (source `r2.env`, provider stop command, `--batch 16` on 24 GB).

### 08.2 · Comparing against GPT-2 and your baseline
- Resources: same GPU box; optional CPU box for the WinoGrande/LAMBADA scan.
- Alternatives added: CPU box table (Hetzner CPX62 recommended, the H100 pod, laptop $0). Terminate line for other providers.

### 08.3 · Writing the evaluation report
- Resources: laptop only, $0. No change needed (plots come from `log.jsonl`, so no tracker account is required).

### 09.1 · Choosing SFT data and a chat template
- Resources: SFT datasets (5 already compared with licenses: smol-smoltalk, smoltalk, smoltalk2, Tulu 3, OpenHermes); SFT GPU.
- Alternatives added: GPU table now covers Vast.ai, Lambda, Hyperbolic besides RunPod H100 SXM/PCIe/4090.
- Artifact fix: base model provenance "from Module 8" → exported in 07.3.

### 09.2 · Running SFT and checking the chat model
- Resources: GPU box; optional tracker.
- Alternatives added: Trackio via `--report-to trackio` (works today through TRL/Transformers 5.22), `none` default.
- Do this: non-RunPod setup and the warning that `autoshutdown.sh` without `--stop-cmd` falls back to an OS shutdown (no billing stop off RunPod). Terminate line for other providers.

### 10.1 · Preference data and the DPO run
- Resources: preference datasets (5 already compared with licenses); GPU box.
- Alternatives added: cost of the same run on Vast.ai / Hyperbolic / Lambda; setup pointer to 09.2.
- Quiz: 10.1 q1 correct option shortened (it was clearly the longest; pre-existing).

### 10.2 · Measuring alignment with a judge
- Resources: judge API; GPU for answer generation + lm-eval; prompt set (no_robots, CC BY-NC, eval only).
- Alternatives added: Gemini 3.1 Pro Preview ($2/$12) and Gemini 3.8 Flash ($0.75/$3.75 until 2026-12-31), gpt-oss-120b via OpenRouter/Groq (~$0.15/$0.60), self-hosted gpt-oss-120b with vLLM on the eval H100 (~$1 of pod time, no account).
- Do this: base-URL table for `--judge openai --base-url` (Gemini, OpenRouter, Groq, localhost vLLM) with `--price-in/--price-out`; vLLM in a separate venv; idle-guard caveat; keep the pod if self-hosting.
- Artifact fix: base model provenance → 07.3.
- Quiz: q5 added (no API account → self-hosted open-weight judge).

### 11.1 · Publishing the model on Hugging Face
- Resources: model hosting. Already had HF Hub + ModelScope, Kaggle Models, GitHub Releases, R2 public, with a GitHub Releases backup step. No change.

### 11.2 · A live demo Space
- Resources: demo hosting. Already had ZeroGPU, CPU basic (PRO), CPU upgrade, T4, Modal, Replicate, RunPod Serverless, Hetzner VPS, local.
- Do this: added "host elsewhere" table (VPS command with `GRADIO_SERVER_NAME`, Modal/Replicate/RunPod need a wrapper not in the lab, local recording).

### 11.3 · The technical report and final cost ledger
- Resources: report venue (already 5), code hosting, CO2 estimate, cleanup of every provider.
- Alternatives added: code hosting (GitHub recommended, GitLab Free, Codeberg), CO2 methods (formula, ML CO2 calculator, CodeCarbon). Cleanup and key-revocation tables now include Hyperbolic, Nebius, Prime Intellect, B2, Gemini/OpenRouter/Groq keys, Trackio.

## Cross-scope issues (Modules 1–6, not changed here)
1. 04.2 sets up and tests only the RunPod stop path (`runpodctl`). Later lessons now explain Vast.ai/Lambda stop commands, but nobody tests them before the main run. Suggest 04.2 add a "test your fallback provider's stop command on a 1-GPU box" step.
2. 01.2 asks for an Anthropic or OpenAI account for the judge; 10.2 now also supports Gemini, OpenRouter, Groq and a self-hosted judge. 01.2 could mention these (and that no account is needed if self-hosting).
3. Modules 4–5 cite `lambda.ai` pricing URLs, which now 301-redirect to `lambda.com`; still work.
4. Pretraining with Trackio needs a `labs/common` change: filed as item 4 in `curriculum/code-requests/module-07.md`.

## Not verifiable on 2026-11-13
- Vast.ai live price grid did not load; figures are third-party snapshots (from $1.73, median $2.40).
- Groq gpt-oss-120b price from third-party trackers (Groq's pricing page has no figures in its text).
- Prime Intellect price range from a third-party directory; Hyperbolic 8-GPU on-demand availability is not stated on its page.
- Lambda Cloud API terminate path: docs page truncated, so the lesson names the endpoint section rather than giving a curl line.
- Kaggle weekly GPU quota (no official number; ~30 h commonly reported). GitLab Free limits from third-party summaries.
