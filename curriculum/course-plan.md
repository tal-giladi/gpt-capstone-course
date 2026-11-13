# Course plan — Train Your Own GPT: the real-world capstone

Status: PLAN FOR REVIEW (2026-11-13). Nothing below is written as lessons yet.

## 1. What this course is

A guided, do-it-for-real capstone for the LLM Research Engineer course. It does not teach new
theory; every lesson is a step you perform on real infrastructure with real money, and every
module ends with an artifact you keep. You finish with **your own ~GPT-2-class model**, built
from a GPT-2 baseline plus improvements you chose and proved with ablations, published on
Hugging Face with a demo and a technical report.

The "my model, not a copy" rule: the baseline is a GPT-2 124M reproduction (so you have an
honest reference), but the final model is your recipe — your architecture/optimizer/data
changes, each one justified by an ablation you ran and logged.

## 2. The model we aim for

| | Baseline (reference) | Your model (main run) |
|---|---|---|
| Size | GPT-2 124M | ~350M params (GPT-2-medium class); fallback ~124–200M if budget tight |
| Data | FineWeb-Edu sample-10BT (10B tokens) | FineWeb-Edu, ~20B tokens (sample-100BT subset) |
| Changes | none (faithful GPT-2) | 3–5 improvements chosen in Module 5 (candidates: RoPE, RMSNorm, SwiGLU, QK-norm, Muon optimizer, better LR schedule, longer context, own tokenizer) |
| Target | match GPT-2 124M on HellaSwag (~29%) | beat the original OpenAI GPT-2 of the same size on HellaSwag + val loss |

## 3. Budget (prices checked 2026-11-13, re-verified in each lesson)

Live H100 SXM on-demand prices seen today: RunPod ~$2.99–3.99/GPU-h, Lambda $3.99/GPU-h on
8× nodes ($4.29 single), Hyperbolic $3.19, Nebius $4.50, Vast.ai marketplace ~$1.3–3.6
(variable). **Budget is planned at $4.00/GPU-h** to leave margin.

| Item | Where | Estimate |
|---|---|---|
| Dev / debugging GPU (cheap 1-GPU box, ~20 h) | RunPod / Vast.ai (RTX 4090 / A100 / 1×H100) | $40 |
| Data download + tokenization (CPU box, a few hours) | RunPod CPU pod or Hetzner | $10 |
| Baseline: GPT-2 124M × 10B tokens (~12 GPU-h on 8×H100) | Lambda / RunPod / Hyperbolic | $50 |
| Ablations: ~10 short runs (124M × ~1B tokens) | 1×H100 | $60 |
| Main run: ~350M × ~20B tokens (~40 GPU-h on 8×H100) | same | $160 |
| Evaluation (lm-eval-harness, ~3 GPU-h) | 1×H100 | $12 |
| SFT + DPO (~6 GPU-h) | 1×H100 | $25 |
| Object storage (~150 GB, 3 months) | Cloudflare R2 ($0.015/GB-mo, free egress) or Backblaze B2 ($0.006/GB-mo) | $7 |
| Tracking, model hosting, demo | W&B free tier, HF Hub free, HF Space free CPU | $0 |
| **Subtotal** | | **~$365** |
| Contingency (crashed runs, idle boxes, price changes) +35% | | ~$130 |
| **Plan for** | | **~$500 hard cap** (expected ~$350–450) |

Cheaper path if wanted: make the main run a ~124–200M improved model on 10B tokens → total
about $200–250.

## 4. Modules (each lesson = a guided step; each module ends in an artifact)

| # | Module | What you do | Artifact you keep |
|---|---|---|---|
| 0 | Plan, accounts, budget guardrails | Create HF, W&B, GitHub, cloud and storage accounts; set spending limits/alerts; budget tracker; project repo from template | Project repo + `budget.md` with live prices + alerts configured |
| 1 | Data: get it, store it | Choose FineWeb-Edu (license ODC-BY), download sample-10BT/100BT shards, push to R2/B2, write a data card | Raw shards in your bucket + `DATA_CARD.md` |
| 2 | Tokenize and shard | GPT-2 tokenizer (and optionally train your own BPE), tokenize on a CPU box, write uint16 shards, held-out val split, decontaminate vs eval sets | Tokenized shards in bucket + token count report |
| 3 | GPUs: pick, rent, benchmark | Compare providers live, rent 1×H100, pinned Docker/uv env, smoke test, measure tokens/s and MFU, auto-shutdown script | Reproducible env + benchmark table |
| 4 | Baseline: reproduce GPT-2 124M | Multi-GPU (DDP) run on 8×H100, checkpoint to bucket, resume after preemption, W&B dashboard | Baseline checkpoint + loss curves + HellaSwag score |
| 5 | Make it yours: ablations | Run the improvement ladder on short runs, one change at a time, same seed/tokens; pick your recipe with evidence | Ablation table + chosen recipe (config) |
| 6 | The main run | Launch your model on 8×H100, monitor loss spikes, handle failures, checkpoint/resume, stay on budget | Your base model weights |
| 7 | Evaluate honestly | lm-eval-harness (HellaSwag, ARC, PIQA, …), compare to OpenAI GPT-2 and your baseline, contamination check | Eval report with tables/plots |
| 8 | Instruction tuning (SFT) | Pick an open SFT dataset (e.g. SmolTalk), chat template, loss masking, train, sanity-chat | Chat model checkpoint |
| 9 | Preference tuning (DPO) | Open preference data (e.g. UltraFeedback binarized), DPO, win-rate vs SFT | Aligned model + win-rate result |
| 10 | Ship it | Upload to HF Hub with model card, Gradio demo Space, technical report, clean GitHub repo, final cost ledger | Public model + live demo + report (your portfolio piece) |

Every lesson states: what to do, which providers/links, today's price, expected time and cost,
"how you know it worked", common failures and recovery. Each lesson has a 3–5 question
multiple-choice quiz; each module has an 8–10 question scenario quiz (Academy rules).

## 5. Code

`labs/common/` holds one small project repo you clone onto every GPU box: data prep, model
(baseline + switchable improvements via config flags), DDP training with checkpoint/resume to
S3-compatible storage, eval, SFT, DPO, upload scripts. It reuses your `llmre` code where it
fits, with pinned versions (PyTorch, tiktoken, datasets, lm-eval, TRL only after the from-
scratch versions you already wrote).

## 6. Open decisions for Tal

1. ~~Main run size~~ DECIDED 2026-11-13: ~350M on ~20B tokens, ~$500 cap.
2. Tokenizer: keep GPT-2's (clean comparison) or train your own (more "yours")?
3. Course name/slug: `gpt-capstone-course` OK?
