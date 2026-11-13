# Train Your Own GPT: the Real-World Capstone

A guided, hands-on capstone: rent real GPUs, download real data, and train, evaluate, align and publish your own GPT-2-class model for a few hundred dollars, with a real artifact at the end of every module.

This course does not teach theory. You already know transformers, BPE, AdamW, DDP, SFT and DPO. Every lesson is a step you perform on real infrastructure with real money: which company to use, what it costs today (prices checked 2026-11-13), the exact commands, how you know it worked, and what to do when it breaks. Each paid resource comes with priced alternatives, so you are never locked into one vendor.

## What you will end up with

| Module | You do | Artifact you keep |
|---|---|---|
| 1 — Plan, accounts and guardrails | Accounts, secrets, spending limits, project repo | Your repo, green CPU tests, `budget.md` |
| 2 — Data | Pick and license FineWeb-Edu, set up object storage, download | Raw data in your bucket + data card + manifest |
| 3 — Tokenizer and shards | Train your own 32k BPE, tokenize ~22B tokens, decontaminate | Tokenizer + token shards in your bucket + token report |
| 4 — GPUs | Compare clouds, rent an H100, benchmark | Box setup script + measured throughput + revised budget |
| 5 — Baseline | Reproduce GPT-2 124M on 8×H100 with checkpoint/resume | Baseline checkpoint + `results.md` |
| 6 — Ablations | Prove which improvements help | `results/ablations.md` + frozen ~350M recipe |
| 7 — Main run | Train your ~350M model on ~20B tokens | Your base model in Hugging Face format |
| 8 — Evaluation | lm-eval-harness vs OpenAI GPT-2 and your baseline | `EVAL_REPORT.md` |
| 9 — SFT | Instruction-tune on open chat data | Chat model |
| 10 — DPO | Preference-tune and measure with an LLM judge | Aligned model + `winrate.json` |
| 11 — Ship it | Publish weights, a live demo and a technical report | Public model, demo Space, report, final cost ledger |

## Budget

Planned total about $350–450, hard cap $500. GPU time is most of it (baseline ~$50, ablations ~$40–60, main run ~$120–190, SFT/DPO/eval ~$30); storage is a few dollars and everything else uses free tiers. Module 4 replaces the plan with numbers you measure yourself.

> [!WARNING]
> This course spends real money on rented cloud machines. A forgotten 8×H100 node costs about $30–35 per hour. Set the spending limits in Module 1 before you rent anything.

## How to use it

Start with [01 · The project, the budget and the spending rules](lessons/module-01/lesson-01.md). The project code lives in [`labs/common/`](labs/common/README.md), a small, tested PyTorch package called `mygpt`; each module's helpers are in `labs/module-NN/`.
