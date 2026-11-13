# Budget — my GPT capstone

Hard cap: **$500**. Expected: $350–450. Planning rate until lesson 04.3: **$4.00 per GPU-hour**.
Path: [ ] main (~350M x 20B tokens)  [ ] cheaper (124–200M x 10B tokens)

## Spending rules (signed: ____________ date: __________)

1. Every rented box gets a hard stop (timer) and an idle guard before any real work starts.
2. Done means **terminated/deleted**, not stopped or powered off. I check the provider's instance list afterwards.
3. GPU providers run on prepaid credit with automatic top-up **off**. Credit is loaded for the next step only.
4. Spending alerts are on everywhere a provider offers them (table below).
5. I check the live price before every rental and write the price I actually pay in the ledger.
6. Every session gets a ledger line the same day: estimate before, actual after.
7. Stop-loss: if a step's actual cost exceeds its estimate by 25% or more, I stop and re-plan before spending more.
8. I go through the kill checklist every time I leave a box.

## Kill checklist (every time I leave a box)

- [ ] Results are in the bucket and `rclone check` reported 0 differences (or code/results are pushed to git).
- [ ] The instance is **terminated / deleted** in the provider console or CLI.
- [ ] The provider's instance list is empty (no stopped pods, no leftover volumes I did not mean to keep).
- [ ] The billing/usage page shows the spend I expected.
- [ ] The ledger line below has the actual cost.

## Plan (from the course plan; lesson 04.3 replaces the GPU lines with measured numbers)

| Item | Where | Estimate |
|---|---|---|
| Dev / debugging GPU (~20 h, 1 GPU) | RunPod Community / Vast.ai | $40 |
| Data download + tokenization (CPU box, a few hours) | Hetzner Cloud or RunPod CPU | $10 |
| Baseline GPT-2 124M x 10B tokens (~12 GPU-h incl. overhead) | 8xH100 node | $50 |
| Ablations (~10 short runs, 124M x ~1B tokens) | 1xH100 | $60 |
| Main run ~350M x ~20B tokens (~40 GPU-h incl. overhead) | 8xH100 node | $160 |
| Evaluation (~3 GPU-h) | 1xH100 | $12 |
| SFT + DPO (~6 GPU-h) | 1xH100 | $25 |
| LLM judge API (Module 10) | Anthropic or OpenAI, prepaid | $5 |
| Object storage (~180 GB, 3 months) | Cloudflare R2 | $8 |
| Tracking, model hosting, demo | W&B free, HF Hub free, HF Space free CPU | $0 |
| **Subtotal** | | **$370** |
| Contingency (crashed runs, idle boxes, price changes) | | $130 |
| **Hard cap** | | **$500** |

## Accounts, limits and alerts

| Service | Account | 2FA | Payment | Limit / alert set | Checked |
|---|---|---|---|---|---|
| GitHub | | | free | n/a | |
| Hugging Face | | | free | n/a | |
| Weights & Biases | | | free | 5 GB storage cap | |
| Cloudflare (R2) | | | card | budget alert at $5 | |
| RunPod | | | prepaid, auto top-up off | default $80/h spend limit; low-balance email | |
| Vast.ai (fallback) | | | prepaid, auto-billing off | low-balance email | |
| Lambda (fallback) | | | card, pay after use | none available: terminate discipline | |
| Hetzner Cloud | | | card / prepayment | server limit, usage page | |
| LLM API (judge) | | | prepaid, auto-reload off | monthly spend limit $10 | |

## Assumptions

- Storage: ~180 GB in R2 for ~3 months (raw parquet ~93 GB, token shards ~65 GB, checkpoints ~20 GB).

## Ledger

| date | lesson | item | estimate | actual |
|---|---|---|---|---|
| YYYY-MM-DD | 01.1 | Budget and spending rules | $0.00 | $0.00 |

**Spent so far:** $0.00 of $500
