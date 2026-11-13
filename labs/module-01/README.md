# Module 1 lab files — plan, accounts and guardrails

Starting files for your project repo. Lesson 01.3 copies them into the `tools/` folder of your repo
(and `budget.template.md` to `budget.md`, `project.gitignore` to `.gitignore` at the repo root).
The project code itself (`mygpt/`, `scripts/`, `configs/`, `tests/`) is in `labs/common/`.

| File | Used in | What it does |
|---|---|---|
| `budget.template.md` | 01.1, every lesson that spends money | Your budget: the $500 plan, spending rules, kill checklist, accounts/alerts table and the ledger (`| date | lesson | item | estimate | actual |`). Becomes `budget.md`. |
| `secrets.env.example` | 01.2 | Environment variables for your HF, W&B and LLM API keys. Copy to `~/.config/mygpt/secrets.env` (outside any repo), `chmod 600`. |
| `check_accounts.py` | 01.2, 01.3, 02.2 | Checks each key against its service (HF whoami, W&B viewer, Anthropic/OpenAI model list, rclone on your bucket), the permissions of your secret files, and that no tracked git file contains a secret value. Never prints a secret. Standard library only. |
| `project.gitignore` | 01.3 | `.gitignore` for the project repo: secrets, virtualenv, data, shards, checkpoints. |

Requirements: Python 3.11+, git; `rclone` only for the R2 check (from lesson 02.2).
Checked against the Hugging Face, W&B, Anthropic and OpenAI APIs on 2026-11-13.

No secrets live in these files. Never commit a filled-in `secrets.env` or `r2.env`.
