---
id: "01.2"
module: 1
minutes: 20
practice_minutes: 75
prerequisites: ["01.1"]
objectives:
  - Create every account the project needs, in an order that surfaces slow verifications (Hetzner, Lambda) early.
  - Create a fine-grained, read-only Hugging Face token and a W&B API key, and store all secrets outside any git repo.
  - Configure the spending guardrails each provider offers (prepaid credit, auto top-up off, budget alerts, API spend limits).
  - Verify every key with a script that never prints a secret.
volatility: implementation
sources:
  - title: "Hugging Face: user access tokens (fine-grained tokens, revoking leaks)"
    url: https://huggingface.co/docs/hub/security-tokens
  - title: "W&B pricing (now hosted by CoreWeave)"
    url: https://coreweave.com/forge-pricing
  - title: "GitHub: push protection for users"
    url: https://docs.github.com/en/code-security/how-tos/secure-your-secrets/prevent-future-leaks/push-protection-for-users
  - title: "Cloudflare budget alerts"
    url: https://developers.cloudflare.com/billing/manage/budget-alerts/
  - title: "RunPod billing (minimum deposit, spend limit)"
    url: https://docs.runpod.io/accounts-billing/billing
  - title: "Vast.ai quickstart (sign up and add credit)"
    url: https://docs.vast.ai/guides/get-started/quickstart
  - title: "Vast.ai billing reference"
    url: https://docs.vast.ai/guides/reference/billing
  - title: "Lambda on-demand billing"
    url: https://docs.lambda.ai/public-cloud/on-demand/billing
  - title: "Hetzner fraud prevention FAQ (account verification)"
    url: https://docs.hetzner.com/general/security-and-identify/fraud-prevention-faq/
  - title: "Hetzner Cloud review 2026 (new-customer server restrictions)"
    url: https://betterstack.com/community/guides/web-servers/hetzner-cloud-review/
  - title: "OpenAI: prepaid API billing"
    url: https://help.openai.com/en/articles/8264644-setting-up-and-managing-prepaid-api-billing
  - title: "Anthropic: how do I pay for my API usage"
    url: https://support.anthropic.com/en/articles/8977456-how-do-i-pay-for-my-api-usage
last_verified: "2026-11-13"
---

# 01.2 · Accounts, API tokens and secrets

Nine services touch this project: one for code, one for models, one for experiment logs, one for storage, three for GPUs, one for cheap CPU boxes and one LLM API for the judge in Module 10. Some of them take days to verify a new account, and every one of them hands you a secret that can spend your money or delete your work. This lesson creates them all in one sitting, in the order that surfaces slow verifications early, gives each key the smallest permissions that work, and puts every secret in exactly two places: your password manager and a private file outside your repo.

## Goal

You have accounts on GitHub, Hugging Face, Weights & Biases, Cloudflare, RunPod, Hetzner Cloud, an LLM API (Anthropic or OpenAI) and, as fallbacks, Vast.ai and Lambda; two-factor authentication on all of them; a fine-grained read-only HF token and a W&B key in `~/.config/mygpt/secrets.env` (mode 600, outside any repo); the spending guardrails each provider offers switched on; and `check_accounts.py` reporting OK for every key you have.

## What you need

- `budget.md` from [01.1](lesson-01.md) and the [Module 1 lab files](../../labs/module-01/).
- A password manager (1Password, Bitwarden, KeePassXC or your browser's) and an authenticator app for 2FA.
- A **credit card in your own name**. Lambda accepts only major credit cards (debit and prepaid cards are common decline reasons); RunPod recommends deposits of at least $100 if you use a prepaid card.
- A government ID at hand: Hetzner may ask for one (or a card prepayment) before your first server.
- About 95 minutes, plus waiting time for verifications.
- **Expected cost: $0 spent today.** RunPod and Vast.ai credit is loaded later (04.1), the LLM credit in Module 10.

## Choices and costs

### The accounts (checked 2026-11-13)

| Service | Used for | Cost for this project | Signup facts that bite |
|---|---|---|---|
| **GitHub** | Project repo (private until Module 11) | Free | 2FA required for most active accounts. User-level push protection blocks known secret formats on pushes to **public** repos by default |
| **Hugging Face** | Downloading FineWeb-Edu (02), benchmarks (03), publishing the model and demo (11) | Free; PRO $9/month not needed | Tokens: read, write or **fine-grained**. Without a token you share a low anonymous rate limit |
| **Weights & Biases** | Loss curves and run comparison (05–10) | Free tier: 5 GB storage, up to 5 seats; Pro from $60/month (not needed) | The pricing page now lives on coreweave.com. Log metrics only; checkpoints go to R2, never to W&B (5 GB fills fast) |
| **Cloudflare** | R2 bucket for data and checkpoints (02.2) | ~$2.55/month at 180 GB, egress free | A payment card is required to activate R2 even inside the free tier. Budget alerts (since April 2026) warn, they do not cap |
| **RunPod** | Primary GPU provider: 1×H100 dev box, 8×H100 runs | Prepaid; minimum deposit $10 by card | Default spend limit $80/hour across all resources (a rate cap, not a total). A pod needs credit for at least one hour of its configuration to start |
| **Hetzner Cloud** | CPU boxes for downloading and tokenizing (02.3, 03) | CPX62 €0.1779/h; billed hourly | New accounts may need ID verification (ID upload + selfie) or a credit-card prepayment. Since 26 June 2026 Hetzner's status page has reported limited server creation for new customers |
| **Vast.ai** (fallback) | Cheaper 1-GPU boxes; 8×H100 fallback | Prepaid; minimum deposit $5 | A $0 balance stops instances but does not delete them; their disk keeps billing |
| **Lambda** (fallback) | 8×H100 fallback | Billed per minute, invoiced weekly to your card | Major credit cards only; billing address required before launch; no prepaid balance and no spending cap, so termination discipline is your only guard. Sign-up credit programs are retired |
| **LLM API**: Anthropic or OpenAI | Judge for the DPO win rate (10.2) | ~$5 prepaid | Prepaid credits. OpenAI: $5 minimum, credits expire after 1 year, the balance is not an instant cutoff. Anthropic: prepaid credits, optional auto-reload, monthly spend limits and alerts in the Console |

> [!NOTE]
> Signup rules and free tiers change often (Hetzner changed its new-customer policy and prices three times in 2026). If a step here does not match what you see, follow the provider's page and note the difference in `budget.md`.

### Why this order

Slow things first. **Hetzner** and **Lambda** can hold a new account for verification, and Hetzner may refuse to create servers for new customers at all for a while. You need a CPU box in 02.3 and an 8-GPU fallback in Module 5, so you want to know now, not on the day. If Hetzner will not give you a server, use a **RunPod CPU pod** instead (roughly $0.03–0.04 per vCPU-hour as of late 2026, shown in the RunPod console when you deploy); Modules 2 and 3 give the commands for both.

### What each secret can do, and how small to make it

| Secret | If it leaks, someone can | Make it small |
|---|---|---|
| HF token | Read (or write, if write) your private repos, impersonate you on the Hub | **Fine-grained, read-only.** A write token comes only in Module 11, scoped to your model and Space repos |
| W&B key | Read and write your runs | One key; rotate it at the end of the project |
| R2 keys (02.2) | Read, overwrite and delete your data and checkpoints | Scoped to the one bucket |
| LLM API key | Spend your API credit | Small prepaid balance, auto-reload off, monthly limit |
| RunPod / Vast.ai / Lambda API keys | Start machines on your card | Do not create them until a lesson needs one; RunPod pods get their own pod-scoped key automatically |
| GitHub | Push to and read your repo | Use your normal login on your laptop; never store a GitHub token on a rented box or in a template |

> [!CAUTION]
> A secret pasted once into a public place (a commit, a notebook output, a screenshot, a W&B config, a chat) is compromised, even if you delete it a minute later; bots scan GitHub continuously. Revoke it at the provider first, then create a new one, then clean up. Hugging Face also lets anyone revoke a leaked HF token they find.

## Do this

### 1. Password manager and 2FA (5 min)

Create a folder `mygpt` in your password manager. For every account below: generate a unique password, save it there, and turn on 2FA with an authenticator app right after signup. Save each provider's recovery codes in the same entry.

### 2. Hetzner Cloud first (10 min + waiting)

1. Sign up at hetzner.com (Cloud). Use your real name and address, a non-free email address if you have one, and no VPN; Hetzner's fraud checks flag those. Complete the ID check or the card prepayment if asked.
2. In the Cloud console create a project `mygpt`, then add a server to prove you can: the cheapest type in any EU location (for example CPX02, €0.0088/h), Ubuntu 24.04. If it is created, **delete it immediately** (server → Delete); the test costs one cent.
3. If the creation is refused or the account stays "in verification" for more than two business days, write in `budget.md` that you will use a RunPod CPU pod for Modules 2 and 3, and go on.

### 3. GitHub (5 min)

Sign up or log in at github.com, turn on 2FA, and under **Settings → Code security** check that **push protection for yourself** is on. Do not create the repo yet; that is 01.3.

### 4. Hugging Face and a fine-grained read token (10 min)

1. Sign up at huggingface.co and confirm your email.
2. **Settings → Access Tokens → Create new token → Fine-grained.** Name it `mygpt-read`. Under user permissions tick only the read permissions for repository contents (your own repos and public gated repos you can access). Leave every write, inference and organization permission off.
3. Copy the token (starts with `hf_`) into your password manager.

### 5. Weights & Biases (5 min)

Sign up at wandb.ai on the free plan. Open https://wandb.ai/authorize and copy your API key into the password manager.

### 6. Cloudflare (10 min)

1. Sign up at cloudflare.com, turn on 2FA.
2. Open **R2 Object Storage** and add a payment method when asked. Do not create the bucket yet (02.2).
3. **Manage Account → Billing → Billable Usage → Budget alerts** (or **Notifications → Add → Budget Alert**): create an alert at **$5** and a second at **$15**. R2 for this project should stay under $3 a month, so either email means something is wrong.

### 7. RunPod (5 min)

Sign up at runpod.io, turn on 2FA. Do **not** add credit yet (04.1 does, with automatic top-up off). Note the default $80/hour spend limit on the billing page; it caps the rate, not the total.

### 8. Fallback GPU providers (10 min)

- **Vast.ai:** sign up, turn on 2FA, leave the balance at $0 and auto-billing **off**. In the billing settings, set a low-balance email for later.
- **Lambda:** sign up at lambda.ai, add a major credit card and your billing address in **Settings → Billing**. If the card is declined, try another credit card in your own name; do not create a second account. Nothing is charged until you launch an instance.

### 9. LLM API for the judge (5 min)

Pick **one**: Anthropic (console.anthropic.com) or OpenAI (platform.openai.com). Create the account and turn on 2FA. Then:

- Set a **monthly spend limit of $10** (Anthropic: Console → Limits; OpenAI: organization limits) and an alert at $5.
- Leave auto-reload / auto-recharge **off**.
- Create an API key named `mygpt-judge` only if your provider lets you without credit; otherwise do it in Module 10 together with the $5 top-up.

### 10. Put the secrets in one private file (10 min)

Your project's convention: secrets live in `~/.config/mygpt/`, never inside a repo. R2 keys get their own `r2.env` in 02.2.

```bash
mkdir -p ~/.config/mygpt && chmod 700 ~/.config/mygpt
cp labs/module-01/secrets.env.example ~/.config/mygpt/secrets.env   # path to where you unzipped the lab files
chmod 600 ~/.config/mygpt/secrets.env
nano ~/.config/mygpt/secrets.env      # paste HF_TOKEN, WANDB_API_KEY, (LLM key), HF_USER, GITHUB_USER
```

Load it in any shell that needs it, and check every key:

```bash
set -a; . ~/.config/mygpt/secrets.env; set +a
python labs/module-01/check_accounts.py
```

> [!TIP]
> On Windows, run these in WSL or Git Bash, where `~/.config/mygpt` and `chmod` work as written. In PowerShell you can load the file with `Get-Content $HOME\.config\mygpt\secrets.env | ? { $_ -match '^[A-Z_]+=' } | % { $k,$v = $_ -split '=',2; Set-Item "env:$k" $v }`; the permission check is then skipped.

### 11. Fill in the accounts table in `budget.md`

Tick 2FA, payment type and the limit/alert you set for each service in the **Accounts, limits and alerts** table.

## How you know it worked

- `check_accounts.py` prints `OK` for Hugging Face with `token=fineGrained`, `OK` for Weights & Biases with your user name, `OK` for secret files (on Linux/macOS/WSL), and `SKIP` only for things you have not set up yet (LLM key, R2). It ends with `0 failed`.
- Your password manager has nine entries with passwords, 2FA recovery codes and the two keys.
- You created and deleted a Hetzner server (or wrote down why you will use RunPod CPU pods instead), and the Hetzner console shows no servers.
- Cloudflare shows two budget alerts; your LLM provider shows a $10 monthly limit; RunPod and Vast.ai show $0 balances with automatic top-up off.

## When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| Hetzner account "under review", or server creation refused | New-customer fraud checks and the 2026 capacity limits | Complete ID verification; wait up to two business days; meanwhile plan Modules 2–3 on a RunPod CPU pod. Don't open a second account |
| Lambda: card declined, or "add a billing address" | Debit/prepaid card, or international 3-D Secure checks | Add a major credit card in your own name and the billing address in Settings → Billing |
| `Hugging Face ... HTTP 401` | Token copied with a space or truncated, or you copied the token's name | Copy again from the password manager; create a new token if in doubt |
| `Hugging Face ... token=write` warning | You made a classic write token | Delete it; create a fine-grained read token. Write access comes in Module 11, scoped |
| `Weights & Biases ... key not accepted` | Old key, or a key from a different W&B host | Copy the key from https://wandb.ai/authorize again |
| `secret files ... readable by others` | File created without `chmod 600` | `chmod 600 ~/.config/mygpt/*.env` |
| `git ... secret found in tracked file(s)` | A key ended up in a file in the repo you ran it from | Revoke the key at the provider **now**, create a new one, then remove the file from git. Rewriting history does not un-leak a pushed secret |
| You put a key in your shell profile (`~/.bashrc`) | Convenience | Move it to `secrets.env`; profiles get copied into dotfile repos and screenshots |

## Artifact

- Accounts with 2FA on all nine services; secrets in your password manager and in `~/.config/mygpt/secrets.env` (mode 600, outside any repo).
- The accounts table in `budget.md` filled in.
- `budget.md` ledger line:

  ```text
  | 2026-11-13 | 01.2 | Accounts, 2FA, tokens, alerts; Hetzner test server ~1 min | $0.01 | $____ |
  ```
