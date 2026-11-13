---
id: "11.3"
module: 11
minutes: 25
practice_minutes: 180
prerequisites: ["11.1", "11.2"]
objectives:
  - Write a technical report whose every number traces to a committed file, with a planned-vs-actual cost table built from budget.md.
  - Choose where to publish the report (GitHub README, Hugging Face article, personal blog) and explain why arXiv is not the place for a reproduction project.
  - Shut down or delete every paid resource across all providers, decide what stays in the bucket from its monthly cost, and revoke or rotate every key.
  - Present the project as a portfolio piece that shows engineering judgment, not just a model.
volatility: implementation
sources:
  - title: "Cloudflare R2 pricing (free tier 10 GB-month, Infrequent Access)"
    url: https://developers.cloudflare.com/r2/pricing/
  - title: "Hugging Face blog articles (who can publish)"
    url: https://huggingface.co/docs/hub/main/blog-articles
  - title: "arXiv moderation (scholarly interest, course projects may be declined)"
    url: https://info.arxiv.org/help/moderation/index.html
  - title: "arXiv endorsement policy"
    url: https://info.arxiv.org/help/endorsement.html
  - title: "Displaying carbon emissions for your model (ML CO2 calculator, Code Carbon)"
    url: https://huggingface.co/docs/hub/model-cards-co2
  - title: "NVIDIA H100 product page (SXM up to 700 W)"
    url: https://www.nvidia.com/en-us/data-center/h100/
  - title: "The FineWeb Datasets (Penedo et al., 2024)"
    url: https://arxiv.org/abs/2406.17557
  - title: "RunPod pod pricing and billing (stopped volumes keep billing)"
    url: https://docs.runpod.io/pods/pricing
  - title: "Hetzner Cloud regular-performance (CPX) prices"
    url: https://www.hetzner.com/cloud/regular-performance/
last_verified: "2026-11-13"
---

# 11.3 · The technical report and final cost ledger

The model and the demo show what you built; the technical report shows how you think. It is the document a hiring manager or a collaborator reads to decide whether your numbers can be trusted: what data, which changes and why, what it cost against the plan, what broke, and how someone else would reproduce it. This lesson turns the reports you have committed since Module 1 into one honest write-up, closes the ledger with the real bills, and then does the unglamorous part that protects your wallet: every machine, volume, bucket prefix and key you created gets an explicit keep-or-delete decision.

## Goal

`REPORT.md` at the root of your GitHub repo (also published as an article or blog post), with abstract, data, tokenizer, architecture and ablations, training, evaluation, alignment, a planned-vs-actual cost table generated from `budget.md`, lessons learned and a reproducibility table of commits, configs and data revisions. `budget.md` is closed with every actual amount filled in from the providers' billing pages; every provider's resource list is empty except what you chose to keep, with its monthly cost written down; every key you no longer need is revoked. The repo README presents the project in five lines with links to the model, the demo and the report.

## What you need

- Everything committed in Modules 1–11: `budget.md`, `DATA_CARD.md`, `TOKEN_REPORT.md`, `results/ablations.md`, `decisions/0001-main-run-recipe.md`, `EVAL_REPORT.md` (08.3), `reports/` (main-run log, SFT and DPO reports, `winrate.json`, `demo.md`), `configs/`.
- Logins to every provider you used: RunPod (and Lambda, Vast.ai if you used them), Hetzner, Cloudflare, Hugging Face, W&B, GitHub, your LLM API console.
- The [Module 11 lab files](../../labs/module-11/): `ledger_summary.py` in `tools/`, `TECH_REPORT.template.md`.
- About 3–4 hours of writing and 30 minutes of cleanup. **Expected cost: $0**, and the cleanup should bring your monthly run-rate to about $0.

## Choices and costs

### What the report contains, and where each part comes from

| Section | Content | Source in your repo |
|---|---|---|
| Abstract | Size, tokens, tokenizer, cost, headline result with error bars | All below |
| Data | Dataset, subsets, revision, license, decontamination | `DATA_CARD.md`, `manifests/`, `TOKEN_REPORT.md` |
| Tokenizer | 32k BPE vs GPT-2 bytes/token | `reports/compare_tokenizers.json` |
| Architecture and ablations | Ablation table, seed noise, chosen recipe | `results/ablations.md`, `decisions/0001-*.md` |
| Training | Hardware, throughput, MFU, wall time, loss curve, incidents | `reports/main-run-log.md`, `log.jsonl`, W&B |
| Evaluation | vs OpenAI GPT-2 124M/355M and your baseline, bpb on common val | `EVAL_REPORT.md` (08.3) |
| Alignment | SFT/DPO data and settings, probes, win rate with CI and biases, regression | `reports/sft_*.md`, `alignment_results.md`, `winrate.json` |
| Cost | Planned vs actual per module, GPU-hours, CO2 estimate | `budget.md` via `ledger_summary.py` |
| Lessons learned | What went wrong, what it cost, what you would change | Run logs, ledger |
| Reproducibility | Commits, tags, configs, data and model revisions, environment | git, Hub tags, manifests |

Write it in the order of the template, but draft sections 8–10 first: the ledger and the reproducibility table expose gaps (a missing actual, an unpinned dataset) while you can still fix them.

### Where to publish it

| Venue | Cost (checked 2026-11-13) | Audience | Verdict |
|---|---|---|---|
| **`REPORT.md` in your GitHub repo**, linked from the README | $0 | Anyone who finds the code; renders tables and math | **Always.** The canonical copy, next to the evidence |
| Hugging Face article (`huggingface.co/new-blog`) | Needs a PRO subscription ($9/month) to publish under your own name | ML practitioners on the Hub; linked from your model page automatically | Worth one month of PRO if you want the visibility; "announce the release of an open-source artifact" is an accepted category |
| Personal blog (GitHub Pages, your own site) | $0 (a custom domain ~$10–15/year) | People who follow you | Good for a shorter narrative version with the cost story |
| LinkedIn / X / forums post | $0 | Recruiters, peers | Three sentences + the loss curve + links |
| arXiv | $0, but needs endorsement for a first submission | Researchers | **Not appropriate** unless you found something new (below) |

**Why not arXiv.** arXiv's moderators expect submissions of "scholarly archival interest" and may decline work that lacks originality, novelty or significance, naming course projects explicitly. A faithful GPT-2-class reproduction with known improvements (RoPE, RMSNorm, SwiGLU, QK-norm, Muon) is valuable engineering and a great portfolio piece, but it is not new research. If your ablations found something that contradicts or extends published results, with enough seeds to be sure, write that finding up as a short focused paper, get an endorser, and then arXiv is the right place.

### What to keep after the project (R2, checked 2026-11-13)

R2 Standard storage is $0.015/GB-month with the first **10 GB-month free**, and zero egress; Infrequent Access is $0.01/GB-month plus $0.01/GB to read, with a 30-day minimum.

| Prefix | Size (typical) | Monthly cost | Keep? |
|---|---|---|---|
| `ckpt/main_350m/` final + pre-decay checkpoints, `log.jsonl` | ~5.8 GB | ~$0.09 (inside the free 10 GB) | **Keep the final one at least**: continued training, exact reproduction |
| `hf/` (base, SFT, DPO exports) | ~4 GB | ~$0.06 | Optional: the Hub now holds the public copies; keep if you want a private backup |
| `ckpt/baseline_124m/` final + log | ~1.5 GB | ~$0.02 | Keep the log; the checkpoint only if you will compare again |
| `tokenizer/`, `manifests/`, `DATA_CARD.md`, reports | < 100 MB | ~$0 | **Keep** (also in git) |
| `tok/bpe32k-100BT-30/` shards | ~45 GB | ~$0.68 | Delete unless you plan more training within ~3 months; re-tokenizing costs a few CPU-box hours |
| `tok/gpt2-10BT/`, common-val sets | ~20 GB | ~$0.30 | Delete (keep the common-val sets if small and you will compare again) |
| `raw/fineweb-edu/` | ~93 GB | ~$1.40 | **Delete**: re-downloadable from Hugging Face at the pinned revision in your manifests |
| `tmp/`, rolled-back or extra checkpoints | varies | | Delete |

Keeping only the checkpoint, logs, tokenizer and manifests fits in the free 10 GB: $0/month. Keeping everything is about $2.70/month; not ruinous, but money spent on nothing if you never come back.

### Every provider, every billable thing

| Provider | What can still bill | Where to check | Keep? |
|---|---|---|---|
| RunPod | Pods (running **or stopped**: stopped volume disk $0.20/GB-month), network volumes ($0.07/GB-month), credit balance | Console → Pods, Storage, Billing | Delete all pods and volumes; leave remaining credit or ask support about it |
| Lambda | Running instances (billed until terminated), persistent filesystems | Console → Instances, Storage | Terminate all; delete filesystems |
| Vast.ai | Instances, including stopped ones (storage keeps billing) | Console → Instances | Destroy all |
| Hetzner Cloud | Servers (billed even when powered off), volumes, snapshots, backups, primary IPs | Console → project → Servers, Volumes, Snapshots, Primary IPs | Delete all |
| Cloudflare R2 | Storage above 10 GB-month | Dashboard → R2 → bucket → Metrics; Billing | Keep the bucket at the size you chose above |
| Hugging Face | PRO subscription (if you took it for 11.2 or the article), paid Space hardware | Settings → Billing | Cancel PRO when done; Space on ZeroGPU / CPU basic |
| LLM API (Anthropic/OpenAI) | Prepaid credit, auto-reload | Console → Billing, API keys | Auto-reload off; revoke the judge key |
| W&B | Free tier; storage cap | Settings → Usage | Keep (your runs are evidence) |
| GitHub | Free for public repos | Settings → Billing | Keep |

> [!CAUTION]
> Deleting is final. A deleted pod, volume, Hetzner server or R2 object cannot be recovered, and `rclone delete` has no undo. Delete only after the thing you need is listed, with the right size, somewhere else (bucket, Hub, git), and run every `rclone delete` with `--dry-run` first.

## Do this

### 1. Close the ledger (30 min)

Open each provider's billing or usage page and fill every `$____` in `budget.md` with the real amount (convert EUR to USD at the rate on the invoice). Then:

```bash
python tools/ledger_summary.py budget.md --cap 500 | tee reports/cost_table.md
```

The output lists planned and actual per module, the total, the share of the $500 cap used, and any row still missing an actual. Repeat until that list is empty. Add one row per month for what you keep running (step 4).

### 2. Write the report (2–3 h)

```bash
cp tools/TECH_REPORT.template.md REPORT.md
```

Fill it section by section from the table above. Rules that make it credible:

- **Every number cites a file** in brackets, e.g. `[reports/winrate.json]`. If you cannot point to a file, either produce the file or remove the number.
- **Error bars or sample sizes** next to every evaluation number; say which differences exceed two standard errors and which do not.
- **Label ESTIMATES** (CO2, anything not measured) and show their arithmetic. CO2 for the whole project: total H100-hours from the ledger × 0.7 kW × 1.2 PUE × your region's grid intensity. For example 120 H100-hours → ~100 kWh → ~40 kg CO2e at 0.4 kg/kWh.
- **Negative results stay in.** An ablation that did not help, a DPO win rate whose CI contains 50%, a rollback that cost $14: these are what make the positive numbers believable.
- **Reproducibility table:** the commit of tag `release-v1.0`, `configs/main_350m.yaml`, the FineWeb-Edu revision from your manifests, the SFT and DPO dataset revisions, `you/my-gpt-350m@v1.0`, lm-eval 0.4.13, the pins in `pyproject.toml`. Paste the exact commands from your run logs.
- Put the loss curve and the ablation chart as PNGs in `assets/` and link them.

### 3. Shut down and revoke, provider by provider (30 min)

Go through the provider table above in order. On the command line where a CLI exists:

```bash
runpodctl pod list                        # must print no pods; delete leftovers in the console
hcloud server list && hcloud volume list && hcloud image list --type snapshot && hcloud primary-ip list
vastai show instances                     # if you used Vast.ai
```

Then revoke or rotate keys. Do this in each console, not by deleting a local file:

| Key | Action |
|---|---|
| RunPod, Lambda, Vast.ai, Hetzner API tokens | **Revoke** (create a new one if you ever come back) |
| R2 API token used by rclone | Revoke the read-write token; keep a **read-only** token only if you need scripted downloads |
| Hugging Face write token | **Revoke**; keep or create a fine-grained read token if you need one |
| Anthropic / OpenAI judge key | **Revoke**; turn auto-reload off |
| W&B API key | Rotate if it ever sat on a rented box |
| SSH keys added to providers | Remove from the provider accounts |

Finally delete the local copies: `~/.config/mygpt/secrets.env` and `r2.env` lines for revoked keys, and the `.env` files on any remaining machine.

> [!WARNING]
> A key that sat on a rented machine may still exist in that machine's disk image, shell history or a snapshot you forgot. Treat every key that was ever on a pod or VPS as exposed: revoke it, do not just stop using it.

### 4. Trim the bucket (15 min)

```bash
rclone size r2:mygpt                                      # before
rclone delete r2:mygpt/raw/ --dry-run                     # read the list
rclone delete r2:mygpt/raw/
rclone delete r2:mygpt/tok/gpt2-10BT/ --dry-run && rclone delete r2:mygpt/tok/gpt2-10BT/
rclone delete r2:mygpt/tmp/ --dry-run && rclone delete r2:mygpt/tmp/
rclone size r2:mygpt                                      # after: write it into budget.md
```

Apply your own keep list from the table; keep `ckpt/main_350m/` with at least the final checkpoint. Add the monthly cost of what remains to `budget.md` (often $0.00 under the 10 GB free tier).

### 5. Frame it as a portfolio piece (30 min)

Rewrite the top of your repo's `README.md`:

```markdown
# my-gpt-350m — a GPT-2-medium-class model trained from scratch for $___

337M parameters, own 32k BPE tokenizer, 20B tokens of FineWeb-Edu on 8×H100, then SFT and DPO.
Beats OpenAI GPT-2 medium on HellaSwag (__._ vs __._, ±0.5) at ___ % of the planned budget.

[Model](https://huggingface.co/<you>/my-gpt-350m) · [Demo](https://huggingface.co/spaces/<you>/my-gpt-350m-chat) · [Technical report](REPORT.md) · [Cost ledger](budget.md)
```

Use your real results; if you did not beat GPT-2, say what you matched and what you learned. What reviewers look for, in order: evidence-backed decisions (ablations, not guesses), honest evaluation with error bars, cost control (planned vs actual), recovery from failures, and reproducibility. Pin the GitHub repo and the Space on your profiles, and add the project to your CV as one line with the three links.

Optionally publish the report as a Hugging Face article (needs PRO) or a blog post, linking back to `REPORT.md` as the canonical version.

## How you know it worked

- `ledger_summary.py` reports no rows without an actual, and the total matches the sum of your providers' invoices to within a few dollars.
- Every number in `REPORT.md` has a file reference; a friend can follow the reproducibility table to the exact commit, config, data revision and model tag.
- RunPod, Lambda, Vast.ai and Hetzner lists show no instances, pods, volumes, snapshots or IPs; their billing pages show no charges after today.
- Every API token you listed is revoked or rotated, and the providers' token pages show only what you chose to keep.
- `rclone size r2:mygpt` matches your keep list, and `budget.md` has the monthly run-rate line.
- The repo README shows the model, demo and report links in its first lines.

## When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| Ledger total is well below the invoices | Unlogged items: idle time, stopped volumes, taxes, failed runs | Add a row per invoice line; that gap is itself a lesson for the report |
| A charge appears days after "everything was deleted" | A stopped pod's volume, a Hetzner snapshot or primary IP, a RunPod network volume | Check each provider's storage pages, not only the instance list |
| You cannot reproduce a number in the report | Generated by an uncommitted script or an older checkpoint | Find the file or rerun; otherwise remove the number and say so |
| `rclone delete` removed more than intended | No `--dry-run`, or a too-broad path | Restore from the Hub or git if possible; afterwards always dry-run and name the prefix |
| Report is long and nobody reads it | Everything in prose | Lead with the abstract and three tables (eval, cost, ablations); move detail to an appendix |
| Tempted to call the model "state of the art" | Comparing against weak or mismatched baselines | Compare like with like (size, tokens, eval settings) and say exactly that |

## Artifact

- In git, tagged `report-v1.0`: `REPORT.md`, `reports/cost_table.md`, figures in `assets/`, the updated `README.md`, and `budget.md` with every actual filled in.
- Accounts: no running or stopped compute anywhere, keys revoked, bucket trimmed to your keep list; public model repos, demo Space and report links working.
- Final `budget.md` lines:

  ```text
  | 2026-11-13 | 11.3 | Report, cleanup, keys revoked; bucket trimmed to ~7 GB | $0.00 | $0.00 |
  | 2026-11-13 | 11.3 | Ongoing: R2 ~7 GB (free tier), HF Hub public, Space on ZeroGPU | $0.00/month | $0.00/month |
  ```
