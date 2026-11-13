---
id: "11.1"
module: 11
minutes: 25
practice_minutes: 120
prerequisites: ["10.2", "07.3", "02.3", "01.2"]
objectives:
  - Choose where to publish the weights (Hugging Face Hub, ModelScope, Kaggle Models, GitHub Releases, a public R2 bucket) from current limits and prices, and justify the choice.
  - Choose a license for the weights that is compatible with the training data's terms, and decide between a public and a gated repo.
  - Publish the base model and the chat model (DPO on main, SFT on a branch, tag v1.0) as safetensors with a complete model card.
  - Prove the release works by loading it with a transformers pipeline in a fresh environment with an empty cache and no token.
volatility: implementation
sources:
  - title: "Hugging Face Hub storage limits (free, PRO, private pay-as-you-go, file size limits)"
    url: https://huggingface.co/docs/hub/storage-limits
  - title: "Hugging Face pricing (PRO $9/month, storage per TB)"
    url: https://huggingface.co/pricing
  - title: "Hugging Face model cards (metadata, license, library_name, eval results)"
    url: https://huggingface.co/docs/hub/model-cards
  - title: "Displaying carbon emissions for your model (co2_eq_emissions)"
    url: https://huggingface.co/docs/hub/model-cards-co2
  - title: "Gated models"
    url: https://huggingface.co/docs/hub/models-gated
  - title: "huggingface_hub CLI guide (hf repos create/settings/branch/tag, hf upload)"
    url: https://huggingface.co/docs/huggingface_hub/guides/cli
  - title: "GitHub: about releases (2 GiB per file, 1,000 assets, no bandwidth limit)"
    url: https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases
  - title: "Cloudflare R2 public buckets (r2.dev is rate-limited, for development)"
    url: https://developers.cloudflare.com/r2/buckets/public-buckets/
  - title: "FineWeb-Edu dataset card (ODC-By)"
    url: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
  - title: "SmolLM2-360M model card (Apache-2.0, trained on FineWeb-Edu)"
    url: https://huggingface.co/HuggingFaceTB/SmolLM2-360M
last_verified: "2026-11-13"
---

# 11.1 · Publishing the model on Hugging Face

Until now your model has lived in a private bucket that only your scripts can read. Publishing it means three things at once: putting the weights where anyone can load them with one line of `transformers`, choosing a license that respects the terms of the data you trained on, and writing the model card that tells people what the model is, what it was trained on, how well it does and where it fails. This lesson does all three for your base model and your chat model, and ends with the only test that counts: a stranger's machine, with nothing cached and no token, loading your model and generating text.

## Goal

Two public model repos on the Hugging Face Hub: `<you>/my-gpt-350m-base` (your pretrained model) and `<you>/my-gpt-350m` (the DPO chat model on `main`, the SFT model on the branch `sft`, both frozen by the tag `v1.0`). Each holds only `safetensors` weights, config, tokenizer and a model card with training data and attribution, architecture, compute, cost and CO2 estimate, the evaluation tables from Module 8, the win rate from Module 10, limitations and intended use. A fresh virtual environment loads both with `pipeline("text-generation", ...)` without a token.

## What you need

- Your three exported models in the bucket: `r2:mygpt/hf/my-gpt-350m-base/` ([07.3](../module-07/lesson-03.md)), `hf/my-gpt-350m-sft/` ([09.2](../module-09/lesson-02.md)) and the DPO model you chose in [10.2](../module-10/lesson-02.md) (`hf/my-gpt-350m-dpo/` or `-dpo-b01/`).
- Your reports in git: `DATA_CARD.md` (02.3), `TOKEN_REPORT.md` (03.3), `EVAL_REPORT.md` (08.3), `reports/alignment_results.md` and `reports/winrate.json` (10.2), `budget.md`.
- A Hugging Face **write** token in your password manager ([01.2](../module-01/lesson-02.md)); about 4 GB of free disk on your laptop.
- The [Module 11 lab files](../../labs/module-11/) in your repo's `tools/` (and `MODEL_CARD.template.md` next to them).
- About 2.5 hours, all on your laptop. **Expected cost: $0** (R2 egress is free, public Hub storage is free).

## Choices and costs

### Where the weights live (checked 2026-11-13)

Your release is about 1.35 GB per model (fp32 safetensors, 337M parameters), ~4 GB for all three.

| Option | Price and limits | Loads with `from_pretrained("you/name")`? | Verdict |
|---|---|---|---|
| **Hugging Face Hub** | Public repos free ("best-effort", be reasonable beyond a few GB); private storage 100 GB free, PRO ($9/month) 1 TB, then $18/TB-month; files up to 500 GB (recommended < 200 GB); egress included | Yes, plus model page, eval widget, Spaces, `base_model` lineage, download counts | **Use it.** It is where people look for open models and the only option every tool in this course understands |
| ModelScope (modelscope.ai / .cn) | Free model hosting; current storage limits not published on a page we could verify | Through the `modelscope` library, not `transformers` directly | Optional mirror if you want reach in China; not needed |
| Kaggle Models | Free; quota pages are not readable without an account, so limits are not verified here | Through `kagglehub`, then `from_pretrained(local_path)` | Optional mirror for Kaggle notebook users |
| GitHub Releases | Free; each file under 2 GiB, up to 1,000 assets per release, no bandwidth limit | No; download the files first | Good **backup** next to your code (fp32 1.35 GB fits; bf16 0.67 GB fits easily) |
| Cloudflare R2 public bucket | Storage $0.015/GB-month (~$0.06/month for 4 GB), zero egress; the `r2.dev` URL is rate-limited and meant for development, production needs your own domain on Cloudflare | No | Keep your bucket **private** as the master copy; do not use it as the public download |

> [!NOTE]
> Storage terms change. Re-check the Hub's storage-limits page and pricing on the day you publish, and record "Hub: $0 (public)" in `budget.md` only if your release is still within the free public tier.

### One repo with revisions, or one repo per model

| Layout | For | Against |
|---|---|---|
| One repo per model (`-base`, `-sft`, `-dpo`) | Simple; each has its own card and download count | Three cards to keep in sync; visitors land on the wrong one |
| One repo, everything on branches | One page | A base model and a chat model are different products with different usage; `base_model` lineage cannot express it |
| **Base repo + chat repo (DPO on `main`, SFT on branch `sft`), tag `v1.0`** | The page people find is the chat model; SFT stays available with `revision="sft"`; `base_model` links the two; the tag pins exactly what your report evaluated | Branches are less visible; mention them in the card |

Use the third layout. A tag is cheap and is what makes your technical report reproducible: "evaluated `you/my-gpt-350m@v1.0`" means the same bytes forever, while `main` can change.

### Which license for the weights

What you trained on decides what you can promise. FineWeb-Edu is ODC-By 1.0 (attribution when you share the data or a database derived from it) and is subject to the Common Crawl Terms of Use; smol-smoltalk is Apache-2.0; UltraFeedback is MIT; `no_robots` (CC BY-NC) was only used to evaluate, never trained on. None of these forbids publishing trained weights under a permissive license, which is exactly what Hugging Face did with SmolLM2-360M (Apache-2.0, trained on FineWeb-Edu among others).

| License | What it says | Fit |
|---|---|---|
| **Apache-2.0** | Permissive, explicit patent grant, keep the NOTICE/attribution | **Recommended.** The norm for small open models, same as SmolLM2 and the SFT data |
| MIT | Permissive, very short, no patent grant | Fine too; slightly weaker legal clarity than Apache-2.0 |
| CC-BY-4.0 | Attribution license designed for content, not software | Usable, but unusual for weights; tools filter on Apache/MIT |
| OpenRAIL-M | Permissive plus use restrictions that must be passed on | Not "open source" by OSI terms; extra burden for a research toy |
| CC-BY-NC-4.0 | No commercial use | Only if a training dataset required it; none of yours does |

This is a reasoned recommendation, not legal advice. Whatever you pick, the card states the data licenses and the FineWeb-Edu attribution separately, because your weights license does not change the data's terms.

### Public or gated

| Mode | Effect | Use when |
|---|---|---|
| **Public** | Anyone loads it, no account needed | **Your case**: a small research model with no special risk |
| Gated, automatic approval | Users must log in and share username and email with you; `transformers` then needs their token | You need to know who downloads (and you then handle their personal data) |
| Gated, manual approval | You approve each request | Early or risky releases |
| Private | Only you | The 07.3 copy, and the first upload below until the card is checked |

Gating a 350M educational model adds friction, makes the demo and other people's notebooks need tokens, and makes you a holder of strangers' emails. Publish it public, after a private dress rehearsal.

## Do this

### 1. Pull the three models and assemble clean release folders (20 min)

```bash
cd ~/mygpt && set -a && . ~/.config/mygpt/r2.env && set +a
for m in base sft dpo; do bash scripts/sync.sh pull-hf my-gpt-350m-$m hf/my-gpt-350m-$m; done   # use dpo-b01 if that won in 10.2
ls -la hf/my-gpt-350m-dpo
```

TRL's output folders contain things that must not be published: `training_args.bin` (a pickle), sometimes `checkpoint-*` folders, and an auto-generated `README.md`. Copy only what a user needs:

```bash
KEEP='config.json generation_config.json model.safetensors tokenizer.json tokenizer_config.json special_tokens_map.json chat_template.jinja mygpt_export.json'
for pair in "base:my-gpt-350m-base" "dpo:my-gpt-350m" "sft:my-gpt-350m-sft"; do
  src=hf/my-gpt-350m-${pair%%:*}; dst=release/${pair##*:}; mkdir -p $dst
  for f in $KEEP; do [ -f $src/$f ] && cp $src/$f $dst/; done
done
ls -la release/*
echo "release/" >> .gitignore
```

Each folder should hold `model.safetensors` (~1.35 GB), `config.json`, `generation_config.json` and the tokenizer files; the chat folders also `chat_template.jinja` (or the template inside `tokenizer_config.json`).

### 2. Write the model cards (60 min)

```bash
cp tools/MODEL_CARD.template.md release/my-gpt-350m/README.md
```

Fill every `<...>` and `____` from your own files, never from memory:

| Card section | Copy from |
|---|---|
| Architecture, params | `release/my-gpt-350m-base/mygpt_export.json`, `configs/main_350m.yaml` |
| Training data, revisions, attribution | `DATA_CARD.md`, `manifests/*.json`, `reports/sft_data.md`, `reports/dpo_runs.md` |
| Decontamination | `TOKEN_REPORT.md`, `reports/decontam-*.json` |
| Evaluation table | `EVAL_REPORT.md` (08.3); regression rows from `evals/regression.md` (10.2) |
| Win rate | `reports/winrate.json`, `reports/alignment_results.md` |
| Limitations | `reports/sft_probe.md`, `reports/dpo_probe.md`: paste two real failures |
| Compute and cost | `budget.md`, `reports/main-run-log.md` |

The CO2 line is an ESTIMATE; show the arithmetic. Example with the main run alone: 8 GPUs × 5.7 h = 45.6 H100-hours × 0.7 kW (H100 SXM TDP) × 1.2 (data-centre PUE) ≈ 38 kWh; at 0.4 kg CO2e/kWh (look up your provider region's grid on Electricity Maps) ≈ 15 kg CO2e, i.e. 15,000 g in the `co2_eq_emissions.emissions` field, which is in grams. Do the same for all GPU-hours of the project in the Compute section.

Then make two short variants: for `release/my-gpt-350m-base/README.md` remove `base_model`, the SFT/DPO rows and the chat usage (show plain completion instead: `pipeline("text-generation", model="<you>/my-gpt-350m-base")("The history of the printing press")`), and set `training_type: pre-training`. For `release/my-gpt-350m-sft/README.md` copy the chat card and change the first lines to say it is the SFT revision.

```bash
for d in release/*/; do python tools/check_release.py $d --load; done
```

Every folder must end with `RELEASE OK`. Warnings about placeholders mean you missed a field.

> [!CAUTION]
> Everything you upload to a public repo is copied, cached and mirrored within minutes; deleting it later does not un-publish it. Before the public step, grep the release for secrets (`grep -rIl "hf_\|sk-\|AKIA\|RCLONE_CONFIG" release/` must print nothing) and make sure no file contains your bucket credentials, API keys or personal data.

### 3. Upload privately first (20 min, mostly upload time)

```bash
hf auth login                       # paste the WRITE token; do not store it in the repo
hf auth whoami                      # prints your username
U=<your username>

hf repos create $U/my-gpt-350m --private --exist-ok
hf upload $U/my-gpt-350m release/my-gpt-350m . --commit-message "v1.0: DPO chat model + model card"
hf upload $U/my-gpt-350m release/my-gpt-350m-sft . --revision sft --commit-message "v1.0: SFT model"

hf repos create $U/my-gpt-350m-base --private --exist-ok    # exists already if you made the 07.3 copy
hf upload $U/my-gpt-350m-base release/my-gpt-350m-base . --commit-message "v1.0: base model + model card"
```

`--revision sft` creates the branch `sft` from `main` and then replaces its files with the SFT model's. Open both repos in the browser while logged in: the card renders, the metadata shows license, datasets and the eval widget, and the **Files** tab shows only the files you meant to publish.

> [!WARNING]
> `hf upload` to a repo that does not exist creates it with the default visibility, which is **public** for a personal account. Always create the repo with `--private` first, as above. (`export_hf --push` creates its repo private by default and prints the command that makes it public.)

### 4. Make them public and tag v1.0 (5 min)

```bash
hf repos settings $U/my-gpt-350m --public
hf repos settings $U/my-gpt-350m-base --public
hf repos tag create $U/my-gpt-350m v1.0
hf repos tag create $U/my-gpt-350m v1.0-sft --revision sft
hf repos tag create $U/my-gpt-350m-base v1.0
git tag release-v1.0 && git push --tags         # the code commit your card links to
```

### 5. Verify from a clean environment (20 min)

A test on your laptop's normal environment proves little: the cache already has your files and your token is logged in. Use a new venv, an empty cache and no token:

```bash
cd "$(mktemp -d)"
python3 -m venv venv && . venv/bin/activate          # Windows Git Bash: venv/Scripts/activate
pip install -q torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -q transformers==5.22.2
export HF_HOME="$PWD/hf-cache"; unset HF_TOKEN
python - <<'EOF'
from transformers import pipeline
U = "<you>"
chat = pipeline("text-generation", model=f"{U}/my-gpt-350m", revision="v1.0")
msgs = [{"role": "user", "content": "Explain photosynthesis in two sentences."}]
print("DPO:", chat(msgs, max_new_tokens=80, do_sample=False)[0]["generated_text"][-1]["content"])
sft = pipeline("text-generation", model=f"{U}/my-gpt-350m", revision="sft")
print("SFT:", sft(msgs, max_new_tokens=80, do_sample=False)[0]["generated_text"][-1]["content"])
base = pipeline("text-generation", model=f"{U}/my-gpt-350m-base")
print("BASE:", base("The three states of matter are", max_new_tokens=40, do_sample=False)[0]["generated_text"])
EOF
```

Compare the three answers with what `scripts/chat.py` printed for the same prompts in 09.2 and 10.1 (greedy decoding gives the same text). Then delete the temporary folder.

### 6. Optional backup on GitHub Releases (10 min)

```bash
gh release create v1.0 --title "my-gpt-350m v1.0" --notes "Weights: https://huggingface.co/$U/my-gpt-350m (tag v1.0)" \
  release/my-gpt-350m/model.safetensors#dpo-model.safetensors
```

Only the weights file needs to be there; the Hub stays the place people load from.

## How you know it worked

- `https://huggingface.co/<you>/my-gpt-350m` loads without logging in, shows **Apache-2.0** (or your choice), `transformers`, `text-generation`, the three datasets, an evaluation widget with your HellaSwag number, and a "Finetuned from `<you>/my-gpt-350m-base`" link.
- The **Files** tab of both repos lists `model.safetensors` (~1.35 GB) and no `*.bin`, `*.pt`, `training_args.bin` or `checkpoint-*`; the branch selector shows `main` and `sft`; tags `v1.0` and `v1.0-sft` exist.
- The clean-environment script prints three answers with no token and an empty cache, and the greedy outputs match your earlier `chat.py` runs.
- `check_release.py` printed `RELEASE OK` for all three folders, and the card's numbers match `reports/` exactly.

## When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `check_release.py`: `training_args.bin: pickle-based file` | You uploaded TRL's output folder as is | Rebuild the release folder with the `KEEP` list in step 1 |
| The Hub shows a TRL-generated card ("This model is a fine-tuned version of...") | TRL wrote `README.md` into the output folder and it was uploaded | Replace it with your card and upload `README.md` again |
| `401` / `Repository Not Found` in the clean env | Repo still private, or you typed the username wrong | `hf repos settings ... --public`; check the URL in a logged-out browser window |
| `OSError: ... does not appear to have a file named config.json` for `revision="sft"` | The `sft` upload went to `main` or failed | `hf repos branch create $U/my-gpt-350m sft` then rerun the `--revision sft` upload |
| Pipeline returns the prompt plus rambling text for the chat model | You passed a string instead of a messages list, so no chat template was applied | Pass `[{"role": "user", "content": ...}]`; check `chat_template.jinja` was uploaded |
| Answers in the clean env differ from `chat.py` | Sampling on, or a different revision or dtype | Use `do_sample=False`, the `v1.0` tag, and fp32 on CPU |
| Eval widget does not appear | `model-index` YAML invalid or numbers left as `____` | Fix the YAML (the Hub shows a metadata error banner on the card) |
| You published something you should not have (a key, personal data) | Missed the grep | Revoke the key **first**, then delete the file; if it was in git history, `hf repos delete` and re-create, because old commits stay downloadable |

## Artifact

- Public Hub repos `<you>/my-gpt-350m` (DPO on `main`, SFT on `sft`, tags `v1.0`, `v1.0-sft`) and `<you>/my-gpt-350m-base` (tag `v1.0`), each with a complete model card.
- In git: the three model cards copied to `release_cards/` (`cp release/*/README.md` into `release_cards/<name>.md`), the git tag `release-v1.0`, and the clean-env output pasted into `reports/release_check.md`.
- `budget.md` line:

  ```text
  | 2026-11-13 | 11.1 | HF Hub public repos (base + chat, ~4 GB), R2 egress to laptop | $0.00 | $0.00 |
  ```
