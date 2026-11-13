---
id: "03.1"
module: 3
minutes: 20
practice_minutes: 60
prerequisites: ["01.3", "02.3"]
objectives:
  - Justify a 32,768-token vocabulary for a ~350M model with parameter, compute and memory arithmetic against 50,257 and 65,536.
  - Train a byte-level BPE tokenizer on a held-back FineWeb-Edu file on a rented CPU box.
  - Measure bytes per token of your tokenizer and GPT-2's on held-out documents and project the main-run token count.
  - Explain which special tokens the tokenizer has, how the chat format works without new ones, and why literal special strings in web text must stay ordinary text.
volatility: implementation
sources:
  - title: "Hugging Face tokenizers: trainers (BpeTrainer)"
    url: https://huggingface.co/docs/tokenizers/api/trainers
  - title: "tokenizers 0.23.1 release notes"
    url: https://github.com/huggingface/tokenizers/releases
  - title: "Language Models are Unsupervised Multitask Learners (GPT-2, 50,257-token BPE)"
    url: https://cdn.openai.com/better-language-models/language-models.pdf
  - title: "nanochat: tokenizer trained on ~2B characters of FineWeb-Edu"
    url: https://github.com/karpathy/nanochat/discussions/1
  - title: "FineWeb-Edu dataset card"
    url: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
  - title: "Hetzner Cloud regular-performance (CPX) prices"
    url: https://www.hetzner.com/cloud/regular-performance/
  - title: "Hetzner Cloud general-purpose (CCX) prices"
    url: https://www.hetzner.com/cloud/general-purpose/
last_verified: "2026-11-13"
---

# 03.1 · Training your own BPE tokenizer

Your baseline uses GPT-2's tokenizer so it can be compared with OpenAI's model, but your main model gets its own: a 32,768-token byte-level BPE trained on the same FineWeb-Edu text it will learn from. A smaller vocabulary than GPT-2's frees parameters and compute for the transformer blocks, and training it in-domain keeps compression close to GPT-2's. In this lesson you rent the CPU box that will carry all of Module 3, train the tokenizer in about a quarter of an hour, and measure it against GPT-2 before you commit 22 billion tokens to it.

## Goal

You have `tokenizers/bpe32k/tokenizer.json` (32,768 tokens, `<|endoftext|>` = id 0) trained on half a million FineWeb-Edu documents that are neither in your validation split nor otherwise special, a comparison against GPT-2 on 20,000 held-out documents (bytes per token, projected token count of the main-run subset), the tokenizer in your bucket under `tokenizer/` and in git, and a running CPU box ready for 03.2.

## What you need

- Your project repo from [01.3](../module-01/lesson-03.md) and the raw FineWeb-Edu files in `r2:mygpt/raw/fineweb-edu/` from [02.3](../module-02/lesson-03.md), with `r2.env` from [02.2](../module-02/lesson-02.md).
- The [Module 3 lab files](../../labs/module-03/), copied into `tools/` in your repo and committed.
- A Hetzner Cloud account (or RunPod credit for a CPU pod, see 01.2) and your SSH key.
- About 80 minutes, of which ~35 on the box.
- **Expected cost: ~$0.15** for this lesson (under an hour of a CPX62). The same box continues in 03.2 and 03.3; Module 3 in total costs about $1–3.

## Choices and costs

### Vocabulary size: why 32,768

Your main model is `configs/main_350m.yaml`: $d = 1024$, 24 layers, context 1024, tied input/output embeddings. Its transformer blocks hold $N_{\text{blocks}} = 24 \times (4 \cdot 1024^2 + 3 \cdot 1024 \cdot 2752) \approx 303.6\text{M}$ parameters regardless of vocabulary. The vocabulary adds $V \cdot d$ embedding parameters and, per training token, about $6 V d$ FLOPs for the output layer, next to $6 N_{\text{blocks}} + 6 \cdot n_{\text{layer}} \cdot T \cdot d \approx 1.97$ GFLOPs for the rest.

| Vocabulary $V$ | Embedding params | Total params | Output-layer share of training FLOPs per token | fp32 logits per micro-batch (32 × 1024 tokens) | Tokens for the same text (estimate) | Compute per byte of text vs 32k |
|---|---|---|---|---|---|---|
| 16,384 | 16.8M | 320M | 4.9% | 2.1 GB | ~+8% | ~+3% |
| **32,768** | **33.6M** | **337M** | **9.3%** | **4.3 GB** | **reference** | **reference** |
| 50,304 (GPT-2's 50,257 padded to a multiple of 64) | 51.5M | 355M | 13.6% | 6.6 GB | ~−4% | ~+1% |
| 65,536 | 67.1M | 371M | 17.0% | 8.6 GB | ~−6% | ~+3% |

How to read it:

- **Compute per byte is flat around 32k–50k.** A bigger vocabulary needs fewer tokens for the same text, but each token costs more in the output layer. The token-count column is an ESTIMATE (a rule of thumb for in-domain BPE: doubling the vocabulary saves a few percent of tokens); you measure the real number for 32k vs GPT-2 below. The differences are a few percent either way: 32k is a sound choice, not a magic one.
- **Parameters go where they help.** At 350M, 33.6M embedding parameters are 10% of the model; 65k would make it 18%. Embedding rows of rare tokens get few updates in 20B tokens; block parameters are used by every token.
- **Memory.** The logits tensor for one micro-batch of 32 × 1024 tokens is $32 \cdot 1024 \cdot V \cdot 4$ bytes in fp32 for the loss: 4.3 GB at 32k, 8.6 GB at 65k, on top of activations.
- **Hard limits.** Shards store tokens as uint16, so $V \le 65{,}536$. 32,768 = $2^{15}$ is a multiple of 64 and 128, so no padding is needed for efficient matmuls.
- **Precedent.** Small open models use 32k–50k (Llama 2: 32,000; SmolLM2: 49,152; nanochat defaults to 32,768).

### Training sample and special tokens

| Decision | Choice | Why |
|---|---|---|
| Text to train on | First 500,000 documents of `sample/100BT/002_00009.parquet` (the 30th main-run file), ~2.3 GB | Same distribution as training. Not file `000_00000`, whose first documents become the validation split in 03.2, so your compression measurement stays honest. nanochat trains its tokenizer on ~2B characters of FineWeb-Edu; more text barely changes the merges at 32k |
| Algorithm | Byte-level BPE, GPT-2 pre-tokenizer regex (`scripts/train_tokenizer.py`, Hugging Face `tokenizers` 0.23) | Lossless: any UTF-8 byte sequence round-trips; no unknown tokens |
| Special tokens | Only `<\|endoftext\|>` (id 0), added by `train_tokenizer.py` | It separates documents in the shards and ends every answer in SFT |
| Chat format for Module 9 | Plain-text tags `<\|user\|>` and `<\|assistant\|>`, written by `export_hf.py` into the chat template; the answer ends with `<\|endoftext\|>` | No new tokens, so the vocabulary and embeddings never change after pretraining. The tags cost a few ordinary tokens each, which is irrelevant at this scale |

One special-token rule matters already now: the string `<|endoftext|>` also appears **as text** on some web pages (tutorials about GPT, for example). In your pretraining data it must stay ordinary text. If the tokenizer turned it into the real id 0, the page would be cut into two fake documents. `compare_tokenizers.py` checks this.

### Where to run it (prices checked 2026-11-13)

| Box | Price | RAM / cores / disk | Fit for Module 3 |
|---|---|---|---|
| **Hetzner CPX62** (shared AMD vCPU), Helsinki | €0.1779/h excl. VAT (~$0.21/h) | 32 GB, 16 vCPU, 640 GB NVMe, 20 TB traffic | **Recommended.** Enough RAM for 500k documents, enough disk for raw data plus all shards. Same box as 02.3 |
| Hetzner CCX43 / CCX53 (dedicated vCPU) | €0.3766 / €0.7267 per hour | 64 GB / 16 cores, 128 GB / 32 cores | Faster tokenizing in 03.2 at about the same total cost; showed "currently unavailable" on 2026-11-13 |
| RunPod CPU pod | roughly $0.03–0.04 per vCPU-hour (late-2026 estimate; the console shows the live price) | choose ≥ 32 GB RAM and ≥ 250 GB container disk | Fallback if Hetzner will not create servers for you |
| Your laptop | $0 | needs ≥ 16 GB free RAM | Possible for this lesson only; 03.2 needs the box anyway |

> [!NOTE]
> Hetzner changed prices three times in 2026 and has limited server creation for new customers since June. Check the live price and availability for your location before creating the server, and put the price you see in `budget.md`.

> [!WARNING]
> From now until the end of 03.3 a server bills by the hour, also while you read or sleep. If you stop for the day, upload your results and **delete** the server (`hcloud server delete mygpt-cpu`); powering it off does not stop billing. Re-creating it costs ten minutes.

## Do this

### 1. Create the box and set it up (15 min)

```bash
hcloud server create --name mygpt-cpu --type cpx62 --image ubuntu-24.04 --location hel1 --ssh-key my-laptop
scp ~/.config/mygpt/r2.env root@<IP>:/root/r2.env
ssh root@<IP>
```

On the box (same setup as 02.3):

```bash
apt-get update && apt-get install -y git tmux htop
curl https://rclone.org/install.sh | bash
curl -LsSf https://astral.sh/uv/install.sh | sh && . ~/.local/bin/env
tmux new -s tok
git clone https://github.com/<you>/mygpt.git && cd mygpt
uv venv --python 3.12 && . .venv/bin/activate
uv pip install -e . --torch-backend cpu
chmod 600 /root/r2.env && set -a && . /root/r2.env && set +a
export TOKENIZERS_PARALLELISM=false
rclone lsf r2:mygpt/raw/fineweb-edu/sample-100BT/ | head -3
```

### 2. Pull the two files this lesson needs (1 min)

```bash
mkdir -p data/raw/sample-100BT
rclone copy r2:mygpt/raw/fineweb-edu/sample-100BT/ data/raw/sample-100BT/ \
  --include "002_00009.parquet" --include "000_00000.parquet" --multi-thread-streams 8 --progress
ls -lh data/raw/sample-100BT/
```

### 3. Train the tokenizer (10–30 min)

```bash
/usr/bin/time -v python scripts/train_tokenizer.py \
  --input data/raw/sample-100BT/002_00009.parquet \
  --vocab-size 32768 --max-docs 500000 --out tokenizers/bpe32k 2>&1 | tee train_tokenizer.log
```

Watch memory in a second tmux window (`Ctrl-b c`, then `htop`). `--input` points at one file, so only that file is read; `--max-docs` stops after 500,000 documents.

### 4. Compare with GPT-2 on held-out documents (3 min)

```bash
mkdir -p reports
python tools/compare_tokenizers.py --tokenizer tokenizers/bpe32k \
  --input data/raw/sample-100BT/000_00000.parquet --docs 20000 \
  --gpt2-tokens-subset 22.54e9 --out reports/compare_tokenizers.json
```

`--gpt2-tokens-subset` is the GPT-2 token estimate of your 30-file subset from the `sample-100BT-30` manifest in 02.3; the script converts it into the token count your tokenizer will produce.

### 5. Save the tokenizer (5 min)

```bash
sha256sum tokenizers/bpe32k/tokenizer.json | tee tokenizers/bpe32k/tokenizer.json.sha256
bash scripts/sync.sh push-tokenizer tokenizers/bpe32k        # -> r2:mygpt/tokenizer/
rclone lsf r2:mygpt/tokenizer/
cp train_tokenizer.log reports/
git add -f tokenizers/bpe32k/tokenizer.json tokenizers/bpe32k/tokenizer_meta.json tokenizers/bpe32k/tokenizer.json.sha256
git add reports/compare_tokenizers.json reports/train_tokenizer.log
```

Commit from the box only if it has GitHub credentials you are happy to leave there for a few hours; otherwise `scp` the files to your laptop and commit there (see 04.2). Start `TOKEN_REPORT.md` from [`TOKEN_REPORT.template.md`](../../labs/module-03/TOKEN_REPORT.template.md) and fill in the **Tokenizer** section now.

Keep the box running if you go straight on to 03.2; otherwise delete it.

## How you know it worked

- `train_tokenizer.py` ends with a line like `wrote tokenizers/bpe32k/tokenizer.json | vocab 32768 | eot_id 0 | 900s` and a sample round-trip line. The vocabulary is exactly **32768**, `eot_id` is **0**.
- `/usr/bin/time -v` reports the peak memory as "Maximum resident set size"; it should stay well below the box's 32 GB. Write it in `TOKEN_REPORT.md`.
- `compare_tokenizers.py` shows GPT-2 at about **4.5–4.6 bytes per token** on FineWeb-Edu (lesson 02.1 measured ~4.57) and yours within about **±5%** of it (ESTIMATE: a 32k in-domain BPE roughly matches GPT-2's 50k general one on English educational text). The projected subset size should land around **21–24B tokens**, enough for the 20B-token main run.
- The example lines round-trip, the special-token section lists `<|endoftext|>` at id 0, the literal-text check prints `OK`, and the script ends with `ALL CHECKS PASSED`.
- `rclone lsf r2:mygpt/tokenizer/` lists `tokenizer.json`, `tokenizer_meta.json` and the `.sha256` file.

## When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| Process killed, `dmesg` shows `Out of memory` | Too many documents for 32 GB RAM | Rerun with `--max-docs 250000`; compression changes by well under 1% |
| Training seems stuck at "Count pairs" for many minutes | Normal for 2+ GB of text on 16 shared vCPUs | Wait; check CPU use in `htop`. Only worry if nothing uses CPU for 10 minutes |
| Vocabulary smaller than 32,768 | Too little text (wrong file, `--max-docs` tiny) so merges ran out of frequent pairs | Check `--input` points at a full 2.15 GB file; rerun |
| Your tokenizer needs >10% more tokens than GPT-2 | Trained on the wrong text (for example a broken or non-English file), or `--vocab-size` typo | Re-read the command; check a few documents with `python -c "import pyarrow.parquet as pq; print(pq.read_table('data/raw/sample-100BT/002_00009.parquet', columns=['text']).slice(0,1)[0][0])"` |
| `FAIL literal '<\|endoftext\|>' ... encoded as the real EOT id` | The project's tokenizer wrapper still treats special-token strings inside text as special | Do not tokenize the main set yet. Update your repo from the course lab code (`labs/common/mygpt/tokenizer.py`), reinstall, rerun the check |
| `push-tokenizer` puts files somewhere unexpected | Wrong local path (for example `tokenizer/` instead of `tokenizers/bpe32k`) | `rclone lsf r2:mygpt/ --max-depth 2`, delete strays with `rclone deletefile`, rerun with the right path |
| `hcloud server create` fails with a resource or limit error | Type unavailable in that location, or new-customer limits | Try `--location fsn1` or `nbg1`, or `cpx52`; else use a RunPod CPU pod (01.2) |

## Artifact

- `tokenizers/bpe32k/tokenizer.json` (+ meta and sha256) in git and in `r2:mygpt/tokenizer/`; `reports/compare_tokenizers.json` and `reports/train_tokenizer.log` in git; the Tokenizer section of `TOKEN_REPORT.md` filled in.
- `budget.md` line (use your real time and the price you saw):

  ```text
  | 2026-11-13 | 03.1 | Hetzner CPX62 hel1, ~0.75 h at EUR 0.1779/h, BPE tokenizer training | ~$0.15 | $____ |
  ```
