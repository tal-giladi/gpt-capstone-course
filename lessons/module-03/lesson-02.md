---
id: "03.2"
module: 3
minutes: 15
practice_minutes: 240
prerequisites: ["02.2", "02.3", "03.1"]
objectives:
  - Estimate tokenization hours and cost from measured tokens per second per core, and pick a CPU box accordingly.
  - Tokenize FineWeb-Edu sample/10BT with GPT-2's tokenizer and 30 files of sample/100BT with your BPE into uint16 shards.
  - Verify every shard set against its meta.json before trusting it.
  - Upload both shard sets to the bucket with a checked copy, at the paths the training configs expect.
volatility: implementation
sources:
  - title: "Hetzner Cloud regular-performance (CPX) prices"
    url: https://www.hetzner.com/cloud/regular-performance/
  - title: "Hetzner Cloud general-purpose (CCX) prices"
    url: https://www.hetzner.com/cloud/general-purpose/
  - title: "DigitalOcean Droplet pricing (CPU-Optimized)"
    url: https://www.digitalocean.com/pricing/droplets
  - title: "RunPod pod pricing and billing"
    url: https://docs.runpod.io/pods/pricing
  - title: "tiktoken (GPT-2 encoding)"
    url: https://github.com/openai/tiktoken
  - title: "Hugging Face tokenizers documentation"
    url: https://huggingface.co/docs/tokenizers/api/trainers
  - title: "rclone S3 backend (multipart tuning)"
    url: https://rclone.org/s3/
last_verified: "2026-11-13"
---

# 03.2 · Tokenizing 20B tokens on a rented CPU box

Training reads tokens, not text: every training step pulls a window of token ids from a flat binary file, and nothing on the GPU box should ever tokenize. So before any GPU is rented, you turn ~93 GB of parquet into two shard sets: about 10 billion GPT-2 tokens for the baseline and about 22 billion tokens of your own tokenizer for the ablations and the main run. It is a few hours of pure CPU work that costs less than a dollar, as long as you measure the speed first and delete the box at the end.

## Goal

Your bucket holds `tok/gpt2-10BT/` (~10.0B tokens, ~20 GB) and `tok/bpe32k-100BT-30/` (~21–24B tokens, ~45 GB): uint16 shards of 100M tokens, a ~50M-token validation split each, `meta.json`, and for your tokenizer a copy of `tokenizer.json`. Both sets passed `shard_check.py`, both uploads passed `rclone check`, and you have measured tokens per second for both tokenizers on your box.

## What you need

- The CPU box and `tokenizers/bpe32k/` from [03.1](lesson-01.md) (or recreate the box and pull the tokenizer: `bash scripts/sync.sh pull-tokenizer tokenizers/bpe32k`).
- The raw files in `r2:mygpt/raw/fineweb-edu/` from [02.3](../module-02/lesson-03.md).
- About 4 hours, of which you are needed for ~45 minutes; the rest runs in tmux.
- **Expected cost: about $0.50–1.00** (3–4 h of a Hetzner CPX62 at €0.1779/h; ~$2–3 on a RunPod CPU pod or DigitalOcean droplet).

## Choices and costs

### How long will it take? Measure, then multiply

Tokenization is embarrassingly parallel: `tokenize_shards.py` hands batches of 256 documents to `--workers` processes and writes shards in order. Time ≈ tokens ÷ (tokens per second per core × cores). Measured single-core rates on English text on 2026-11-13 (a laptop core; a shared cloud vCPU is often 30–50% slower):

| Tokenizer | Library | Tokens/s per core (laptop) |
|---|---|---|
| GPT-2 (50,257) | `tiktoken` 0.14 | ~1.4M |
| Your BPE (32,768) | Hugging Face `tokenizers` 0.23 | ~0.25M |

From those, ESTIMATES for both sets (10B GPT-2 + ~22.5B own tokens), assuming ~0.15M own tokens/s per shared vCPU and ~0.22M per dedicated core, workers = cores − 1:

| Box (prices checked 2026-11-13, excl. VAT) | Price/h | Own BPE, 22.5B | GPT-2, 10B | Total box time incl. download/upload | Cost |
|---|---|---|---|---|---|
| **Hetzner CPX62**, 16 shared vCPU, 32 GB, 640 GB | €0.1779 | ~2.8 h | ~0.3 h | ~3.5 h | **~€0.62 (~$0.72)** |
| Hetzner CCX43, 16 dedicated, 64 GB, 360 GB | €0.3766 | ~1.9 h | ~0.2 h | ~2.5 h | ~€0.94 |
| Hetzner CCX63, 48 dedicated, 192 GB, 960 GB | €1.1626 | ~0.6 h | ~0.1 h | ~1.1 h | ~€1.28 |
| DigitalOcean CPU-Optimized, 32 vCPU / 64 GiB, 400 GiB | $1.00 | ~1.4 h | ~0.15 h | ~2 h | ~$2 |
| RunPod CPU pod, 32 vCPU, ≥ 200 GB disk | ~$1.0–1.3 (estimate, check the console) | ~1.4 h | ~0.15 h | ~2 h | ~$2–2.60 |

The work is a fixed number of core-hours, so what matters is the price per core-hour: more cores at the same per-core price cost the same and finish sooner. Dedicated cores cost about twice as much per core-hour as shared ones but run faster, so the bigger boxes end up somewhat more expensive in total. Every option stays under about $3; pick by availability and how long you want to wait. The CPX62 from 03.1 is the default; CCX types showed "currently unavailable" on 2026-11-13.

> [!NOTE]
> These rates are estimates from a laptop measurement. Step 2 measures your box in two minutes; use that number, not this table, for the ledger estimate. Re-check the live price before creating a new box.

### What gets written

| Shard set | Input | Tokenizer | Expected size | Bucket path = local path |
|---|---|---|---|---|
| `gpt2-10BT` | `raw/fineweb-edu/sample-10BT/` (14 files) | `gpt2` (tiktoken) | ~10.0B tokens, ~100 train shards, ~20 GB | `tok/gpt2-10BT/` = `data/tok/gpt2-10BT/` |
| `bpe32k-100BT-30` | `raw/fineweb-edu/sample-100BT/` (30 files) | `tokenizers/bpe32k` | ~21–24B tokens, ~220 train shards, ~45 GB | `tok/bpe32k-100BT-30/` = `data/tok/bpe32k-100BT-30/` |

Format (fixed by the project code): each `train_000123.bin` is a 1 KB header (magic 20261113, version, token count, vocab size) followed by up to 100M uint16 token ids, 200 MB per full shard. Every document is written as `[EOT] + tokens`. The first ~50M tokens of each input (whole documents) go to `val_*.bin`, everything after to `train_*.bin`, so the validation split never changes if you later add data. `configs/baseline_124m.yaml` reads `data/tok/gpt2-10BT`; `ablation_base.yaml` and `main_350m.yaml` read `data/tok/bpe32k-100BT-30`.

Disk on the box: 93 GB raw + 20 GB + 45 GB shards ≈ 160 GB, well inside the CPX62's 640 GB.

> [!WARNING]
> The box bills until it is deleted. Tokenization runs unattended for hours: start it in tmux, set a phone timer for the expected end, and do not leave the box running overnight "in case". If you stop before 03.3, upload and delete it.

## Do this

### 1. Pull the raw data (5–10 min)

On the box from 03.1, inside tmux, with the venv active and `r2.env` loaded:

```bash
cd ~/mygpt && . .venv/bin/activate && set -a && . /root/r2.env && set +a
export TOKENIZERS_PARALLELISM=false
bash scripts/sync.sh pull-raw sample-10BT  data/raw/sample-10BT
bash scripts/sync.sh pull-raw sample-100BT data/raw/sample-100BT
ls data/raw/sample-10BT | wc -l; ls data/raw/sample-100BT | wc -l    # 14 and 30
df -h /                                                              # >= 200 GB free
```

### 2. Measure your box (5 min)

```bash
nproc
python scripts/tokenize_shards.py --input data/raw/sample-100BT --tokenizer tokenizers/bpe32k \
  --out /tmp/trial --workers $(( $(nproc) - 1 )) --max-tokens 300000000
rm -rf /tmp/trial
```

It prints a line every 50M tokens, for example `0.30B tokens | 2.4M tok/s`. Compute the expected time for the main set, then write the estimate in the ledger:

```bash
python -c "r=2.4e6; print(f'own BPE: {22.5e9/r/3600:.1f} h')"    # put your M tok/s here
```

### 3. Tokenize the baseline set with GPT-2 (15–30 min)

```bash
mkdir -p reports
python scripts/tokenize_shards.py --input data/raw/sample-10BT --tokenizer gpt2 \
  --out data/tok/gpt2-10BT --workers $(( $(nproc) - 1 )) 2>&1 | tee reports/tokenize-gpt2-10BT.log
python tools/shard_check.py --shards data/tok/gpt2-10BT
```

### 4. Tokenize the main set with your BPE (2–3 h on a CPX62)

Start it, then upload the finished GPT-2 set from a second tmux window while it runs:

```bash
python scripts/tokenize_shards.py --input data/raw/sample-100BT --tokenizer tokenizers/bpe32k \
  --out data/tok/bpe32k-100BT-30 --workers $(( $(nproc) - 1 )) 2>&1 | tee reports/tokenize-bpe32k-100BT-30.log
```

```bash
# second window: Ctrl-b c
cd ~/mygpt && . .venv/bin/activate && set -a && . /root/r2.env && set +a
bash scripts/sync.sh push-tok gpt2-10BT data/tok/gpt2-10BT
rclone check data/tok/gpt2-10BT r2:mygpt/tok/gpt2-10BT --one-way
```

`htop` should show all cores near 100% for the BPE run. Detach with `Ctrl-b d` and come back later with `tmux attach -t tok`.

### 5. Check and upload the main set (15 min)

```bash
python tools/shard_check.py --shards data/tok/bpe32k-100BT-30
bash scripts/sync.sh push-tok bpe32k-100BT-30 data/tok/bpe32k-100BT-30
rclone check data/tok/bpe32k-100BT-30 r2:mygpt/tok/bpe32k-100BT-30 --one-way
rclone size r2:mygpt/tok/
```

### 6. Record it

Copy the two `meta.json` files and the logs into `reports/` (for example `cp data/tok/gpt2-10BT/meta.json reports/meta-gpt2-10BT.json`), and fill in the **Shard sets** table of `TOKEN_REPORT.md` with train/val tokens, shard counts, sizes and bytes per token. Go straight on to 03.3 on this box, or delete it if you stop here (the shards are safe in the bucket).

## How you know it worked

- The GPT-2 run's final `meta.json` shows `"val_tokens"` just above 50,000,000, `"train_tokens"` around 9.95B and `"total_tokens"` about 10.0B (9.67M documents, so ~9.67M of those tokens are end-of-text). `total_bytes / total_tokens` is about 4.5–4.6.
- The BPE run's `total_tokens` is close to the projection `compare_tokenizers.py` gave you in 03.1 (within ~2%), and its bytes per token matches 03.1's held-out number within ~2%.
- Both `shard_check.py` runs end with `OK`: token totals match `meta.json`, the end-of-text count equals the document count, all ids are below the vocabulary size.
- `ls data/tok/gpt2-10BT` shows `val_000000.bin`, `train_000000.bin` … about `train_000099.bin` and `meta.json`; `data/tok/bpe32k-100BT-30` also contains `tokenizer.json`. Full shards are 200,001,024 bytes (1,024-byte header + 100M × 2 bytes); only the last train shard and the val shard are smaller.
- Both `rclone check` runs report `0 differences found`, and `rclone size r2:mygpt/tok/` reports roughly 65 GB.

## When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| BPE run far slower than the trial, `htop` shows one busy core | `--workers` missing or 1 | Stop (`Ctrl-c`), delete the output folder, rerun with `--workers $(( $(nproc) - 1 ))` |
| All cores busy but much slower than expected, load average far above core count | Rust thread pools inside every worker (`TOKENIZERS_PARALLELISM` not set) | `export TOKENIZERS_PARALLELISM=false`, rerun |
| Warning `The current process just got forked, after parallelism has already been used` | Same cause | Same fix |
| `shard_check.py`: end-of-text count higher than the document count | Literal `<\|endoftext\|>` strings in the web text were encoded as real end-of-text tokens (see 03.1) | Update `mygpt/tokenizer.py` from the course lab code, delete the set and re-tokenize. Don't train on it |
| `shard_check.py`: token totals differ from `meta.json` | Run interrupted, or you re-ran into a folder holding old shards | Delete the folder entirely and re-tokenize; never mix two runs in one folder |
| `No space left on device` | Small disk (RunPod default 20 GB container disk, or a smaller Hetzner type) | Use ≥ 250 GB disk; on a small box delete `data/raw/sample-10BT` after the GPT-2 set is uploaded and checked |
| Killed with `Out of memory` | Too many workers for the RAM of a small box (each holds a tokenizer and batches) | Fewer workers, or a box with ≥ 2 GB RAM per worker |
| SSH dropped and the run stopped | Not started inside tmux | Restart in tmux; a stopped run must start over into an empty folder |
| Box still listed after you finished | Powered off instead of deleted | `hcloud server delete mygpt-cpu` (DigitalOcean: `doctl compute droplet delete mygpt-cpu`; RunPod: Terminate in the console); check the console list is empty |

## Artifact

- In the bucket: `tok/gpt2-10BT/` and `tok/bpe32k-100BT-30/` (shards, `meta.json`, and `tokenizer.json` for your BPE), both checked with `rclone check`.
- In git: `reports/tokenize-*.log`, `reports/meta-*.json`, the **Shard sets** table of `TOKEN_REPORT.md`.
- `budget.md` line (your real hours and price):

  ```text
  | 2026-11-13 | 03.2 | Hetzner CPX62 hel1, ~3.5 h at EUR 0.1779/h, tokenize 10B GPT-2 + 22B own BPE | ~$0.72 | $____ |
  ```
