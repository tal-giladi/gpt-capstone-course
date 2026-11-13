---
id: "03.3"
module: 3
minutes: 20
practice_minutes: 180
prerequisites: ["03.2"]
objectives:
  - Explain how the validation split is formed and check that validation documents are not in the other model's training data.
  - Build a shared validation set that both tokenizers see as identical text, for honest bits-per-byte comparisons.
  - Measure 13-gram overlap between your training shards and HellaSwag, ARC and PIQA, judge the flagged examples, and remove contaminated documents.
  - Produce the token report that later configs, evals and the technical report cite.
volatility: implementation
sources:
  - title: "Language Models are Few-Shot Learners (GPT-3), appendix C: 13-gram contamination analysis"
    url: https://arxiv.org/abs/2005.14165
  - title: "HellaSwag dataset (Rowan/hellaswag)"
    url: https://huggingface.co/datasets/Rowan/hellaswag
  - title: "AI2 ARC dataset (allenai/ai2_arc)"
    url: https://huggingface.co/datasets/allenai/ai2_arc
  - title: "PIQA dataset copy used by lm-eval (baber/piqa)"
    url: https://huggingface.co/datasets/baber/piqa
  - title: "FineWeb-Edu dataset card (columns, samples)"
    url: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
  - title: "Hetzner Cloud regular-performance (CPX) prices"
    url: https://www.hetzner.com/cloud/regular-performance/
  - title: "DigitalOcean Droplet pricing (CPU-Optimized)"
    url: https://www.digitalocean.com/pricing/droplets
last_verified: "2026-11-13"
---

# 03.3 · Validation split, decontamination and the token report

Two things can quietly make your numbers lie: validation text the model has already trained on, and benchmark questions that sit in the training data. Neither shows up as an error; both show up as results that are better than the model really is. This lesson checks your validation splits against the other shard set, builds one shared validation text for comparing the GPT-2 baseline with your own-tokenizer model, scans both training sets for 13-gram overlap with the benchmarks you will report in Module 8, removes what it finds, and writes the token report that the rest of the course cites.

## Goal

You know how many validation documents of each shard set appear in the other set's input, you have two val-only shard sets with identical text (`tok/common-val-gpt2/`, `tok/common-val-bpe32k/`), both training sets are filtered of documents that share a 13-gram with HellaSwag, ARC-Easy, ARC-Challenge or PIQA, and `TOKEN_REPORT.md` holds every count with the JSON reports behind it. The CPU box is deleted.

## What you need

- The box, raw data and both shard sets from [03.2](lesson-02.md) (or a new box: `pull-raw` both subsets and `pull-tok` both sets, ~10 min).
- `HF_TOKEN` exported on the box (the benchmark datasets download from Hugging Face).
- About 3.5 hours, mostly waiting for the decontamination scan.
- **Expected cost: about $0.50–0.70** (~2.5–3.5 h of a CPX62 at €0.1779/h).

## Choices and costs

### The validation split you already have

`tokenize_shards.py` wrote the first ~50M tokens of each input, as whole documents, to `val_000000.bin`, and everything after to `train_*.bin`. Those documents are never trained on by the model that uses this set, and the split does not move if you add data later. 50M tokens is plenty: the training loop evaluates on a slice (`eval_tokens`), and the full split gives a low-noise final number.

The catch is **across** sets. `sample-10BT` and the first 30 files of `sample-100BT` are separate random samples of the same corpus and may share documents. If they do, some of the main set's validation documents are in the baseline's training data, and the other way round. Comparing the two models on either set's val split would then favour one of them. `val_overlap.py` checks this by FineWeb's unique document `id` instead of trusting assumptions about how the samples were drawn.

| How to compare baseline and main model | Verdict |
|---|---|
| Loss per token, each on its own val split | Wrong: different tokenizers make per-token losses incomparable |
| Bits per byte, each on its own val split | Acceptable: same distribution, both held out from their own model |
| Bits per byte on the **common val set** (main val documents not in `sample-10BT`, tokenized by both tokenizers) | **Best:** identical text, held out from both models. Used in Module 8 |

### Decontamination: what and how

A benchmark item is contaminated if its text (or most of it) appears in the training data, so the model can recall rather than reason. The standard cheap test, used since GPT-3, is n-gram overlap: does any 13-token window of the item appear anywhere in the training tokens? `scripts/decontam.py` hashes every 13-gram of the benchmark items (correct answer included, tokenized with the same tokenizer as the shards), then streams all training shards and flags every document containing one of those hashes.

| Benchmark (split lm-eval scores) | Items | Where its text comes from | What to expect (ESTIMATE) |
|---|---|---|---|
| HellaSwag (validation) | 10,042 | ActivityNet captions and WikiHow articles | The most hits: WikiHow pages are on the open web and in Common Crawl |
| ARC-Easy / ARC-Challenge (test) | 2,376 / 1,172 | Grade-school science exam questions | Some hits from quiz and homework sites |
| PIQA (validation) | 1,838 | Crowd-written physical commonsense, inspired by instructables.com | Few hits |

Choices you make here:

| Decision | Options | Pick |
|---|---|---|
| What to do with flagged documents | Keep and report; drop from training; drop the benchmark items instead | **Drop the documents** (with `tools/filter_shards.py`) and report the counts. They are a tiny fraction of tokens; the benchmarks stay complete and comparable with published numbers |
| Which sets | Main only; both | **Both**: the baseline is compared with your model, so both get the same treatment |
| Re-tokenize with `--exclude` vs filter shards | Re-tokenizing costs hours; filtering rewrites only affected shards in minutes | **Filter shards** in place; `meta.json` keeps the before/after counts |
| False positives | 13-gram matches on stock phrases | Read the examples in the report; drop anyway (cost ~0), but say in the report what kind of matches they were |

### Cost and time (checked 2026-11-13)

The scan is single-process per shard set and numpy-bound: about 5 seconds per 20M tokens per core (ESTIMATE from a laptop core), so roughly **2 hours for the 22B-token set** and **1 hour for the 10B set**, run in parallel. Everything else takes minutes. On the CPX62 (€0.1779/h) the lesson costs about €0.50.

| Box for this lesson | Price (checked 2026-11-13) | Lesson cost | Note |
|---|---|---|---|
| **Hetzner CPX62** (the box from 03.2) | €0.1779/h | ~€0.50 | Default; the scan is single-process per set, so 2 busy cores are enough |
| Hetzner CPX42 (8 vCPU, 16 GB, 320 GB) | €0.0953/h | ~€0.30 | Enough RAM and disk if you start fresh (pull both shard sets, ~65 GB, and the raw files for `val_overlap.py`, ~93 GB) |
| RunPod CPU pod or DigitalOcean CPU-Optimized (16 vCPU) | ~$0.50–0.60/h (estimate for RunPod; $0.50 on DigitalOcean) | ~$1.50–2 | Fallbacks; ≥ 200 GB disk |

> [!NOTE]
> Re-check the box price if you create a new box, and put the hours you actually used in the ledger.

> [!CAUTION]
> `filter_shards.py` rewrites shards **in place** and the bucket copy is replaced on the next upload. Run it only after the original shard sets are in the bucket and checked (end of 03.2), run it with `--dry-run` first, and never run it twice on the same directory (it refuses to).

## Do this

### 1. Check validation overlap and build the common val set (10 min)

On the box, in tmux, venv and `r2.env` loaded, `HF_TOKEN` exported:

```bash
cd ~/mygpt && mkdir -p reports
python tools/val_overlap.py \
  --main-raw data/raw/sample-100BT --main-shards data/tok/bpe32k-100BT-30 \
  --base-raw data/raw/sample-10BT  --base-shards data/tok/gpt2-10BT \
  --out reports/val_overlap.json --write-common-val data/common_val/common_val.parquet
```

`--main-raw` must contain exactly the 30 files you tokenized, and `--base-raw` the 14 files of `sample-10BT`: the script maps val documents to rows by file order.

### 2. Tokenize the common val set with both tokenizers (2 min)

`--val-tokens` larger than the input puts every document into `val_*.bin`:

```bash
export TOKENIZERS_PARALLELISM=false
python scripts/tokenize_shards.py --input data/common_val --tokenizer gpt2 \
  --out data/tok/common-val-gpt2 --val-tokens 1000000000000 --workers $(( $(nproc) - 1 ))
python scripts/tokenize_shards.py --input data/common_val --tokenizer tokenizers/bpe32k \
  --out data/tok/common-val-bpe32k --val-tokens 1000000000000 --workers $(( $(nproc) - 1 ))
python tools/shard_check.py --shards data/tok/common-val-gpt2
python tools/shard_check.py --shards data/tok/common-val-bpe32k
bash scripts/sync.sh push-tok common-val-gpt2   data/tok/common-val-gpt2
bash scripts/sync.sh push-tok common-val-bpe32k data/tok/common-val-bpe32k
```

### 3. Scan both training sets for benchmark overlap (1–2 h, in parallel)

Window 1:

```bash
python scripts/decontam.py --shards-or-parquet data/tok/bpe32k-100BT-30 \
  --tokenizer data/tok/bpe32k-100BT-30 --benchmarks hellaswag,arc_easy,arc_challenge,piqa \
  --out reports/decontam-bpe32k-100BT-30.json --drop reports/drop-bpe32k-100BT-30.json \
  2>&1 | tee reports/decontam-bpe32k-100BT-30.log
```

Window 2 (`Ctrl-b c`):

```bash
cd ~/mygpt && . .venv/bin/activate && export HF_TOKEN
python scripts/decontam.py --shards-or-parquet data/tok/gpt2-10BT \
  --tokenizer gpt2 --benchmarks hellaswag,arc_easy,arc_challenge,piqa \
  --out reports/decontam-gpt2-10BT.json --drop reports/drop-gpt2-10BT.json \
  2>&1 | tee reports/decontam-gpt2-10BT.log
```

For your BPE set, `--tokenizer` points at the shard folder, which holds the `tokenizer.json` the shards were made with; the benchmark text must be tokenized with exactly that tokenizer or no 13-gram can match.

### 4. Read what was flagged (10 min)

```bash
python -c "
import json
for name in ['gpt2-10BT', 'bpe32k-100BT-30']:
    r = json.load(open(f'reports/decontam-{name}.json'))
    print(name, r['tokens_scanned'], 'tokens,', r['flagged_docs'], 'docs flagged')
    for b, v in r['benchmarks'].items():
        print(f'  {b:14s} {v[\"contaminated_items\"]:>5}/{v[\"items\"]:<6} {100*v[\"contaminated_frac\"]:.2f}%')
        for ex in v['examples'][:2]: print('     ', ex[:150].replace(chr(10), ' '))
"
```

The examples are benchmark items that matched. To see the training document behind a match, decode one entry of the drop list (before step 5, which removes them):

```bash
python -c "
import json
from mygpt.data import read_shard
from mygpt.tokenizer import get_tokenizer
f, s, e = json.load(open('reports/drop-gpt2-10BT.json'))[0]
print(get_tokenizer('gpt2').decode(read_shard('data/tok/gpt2-10BT/' + f)[s:e].tolist())[:1500])
"
```

Decide for two or three matches whether this is real contamination (a quiz page with the question and answer, the WikiHow article a HellaSwag context was cut from) or a stock phrase. Write your judgement in the Decontamination section of `TOKEN_REPORT.md`.

### 5. Remove the flagged documents and re-upload (15 min)

```bash
for s in gpt2-10BT bpe32k-100BT-30; do
  python tools/filter_shards.py --shards data/tok/$s --drop reports/drop-$s.json --dry-run
done
for s in gpt2-10BT bpe32k-100BT-30; do
  python tools/filter_shards.py --shards data/tok/$s --drop reports/drop-$s.json     && python tools/shard_check.py --shards data/tok/$s     && bash scripts/sync.sh push-tok $s data/tok/$s     && rclone check data/tok/$s r2:mygpt/tok/$s --one-way     || { echo "STOP: $s failed, nothing more uploaded"; break; }
done
```

`push-tok` uses `rclone copy`, which replaces the rewritten shards and `meta.json` in the bucket and adds `meta.pre_decontam.json`; untouched shards are skipped. The `&&` chain makes sure a failed filter or check never reaches the bucket.

### 6. Token reports and `TOKEN_REPORT.md` (20 min)

```bash
for s in gpt2-10BT bpe32k-100BT-30 common-val-gpt2 common-val-bpe32k; do
  python scripts/token_report.py --shards data/tok/$s --out reports/token_report-$s.json
  cp data/tok/$s/meta.json reports/meta-$s.json
done
```

Fill in the remaining sections of `TOKEN_REPORT.md` from `reports/`. Copy the reports to your laptop and commit them, and put a copy of the report next to the shards:

```bash
# on your laptop
scp -r root@<IP>:mygpt/reports/ .
git add reports/ TOKEN_REPORT.md && git commit -m "Token report: shards, val overlap, decontamination" && git push
rclone copy TOKEN_REPORT.md r2:mygpt/tok/
```

### 7. Kill checklist and delete the box (5 min)

All shard sets checked in the bucket, reports in git, then:

```bash
rclone lsf r2:mygpt/tok/                 # gpt2-10BT/ bpe32k-100BT-30/ common-val-gpt2/ common-val-bpe32k/ TOKEN_REPORT.md
hcloud server delete mygpt-cpu
hcloud server list                       # empty
```

On DigitalOcean: `doctl compute droplet delete mygpt-cpu`; on RunPod: Terminate the pod in the console.

## How you know it worked

- `val_overlap.py` prints the four counts. Whatever they are, your report states them; if "main val docs the baseline trains on" is above zero, the common val set is smaller than the main val split by exactly that many documents, and you will use it for every cross-model comparison.
- Each common-val set passes `shard_check.py` with `train: none (val-only directory)` and the same number of documents in both; their token counts differ in the same ratio as the bytes per token you measured in 03.1.
- Both decontam logs end with one line per benchmark and `N training documents flagged`. Expect the flagged documents to be a tiny fraction of the corpus (far below 0.1% of tokens); a large fraction means something is off (see below).
- `filter_shards.py` prints the removed documents and tokens, and `shard_check.py` stays `OK` afterwards (the document and end-of-text counts drop by the same number). `meta.json` has a `decontam` entry; `meta.pre_decontam.json` holds the original counts.
- `TOKEN_REPORT.md` has every table filled in, and the box no longer exists.

## When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `val_overlap.py`: `meta.json has no val_docs` | Shards made with an older `tokenize_shards.py` | Update the repo from the course code; re-tokenize (meta needs per-split document counts) |
| Overlap counts look absurd (for example 100% of val in the other set) | `--main-raw`/`--base-raw` point at more or other files than were tokenized | Point them at exactly the folders `tokenize_shards.py` read |
| decontam reports 0 hits everywhere, even HellaSwag | Benchmark text tokenized with a different tokenizer than the shards (for example `--tokenizer gpt2` on the BPE set) | Use the shard folder as `--tokenizer` for your BPE set |
| Thousands of HellaSwag items "contaminated", flagged docs in the hundreds of thousands | Very short or generic items, or the wrong `--n` | Keep the default `--n 13`; read examples; report honestly. Dropping is still cheap, but check the token fraction before you do |
| decontam is far slower than ~5 s per 20M tokens | Old version of `decontam.py` using a slow membership test | Update the repo from the course code; the scan should take hours, not a day |
| `ConnectionError` or 429 while loading benchmarks | No token, or a network blip | `export HF_TOKEN=...`; rerun; finished work is not lost because nothing is written until the end |
| `filter_shards.py` refuses: `already filtered` | You ran it before on this folder | Fine if it succeeded (check `meta.json`'s `decontam` entry) |
| `filter_shards.py` crashed halfway (disk full, killed) | Some shards rewritten, `meta.json` not yet updated | Do not rerun on that folder. Delete it, `sync.sh pull-tok` the unfiltered set from the bucket (the loop uploads only after success), and filter again |
| `rclone check` shows differences after the filter | Upload interrupted | Rerun `sync.sh push-tok`; check again |

## Artifact

- In the bucket: filtered `tok/gpt2-10BT/` and `tok/bpe32k-100BT-30/` (with `meta.pre_decontam.json`), val-only `tok/common-val-gpt2/` and `tok/common-val-bpe32k/`, and `tok/TOKEN_REPORT.md`.
- In git: `TOKEN_REPORT.md` and `reports/` (`val_overlap.json`, `decontam-*.json`, `drop-*.json`, `token_report-*.json`, `meta-*.json`, logs). This is the module's artifact: Module 4 benchmarks on these shards, Module 8 cites the contamination numbers, Module 11 copies the tables into the technical report.
- `budget.md` line (your real hours and price):

  ```text
  | 2026-11-13 | 03.3 | Hetzner CPX62 hel1, ~3 h at EUR 0.1779/h, val overlap, decontamination, reports | ~$0.62 | $____ |
  ```
