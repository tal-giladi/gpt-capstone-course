---
id: "02.1"
module: 2
minutes: 15
practice_minutes: 30
prerequisites: ["01.1"]
objectives:
  - Compare FineWeb-Edu, FineWeb, DCLM-baseline, Dolma and SmolLM-Corpus on license, size and fit for a 20B-token run.
  - Explain what the FineWeb-Edu classifier filter keeps and what it costs you.
  - Compute how many gigabytes of parquet and how many files you need for 10B and ~20B tokens.
  - Record the data decision and its license terms in a draft data card.
volatility: implementation
sources:
  - title: "FineWeb-Edu dataset card (HuggingFaceFW/fineweb-edu)"
    url: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
  - title: "FineWeb dataset card (HuggingFaceFW/fineweb)"
    url: https://huggingface.co/datasets/HuggingFaceFW/fineweb
  - title: "The FineWeb Datasets (Penedo et al., 2024)"
    url: https://arxiv.org/abs/2406.17557
  - title: "DCLM-baseline 1.0 dataset card"
    url: https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0
  - title: "DataComp-LM paper (Li et al., 2024)"
    url: https://huggingface.co/papers/2406.11794
  - title: "Dolma dataset card (license change to ODC-By)"
    url: https://huggingface.co/datasets/allenai/dolma/blob/main/README.md
  - title: "SmolLM-Corpus dataset card"
    url: https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus
last_verified: "2026-11-13"
---

# 02.1 · Choosing pretraining data

A 350M model trained on 20B tokens is small enough that the data you pick moves the final scores more than most architecture tweaks you will try in Module 6. This lesson picks the corpus for both runs, checks that its license lets you publish the model, and turns "20B tokens" into a concrete number of files and gigabytes you will download in lesson 02.3.

## Goal

You end with a written data decision: FineWeb-Edu `sample/10BT` for the GPT-2 124M baseline and the first 30 files of `sample/100BT` (about 22B GPT-2 tokens) for your ~350M main run, the license terms you must honour, and the storage size to plan for (about 93 GB of parquet). You also measure the data yourself on a streamed sample instead of trusting the card.

## What you need

- Your project repo from [01.3](../module-01/lesson-03.md) with the project installed (`pip install -e .`), and the HF token from [01.2](../module-01/lesson-02.md) exported as `HF_TOKEN`.
- Your laptop. No cloud machine yet; the sample is streamed, not downloaded.
- About 45 minutes.
- **Expected cost: $0.**

## Choices and costs

### The candidates (checked 2026-11-13)

All of these are free to download from Hugging Face. What differs is size, filtering, license and how much work you need to use them.

| Dataset | What it is | Size | License | Fit for this project |
|---|---|---|---|---|
| **FineWeb-Edu** (`HuggingFaceFW/fineweb-edu`) | FineWeb filtered by an educational-quality classifier (rounded score >= 3 of 5) | 1.3T GPT-2 tokens; ready-made random samples of 10B, 100B, 350B tokens | ODC-By 1.0 + Common Crawl Terms of Use | **Chosen.** Ready samples at exactly our sizes, strong on knowledge/reasoning evals, the standard dataset for GPT-2-scale reproductions |
| FineWeb (`HuggingFaceFW/fineweb`) | Cleaned, deduplicated English Common Crawl, 96+ dumps | 18.5T GPT-2 tokens; samples of 10B/100B/350B | ODC-By 1.0 | Good control. Less "textbook" text; weaker on MMLU/ARC-type evals at small scale |
| FineWeb-Edu score-2 (`HuggingFaceFW/fineweb-edu-score-2`) | Same classifier, looser threshold (>= 2) | 5.4T tokens | ODC-By 1.0 | Only if you need more tokens than score-3 gives (you don't) |
| DCLM-baseline 1.0 (`mlfoundations/dclm-baseline-1.0`) | Common Crawl filtered with a fastText quality classifier (DataComp-LM) | Trillions of tokens, no small official samples (only community extracts) | CC-BY 4.0 | Strong alternative, but you must cut your own subset and shards are compressed JSONL |
| Dolma / OLMo mixes (`allenai/dolma`, `allenai/olmo-mix-1124`) | Full multi-source mixes (web, code, papers, books, wiki) | Dolma v1.7 ~3T tokens; Dolma 3 pool >9T | ODC-By (since 2024-04), plus the terms of each source | Overkill: you would need to re-mix sources yourself |
| SmolLM-Corpus (`HuggingFaceTB/smollm-corpus`) | `fineweb-edu-dedup` (~220B tokens) + Cosmopedia v2 (synthetic textbooks) + Python-Edu | 673 GB total | ODC-By (Python-Edu: The Stack v2 terms, contents fetched separately from Software Heritage) | Interesting later experiment (dedup + synthetic), not for the main run |

### Why FineWeb-Edu for this project

- **The edu filter is the cheapest quality gain you can buy.** The FineWeb authors had Llama-3-70B-Instruct score 500k pages for educational value, trained a small classifier on those scores and kept pages whose rounded score (`int_score`) is 3 or more, that is a raw `score` of about 2.5 and up. In their ablations this filtered subset gave much better results on knowledge and reasoning benchmarks (MMLU, ARC) than unfiltered FineWeb at the same token count ([paper](https://arxiv.org/abs/2406.17557)). At 20B tokens you cannot afford to spend tokens on spam and boilerplate.
- **The price you pay:** an English, explanation-heavy distribution. Less dialogue, fiction and code. Your model will write like a textbook. That is fine for HellaSwag/ARC-style evaluation and a small chat demo; write it down in the data card as a known property.
- **Samples at exactly our sizes.** The card says the 10BT sample is drawn from the 100BT sample, which is drawn from the 350BT sample, all at random. So `sample/10BT` is a fair 10B-token slice for the baseline, and any prefix of `sample/100BT` files is a fair random slice for the main run.
- **The license allows what you will do.** ODC-By lets you use the data for any purpose, including training and publishing a model, as long as you attribute it when you share the data or a derived database. You also accept the Common Crawl Terms of Use. You will put an attribution line in your model card in Module 11.

> [!IMPORTANT]
> A dataset license covers the dataset compilation, not every page inside it. Web text keeps whatever rights its authors have. This is the normal legal position for every web-crawled corpus and is why model cards say what data was used. This course is not legal advice; if you plan commercial use of the model, read the card's license section and Common Crawl's terms yourself.

### How much data: from tokens to files and gigabytes

Measured from the Hugging Face file listing on 2026-11-13:

| Subset | Files | Parquet size | GPT-2 tokens | GB per 1B tokens |
|---|---|---|---|---|
| `sample/10BT` | 14 (13 × 2.15 GB + one 0.54 GB) | 28.52 GB | ~10B | 2.85 |
| `sample/100BT` | 140 (~2.15 GB each) | 286.39 GB | ~100B | 2.86 |

For the main run you want ~20B tokens of **your own tokenizer** (Module 3). A 32k-vocabulary BPE trained on this data produces a token count within roughly ±10% of GPT-2's on English web text (an estimate; you will measure it exactly in lesson 03.3). So take a margin:

$$\text{files} = \left\lceil \frac{20 \times 2.86\ \text{GB} \times 1.12}{2.15\ \text{GB}} \right\rceil = \lceil 29.8 \rceil = 30 \text{ files} \approx 64.6\ \text{GB} \approx 22.5\text{B GPT-2 tokens}$$

Total raw download: 28.5 GB (baseline) + 64.6 GB (main) = **~93 GB**. Tokenized shards in Module 3 add about 2 bytes per token (uint16): ~20 GB for 10B tokens and ~45 GB for 22B tokens. With a few checkpoints, plan for **~180 GB** in your bucket. At Cloudflare R2's $0.015/GB-month that is under $3 per month (lesson 02.2).

> [!NOTE]
> Dataset files and sizes can change when the maintainers publish a new version (FineWeb-Edu's last data update was v1.4.0, July 2025). You pin the exact commit in lesson 02.3 so your numbers stay reproducible.

### How to read a sample (all free)

| Way | What it costs you | When |
|---|---|---|
| **`datasets` streaming** (`peek_fineweb.py` below) | A few hundred MB of network, no disk | Default: measure without downloading |
| One parquet file: `hf download HuggingFaceFW/fineweb-edu sample/10BT/000_00000.parquet --repo-type dataset --local-dir data/peek` | 2.15 GB of disk | Slow or flaky connection where streaming keeps restarting; read it with `pyarrow` |
| The dataset viewer on the card page | Nothing | A first look at a few rows and the `int_score` column, no numbers you can reuse |

Without an HF token all three share the low anonymous rate limit; your free read token from 01.2 raises it.

**Recommendation for this project:** FineWeb-Edu, `sample/10BT` (all 14 files) for the baseline, `sample/100BT` files `000_00000` to the 30th file for the main run. Don't mix in other corpora; one clean variable at a time.

## Do this

1. Open the [FineWeb-Edu card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) and find three things: the license line, the sentence about how the samples were drawn, and the Common Crawl terms link. Copy them into a scratch note.
2. Download the [Module 2 lab files](../../labs/module-02/) and copy them into a `tools/` folder in your project repo (they are small; commit them, except a filled-in `r2.env`). Then, from the repo root with the project's environment active, stream 2,000 documents and measure them. No files are saved; this reads a few hundred MB over the network.

   ```bash
   export HF_TOKEN=hf_...   # your read token from 01.2 (or already in your shell profile)
   python tools/peek_fineweb.py --docs 2000 --show 3
   ```

   It prints three sample documents, the `int_score` mix, average document length, UTF-8 bytes per GPT-2 token, and how many `sample/100BT` files you need for 20B tokens.
3. Read the three printed documents. Ask yourself: would I want my model to write like this? Note one thing the edu filter clearly removed (ads, forum chatter, product pages) and one thing it kept that surprised you.
4. Re-run with `--target-tokens 22e9` and check that the file count matches the 30-file plan above.
5. Start your data card. Copy [`labs/module-02/DATA_CARD.template.md`](../../labs/module-02/DATA_CARD.template.md) to the root of your project repo as `DATA_CARD.md` and fill in the **Summary** and **License and terms** sections now. Leave sizes, sha and bucket for lesson 02.3.
6. Add your storage estimate (~180 GB) to the "assumptions" part of `budget.md`; the storage line itself comes in 02.2.

## How you know it worked

- The script ends with lines like these (your exact numbers vary slightly with the sample):

  ```text
  documents sampled      : 2000
  int_score mix          : 3: 80%, 4: 19%, 5: 1%
  avg GPT-2 tokens / doc : 930
  UTF-8 bytes per token  : 4.57
  for 20B GPT-2 tokens: ~57 GB parquet = ~27 files of sample/100BT
  ```

- Every `int_score` is 3 or higher (a printed `score` of 2.6 is fine: it rounds to 3). If you see an `int_score` of 0, 1 or 2 you streamed the wrong dataset (FineWeb or score-2).
- Bytes per token is between about 4.3 and 4.9. GPT-2's tokenizer on clean English prose lands there; a much lower number means you are counting something odd (for example characters instead of bytes on non-ASCII text).
- `DATA_CARD.md` exists in your repo with dataset, license, attribution text and the two subsets filled in.

## When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `429 Too Many Requests` or very slow streaming | Anonymous requests share a low per-IP limit (3,000 file resolves per 5 minutes) | Export `HF_TOKEN`; free accounts get higher limits, and `huggingface_hub` >= 1.2 waits and retries automatically |
| `ValueError: BuilderConfig 'sample/10BT' not found` | Config names use a dash, file paths use a slash | Use `name="sample-10BT"` in `load_dataset`, `sample/10BT/*` for file paths |
| Script hangs at start for minutes | `datasets` resolving all 114 configs on a slow link | Wait once; or pass a token; it is cached afterwards |
| You are tempted to "just add" a bit of DCLM or code data | More sources feel like more quality | Don't. Two variables change at once and your baseline comparison breaks. Keep extras for an ablation in Module 6 |

## Artifact

- `DATA_CARD.md` in your project repo with the Summary and License sections filled in. Commit it:

  ```bash
  git add DATA_CARD.md budget.md
  git commit -m "Data decision: FineWeb-Edu 10BT + 30 files of 100BT"
  ```

- `budget.md` line (no spend in this lesson):

  ```text
  | 2026-11-13 | 02.1 | Data choice, streaming sample | $0.00 | $0.00 |
  ```
