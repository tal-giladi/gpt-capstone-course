# Module 3 lab files — tokenizer and token shards

Helpers for lessons 03.1–03.3. Copy them into the `tools/` folder of your project repo and
commit them, so the CPU box gets them with `git clone`. The project code they work with
(`scripts/train_tokenizer.py`, `scripts/tokenize_shards.py`, `scripts/decontam.py`,
`scripts/token_report.py`, `scripts/sync.sh`) is in `labs/common/`.

| File | Used in | What it does |
|---|---|---|
| `compare_tokenizers.py` | 03.1 | Bytes per token of your BPE vs GPT-2 on held-out FineWeb-Edu documents, projected token count of the main-run subset, output-layer share of compute; checks the end-of-text token and that a literal `<\|endoftext\|>` in web text is not turned into a real document break. |
| `shard_check.py` | 03.2, 03.3 | Checks a shard directory against `meta.json`: headers, token ids < vocab, token totals, one end-of-text per document, and zero occurrences of any other special token the tokenizer defines. |
| `val_overlap.py` | 03.3 | Compares FineWeb document ids between the two shard sets' inputs: are the val documents of one set in the other set's training data? Writes the shared validation set (`common_val.parquet`) that neither model trains on. |
| `filter_shards.py` | 03.3 | Removes the documents flagged by `decontam.py --drop` from train shards in place and updates `meta.json` (original kept as `meta.pre_decontam.json`). |
| `TOKEN_REPORT.template.md` | 03.3 | Template for the module's artifact, `TOKEN_REPORT.md`. |

Requirements: the project installed (`uv pip install -e . --torch-backend cpu` on a CPU box),
Python 3.11+, numpy, pyarrow, tokenizers (all project dependencies). Run from the repo root so
`mygpt` is importable. Tested on 2026-11-13 on a small synthetic dataset (Windows 11, Python
3.14, tokenizers 0.23.3, tiktoken 0.14); the full-size runs happen on the learner's CPU box.

No secrets in these files. They never delete anything outside the shard directory you name.
