# Code requests from Module 3 (for labs/common owner)

Lessons 03.1–03.3 use only flags that exist in labs/common as of 2026-11-13 14:40. Items 1 and 2
are needed for the lessons' checks and timings to hold; the rest are improvements.

1. **BLOCKING — literal special strings in web text become real special tokens (own BPE).**
   `HFBPETokenizer.encode/encode_batch` use `Tokenizer.encode(..., add_special_tokens=False)`.
   That flag only controls post-processor templates; added tokens are still matched inside the
   text, so any FineWeb page containing the literal string `<|endoftext|>` is encoded with the
   real EOT id (a fake document break). Verified with tokenizers 0.23.3:
   `"a <|endoftext|> b"` -> `[.., 0, ..]`. tiktoken's `encode_ordinary` (GPT-2 path) is fine.
   Fix: in `HFBPETokenizer.__init__` set `self._tok.encode_special_tokens = True` (property
   exists in 0.23.x; with it the string is split into ordinary byte-level tokens). If chat/SFT
   code ever needs the opposite, give it a separate flag. Please add a test: encoding
   `"x <|endoftext|> y"` must not contain `eot_id`.
   Lesson checks that rely on it: `labs/module-03/compare_tokenizers.py` (literal-EOT check) and
   `labs/module-03/shard_check.py` (number of EOT tokens == `train_docs + val_docs` in meta.json).
   On the synthetic test the bug produced 11 extra EOTs in 4,500 docs.
2. **decontam.py speed (shard mode).** `np.isin(h, table)` costs ~16 s per 20M-token chunk on a
   laptop core; `searchsorted` against the already-sorted `table` costs ~3 s
   (`i = np.searchsorted(table, h); i[i == table.size] = 0; mask = table[i] == h`). With isin a
   22.5B-token shard set takes ~6 h single-threaded; with searchsorted ~1.7 h. Lesson 03.3 quotes
   "about 2 hours" for the main set and runs the two sets in parallel. Nice to have: `--workers N`
   over shards (then ~15 min); the lesson does not use it yet.
3. **Reserved chat tokens (decision for Tal / Module 9 author).** `train_bpe` adds only
   `<|endoftext|>`, and `export_hf` uses plain-text `<|user|>`/`<|assistant|>` tags with EOT ending
   each answer, so lesson 03.1 explains that no extra tokens are needed and does not reserve any.
   If ChatML special tokens are preferred for the own-tokenizer model, the time is now: add
   `train_tokenizer.py --special-tokens "<|endoftext|>,<|im_start|>,<|im_end|>,<|pad|>"` (EOT first,
   id 0; vocab size includes them) and tell me; 03.1 then gets one extra flag and a paragraph.
   Adding tokens after pretraining means resizing embeddings with untrained rows.
4. **`train_tokenizer.py` RAM.** The docstring says 30–60 GB for 2M documents and the README
   example uses `--max-docs 2000000`. Lesson 03.1 uses `--max-docs 500000` from one file
   (`002_00009.parquet`, ~2.3 GB of text, similar to nanochat's 2B characters) so it fits the
   32 GB CPX62. If you measured memory, tell me the number.
5. **`tokenize_shards.py` env.** The lesson sets `TOKENIZERS_PARALLELISM=false` before running with
   a multiprocessing pool (avoids Rust thread oversubscription and the fork warning). Consider
   setting it inside the script.
6. **`val_overlap.py` assumption.** The lab helper relies on val documents = the first
   `meta.json["val_docs"]` rows of the input in `_common.parquet_files` order (sorted recursive
   glob) with no `--exclude`. Please keep that ordering, and keep `val_docs`/`train_docs` in meta.json.
7. **New shard sets introduced by 03.3 (for Modules 5 and 8):** `tok/common-val-gpt2/` and
   `tok/common-val-bpe32k/` — val-only directories (`val_*.bin` + meta.json, no train shards) with
   the same documents tokenized by both tokenizers (main val docs not present in sample-10BT).
   Module 8 can compute bits-per-byte for both models on identical text. If `mygpt.evaluate`
   needs a flag to evaluate a checkpoint on another directory's val split, please add/confirm it.
8. **`.gitignore` and `tokenizers/`.** Lesson 01.3 replaces the repo `.gitignore` with
   `labs/module-01/project.gitignore` (a superset of labs/common's). It ignores `tokenizers/`, so
   03.1 commits `tokenizers/bpe32k/tokenizer.json` with `git add -f` (a few MB, worth versioning).
9. **Decontamination workflow.** 03.3 decontaminates the finished shards (decontam shard mode
   `--drop` -> `labs/module-03/filter_shards.py`, which rewrites affected train shards in place and
   updates meta.json) instead of re-tokenizing with `--exclude`, because shard mode is fast and
   re-tokenizing 22B tokens costs hours. `--exclude` stays available.
