# Token report

Filled in during lessons 03.1–03.3. Numbers come from `reports/*.json`; copy them, don't retype
from memory. This file is cited by your training configs, the eval report (Module 8) and the
technical report (Module 11).

## Tokenizer

| Field | Value |
|---|---|
| Type | byte-level BPE (Hugging Face `tokenizers` __version__), GPT-2 pre-tokenizer regex |
| Vocabulary | 32,768 including special tokens |
| Special tokens (id) | `<\|endoftext\|>` (0); chat format uses plain-text tags (no extra tokens) |
| Training text | first ___ documents of `sample/100BT/002_00009.parquet` (___ GB of text) |
| Training time / box | ___ min on ___ |
| `tokenizer.json` sha256 | `________________` |
| Bucket | `r2:mygpt/tokenizer/` |

### Compression on held-out FineWeb-Edu (`compare_tokenizers.py`, first ___ docs of `000_00000`)

| Tokenizer | Vocab | Bytes/token | Tokens vs GPT-2 | Output-layer share of compute (d=1024, 24 layers) |
|---|---|---|---|---|
| GPT-2 | 50,257 | | 0% | |
| Mine | 32,768 | | | |

## Shard sets

| Name | Tokenizer | Source | Train tokens | Val tokens | Train shards | Size | Bytes/token | Bucket |
|---|---|---|---|---|---|---|---|---|
| `gpt2-10BT` | GPT-2 | `sample/10BT`, 14 files | | | | | | `tok/gpt2-10BT/` |
| `bpe32k-100BT-30` | mine | `sample/100BT`, first 30 files | | | | | | `tok/bpe32k-100BT-30/` |
| `common-val-gpt2` | GPT-2 | main val docs not in `sample/10BT` | — | | — | | | `tok/common-val-gpt2/` |
| `common-val-bpe32k` | mine | same documents | — | | — | | | `tok/common-val-bpe32k/` |

Shard format: 256 × int32 header (magic 20261113, version 1), then uint16 tokens; every document
starts with the end-of-text id. `shard_check.py`: [ ] gpt2-10BT OK  [ ] bpe32k-100BT-30 OK

## Validation split

- Each set's val split = its first ~50M tokens (whole documents), never trained on.
- Overlap between the two inputs (`val_overlap.py`): ___ documents in both; main val docs in the
  baseline's input: ___ (___%); baseline val docs in the main input: ___ (___%).
- Rule for comparisons: a model is scored on its own val split, or on `common-val-*`, which
  neither model trains on. Cross-tokenizer comparisons use bits per byte, never loss per token.

## Decontamination (13-gram overlap, `decontam.py`)

| Set | Benchmark | Items | Items with a 13-gram in train | Docs dropped | Tokens dropped |
|---|---|---|---|---|---|
| gpt2-10BT | HellaSwag val | 10,042 | | | |
| | ARC-Easy test | | | | |
| | ARC-Challenge test | | | | |
| | PIQA val | | | | |
| bpe32k-100BT-30 | HellaSwag val | 10,042 | | | |
| | ARC-Easy test | | | | |
| | ARC-Challenge test | | | | |
| | PIQA val | | | | |

After filtering (`filter_shards.py`), `shard_check.py` is OK and `meta.json` has a `decontam` entry: [ ] gpt2-10BT [ ] bpe32k-100BT-30

Examples of what was flagged and my judgement (true contamination or common phrase):

1.
2.

## Cost

| Box | Hours | Price | Cost |
|---|---|---|---|
| | | | |
