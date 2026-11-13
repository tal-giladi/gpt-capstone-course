# mygpt: the capstone's project code

One small Python package (`mygpt`) plus scripts that you clone onto every box: data download,
tokenizer training, sharding, decontamination, a GPT-2-class model with switchable improvements,
DDP pretraining with checkpoint/resume to a bucket, throughput benchmarking, export to Hugging
Face format, and thin TRL wrappers for SFT and DPO. You already wrote all of this from scratch in
the previous course; this version is compact and production-shaped rather than didactic.

Versions checked against PyPI on **2026-11-13**; tested on 2026-11-13 (CPU only, see the end).

## Install on a GPU box (Ubuntu + CUDA)

```bash
git clone <your project repo> && cd <repo>
python3 -m venv .venv && source .venv/bin/activate      # or: uv venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"            # core: torch (PyPI wheel = CUDA 13.0 build, needs driver >= 580), tiktoken, ...
pip install -e ".[post,eval]"      # + TRL/PEFT/accelerate, judge SDKs, lm-eval[hf] (Modules 8-10; .[eval] alone is enough for Module 8)
pip install -e ".[demo]"           # + gradio (Module 11)
pytest -q                          # < 2 min, CPU is enough; also passes on the GPU box
nvidia-smi && python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

CPU-only box (data prep, tokenization): `uv pip install -e . --torch-backend cpu` (or
`pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu`). The data scripts do
not import torch at all.

Pins (lower bound = current stable on 2026-11-13, upper = next major): torch 2.15, numpy 2.5,
pyyaml 6.0.3, tiktoken 0.14, tokenizers 0.23, datasets 5.2, huggingface_hub 2.4, pyarrow 26,
wandb 0.31, transformers 5.22, safetensors 0.8; `[post]` trl 1.17, peft 0.21, accelerate 1.15,
anthropic 2.8, openai 3.50; `[eval]` lm-eval[hf] 0.4.13; `[demo]` gradio 6.32; `[dev]` pytest 9.2.

## Layout

| Path | What |
|---|---|
| `mygpt/config.py` | `ModelConfig`, `TrainConfig`, `load_config(path, overrides)` |
| `mygpt/model.py` | GPT with switches: `pos_emb` learned/rope, `norm` layernorm/rmsnorm, `mlp` gelu/swiglu, `qk_norm`, `bias`, `tie_embeddings` |
| `mygpt/muon.py` | Muon (Newton-Schulz), DDP-safe |
| `mygpt/data.py` | shard format, `ShardDataset`, `DistributedTokenLoader` (resumable) |
| `mygpt/tokenizer.py` | `get_tokenizer("gpt2" or DIR)`, `train_bpe` |
| `mygpt/checkpoint.py` | atomic save, keep-last-N, latest, rclone push/pull |
| `mygpt/train.py` | training entry point (torchrun or single process) |
| `mygpt/evaluate.py` | val loss, bits-per-byte (`--batch N`, `--eval-tokens all` for the whole split), quick HellaSwag |
| `mygpt/export_hf.py` | checkpoint -> `GPT2LMHeadModel` / `LlamaForCausalLM` / `Qwen3ForCausalLM`; `--push` creates a private Hub repo (`--no-private` for public) |
| `scripts/` | data, benchmark, post-training and ops scripts (below) |
| `configs/` | `smoke_cpu`, `baseline_124m`, `ablation_base`, `main_350m` |
| `tests/` | pytest, CPU-only |

## Commands (one example each)

```bash
# --- data (CPU box) ---
python scripts/download_fineweb.py --subset sample-10BT --out data/raw/sample-10BT          # flat *.parquet in --out; --max-files N, --revision SHA
python scripts/train_tokenizer.py --input data/raw/sample-100BT --vocab-size 32768 --max-docs 2000000 --out tokenizers/bpe32k
python scripts/tokenize_shards.py --input data/raw/sample-10BT --tokenizer gpt2 --out data/tok/gpt2-10BT --workers 32
python scripts/token_report.py --shards data/tok/gpt2-10BT                                  # -> data/tok/gpt2-10BT/token_report.json
python scripts/decontam.py --shards-or-parquet data/tok/gpt2-10BT --tokenizer gpt2 --benchmarks hellaswag,arc_easy,arc_challenge,piqa --out decontam.json   # also: winogrande, lambada_openai
python scripts/tokenize_shards.py --input data/raw/sample-100BT --tokenizer tokenizers/bpe32k --out data/tok/bpe32k-100BT-30 --exclude drop.json   # skip docs flagged by decontam --drop

# --- bucket (rclone; root = $MYGPT_REMOTE, default r2:mygpt; only ever `rclone copy`) ---
bash scripts/sync.sh push-tok gpt2-10BT data/tok/gpt2-10BT          # also: push-raw/pull-raw SUBSET, push-tokenizer, push-manifests, push-hf, push/pull, ls
bash scripts/sync.sh pull-ckpt baseline_124m runs/baseline_124m     # newest checkpoint + log.jsonl

# --- GPU box ---
bash scripts/autoshutdown.sh --idle-minutes 20 --grace-minutes 10 &  # --stop-cmd '...' for non-RunPod providers
python scripts/bench.py --config configs/baseline_124m.yaml --price-per-gpu-hour 2.99 --json bench.jsonl
python -m mygpt.train --config configs/smoke_cpu.yaml                                     # CPU smoke test, < 1 min
torchrun --standalone --nproc_per_node=8 -m mygpt.train --config configs/baseline_124m.yaml train.wandb_project=my-gpt   # W&B alerts on NaN stop / finish (train.wandb_alerts=false: off)
# free alternative tracker: pip install -e ".[trackio]" and add train.tracker=trackio (same project key; no alerts)
python -m mygpt.train --config configs/ablation_base.yaml model.pos_emb=rope train.out_dir=runs/abl_rope train.run_name=abl_rope
python -m mygpt.evaluate --ckpt runs/baseline_124m --hellaswag -1                         # file or run dir; --eval-tokens all --batch 16
python -m mygpt.export_hf --ckpt runs/main_350m --out hf/my-gpt-350m                      # --push you/my-gpt-350m: upload to a PRIVATE repo

# --- eval / post-training ---
lm_eval --model hf --model_args pretrained=hf/my-gpt-350m,dtype=bfloat16 --tasks hellaswag,arc_easy,arc_challenge,piqa --batch_size 32
python scripts/sft.py --model hf/my-gpt-350m --out hf/my-gpt-350m-sft                     # default data HuggingFaceTB/smol-smoltalk
python scripts/dpo.py --model hf/my-gpt-350m-sft --out hf/my-gpt-350m-dpo                 # default data trl-lib/ultrafeedback_binarized
python scripts/chat.py --model hf/my-gpt-350m-dpo --chat --prompt "Explain photosynthesis simply."
python scripts/judge_winrate.py --model-a hf/my-gpt-350m-sft --model-b hf/my-gpt-350m-dpo --n 200 --out judge/dpo_vs_sft   # --dry-run first
```

## Training behaviour

- Overrides: `section.key=value` after the config (`train.lr=6e-4 model.qk_norm=true`).
- Gradient accumulation = `total_batch_tokens / (micro_batch_size * seq_len * world_size)`; DDP
  gradient sync only on the last micro-step (`no_sync`).
- Auto-resume: newest `out_dir/ckpt_*.pt` (model, optimizers, step, data position, RNG). If
  `out_dir` is empty and `ckpt_remote` is set, the newest checkpoint (and `log.jsonl`, if
  `out_dir` has none) is pulled first. A resumed run reproduces the uninterrupted loss curve
  (tested). Training config may change on resume (e.g. a smaller `train_tokens` branches a WSD
  decay off a stable checkpoint, tested); the model config may not.
- `train.skip_batches=N`: on resume, advance the data loader N global batches (optimizer steps'
  worth of data) and log it, e.g. to step past data that repeatedly produces a NaN. Applied once:
  resuming again from a checkpoint written with the same N does not skip again.
- Checkpoints every `ckpt_every` steps and at the end, keep `keep_last`, uploaded in the
  background with `rclone copy` when `ckpt_remote` is set (e.g. `r2:mygpt/ckpt/baseline_124m`),
  together with the current `log.jsonl`.
- Optimizers: `weight_decay` is AdamW's (>=2D tensors only); with `optimizer: muon` the block
  matrices use `muon_weight_decay` (default 0.0) instead.
- SIGTERM (preemption, `kill`) -> checkpoint after the current step, exit 0.
- NaN/inf loss or gradient -> stop with exit code 1 and print the last good checkpoint.
- `out_dir/log.jsonl`: step, loss, lr, grad_norm, tokens_per_s, mfu, val_loss, val_bpb, hellaswag.
  MFU uses the dense BF16 peak of the detected GPU (H100 SXM 989, H100 NVL 835, H100 PCIe 756,
  H200 989, B200 2250, A100 312 TFLOP/s). FLOPs/token = 6N + 12·L·d·T, where N excludes lookup
  tables (wpe, and wte when embeddings are untied).
- `data_dir: synthetic` generates a learnable toy dataset (used by `smoke_cpu.yaml`).
- `sft.py` / `dpo.py`: weights loaded and trained in fp32 (`dtype=torch.float32`, master weights)
  with bf16 autocast on a bf16 GPU (`bf16=True`), so small updates are not rounded away.
  `--resume` continues from the newest `OUT/checkpoint-*`; `--save-final-only` deletes those
  checkpoints after the final save; `--dataset-revision SHA` pins a hub dataset. `dpo.py` keeps
  only prompt/chosen/rejected and fixes a string `prompt` next to message lists (H4, Argilla,
  Tulu). `sft.py --packing` requires `--attn-implementation flash_attention_2` (or `_3`).
- `judge_winrate.py`: `summary.json` has the judge's real input/output tokens and the estimated
  cost at list prices, a Wilson 95% CI (ties = 0.5, always in [0, 1]), and invalid prompts split
  into declined by the judge vs no parseable verdict (each invalid row keeps the raw replies).

## Shards

`DIR/train_000000.bin`, ..., `DIR/val_000000.bin`: 256 x int32 header (magic 20261113,
version 1, token count, vocab size) then uint16 tokens; each document starts with the EOT id.
`DIR/meta.json`: tokenizer, vocab_size, eot_id, token and byte counts per split (for
bits-per-byte). For your own BPE, `tokenize_shards.py` copies `tokenizer.json` into `DIR`, so the
shards and their tokenizer travel together (`sync.sh push-tok`).

## Export rule

All-GPT-2 settings -> `GPT2LMHeadModel` (Conv1D weights transposed). `rope + rmsnorm + swiglu +
bias=false` -> `LlamaForCausalLM`, or `Qwen3ForCausalLM` with `qk_norm=true`. Anything else is an
"ablation-only combination" and is refused. Every export checks fp32 logits against the native
model (max abs diff < 1e-3) and writes the tokenizer with a plain-text chat template
(`<|user|>` / `<|assistant|>` tags, EOT ends each answer; no new tokens).

## What was tested where (2026-11-13)

Tested locally on Windows 11, Python 3.14.3, torch 2.15.0+cpu: `pytest -q` (all pass), the
smoke run, `export_hf` for all three architectures (logits match after reload), and a tiny
end-to-end `sft.py` -> `dpo.py` -> `chat.py` run with TRL 1.17.1 / transformers 5.22.2, and
`lm_eval ... --tasks hellaswag --limit 8` (lm-eval 0.4.13) on a tiny exported Qwen3 model.
Not run locally (no GPU): CUDA, bf16 autocast on GPU, `torch.compile`, multi-process DDP /
torchrun (Windows torchrun lacks libuv), NCCL, rclone uploads, wandb, real FineWeb downloads,
judge API calls, gradio (installed only). Run `pytest -q` and `scripts/bench.py` on the GPU box
before spending money on a long run.

Windows: on a cp1252 console lm-eval 0.4.13 writes its results and then crashes with
`UnicodeEncodeError` while printing the table; set `PYTHONUTF8=1` before running `lm_eval`.
W&B (`train.wandb_alerts`), `export_hf --push` and the WinoGrande/LAMBADA loaders in `decontam.py`
are tested with mocks only (no network).
