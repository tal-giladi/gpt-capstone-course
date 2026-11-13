# Code contract — `labs/common/` (the learner's project code)

One small, pinned, tested Python project the learner clones onto every box. Package name
**`mygpt`**. Lessons call ONLY the commands listed here; if a lesson needs something else, it
goes to `curriculum/code-requests/module-NN.md`.

The learner already wrote all of this from scratch in the previous course, so this code is
compact, readable and production-shaped (DDP, bf16, compile, resume) rather than didactic.

## Layout

```
labs/common/
  README.md                 what this is, install, the command list below
  pyproject.toml            package mygpt, Python >=3.11, pinned deps (current stable versions,
                            checked 2026-11): torch, numpy, pyyaml, tiktoken, tokenizers,
                            datasets, huggingface_hub, pyarrow, wandb, transformers;
                            optional extras [post] = trl, peft, accelerate; [eval] = lm-eval;
                            [demo] = gradio; [dev] = pytest
  mygpt/
    __init__.py
    config.py               dataclasses ModelConfig, TrainConfig; load_config(path, overrides)
    model.py                GPT with switchable parts (below)
    muon.py                 Muon optimizer (Newton-Schulz), single-GPU and DDP-safe
    data.py                 ShardDataset / DistributedTokenLoader over .bin shards
    tokenizer.py            get_tokenizer("gpt2" | path) -> object with encode/decode/eot_id/vocab_size
    checkpoint.py           save/load/latest + optional `rclone copy` to a remote
    train.py                entry point (torchrun)
    evaluate.py             val loss, bits-per-byte, HellaSwag (in-training quick eval)
    export_hf.py            checkpoint -> Hugging Face format
  scripts/
    download_fineweb.py     --subset {sample-10BT,sample-100BT} --max-files N --out DIR
    train_tokenizer.py      --input DIR(parquet) --vocab-size 32768 --max-docs N --out DIR
    tokenize_shards.py      --input DIR --tokenizer {gpt2|DIR} --out DIR --shard-tokens 100000000
                            --val-tokens 50000000 --workers N --max-tokens N
    token_report.py         --shards DIR --tokenizer ... -> prints tokens, bytes/token, writes JSON
    decontam.py             --shards-or-parquet DIR --tokenizer ... --benchmarks hellaswag,arc_easy,arc_challenge,piqa
                            13-gram overlap report (JSON); optional --drop to write a filtered list
    bench.py                --config FILE [--steps 30] -> tokens/s, MFU, $/1B tokens at --price-per-gpu-hour
    sft.py                  TRL SFT on an exported HF model (--model DIR --dataset NAME --out DIR ...)
    dpo.py                  TRL DPO (--model DIR --dataset NAME --out DIR ...)
    chat.py                 interactive / one-shot generation from an HF-format model
    judge_winrate.py        pairwise win-rate of two models' answers on N prompts via an
                            OpenAI-compatible or Anthropic API judge (key from env); --dry-run
    autoshutdown.sh         powers the box off after N minutes of GPU idle (nvidia-smi poll)
    sync.sh                 rclone helpers: push/pull data and checkpoints
  configs/
    smoke_cpu.yaml          tiny model, synthetic data, runs on CPU in <1 min
    baseline_124m.yaml      faithful GPT-2 124M, GPT-2 tokenizer, 10B tokens, 8 GPUs
    ablation_base.yaml      124M, own tokenizer, ~1B tokens, 1 GPU (the control run)
    main_350m.yaml          ~350M template (d_model 1024, 24 layers, 16 heads), ~20B tokens, 8 GPUs
  tests/                    pytest, CPU-only, < 2 min total
```

## Model switches (`ModelConfig`)

`vocab_size, n_layer, n_head, n_embd, block_size, dropout=0.0` plus:
- `pos_emb: learned | rope` (rope_theta)
- `norm: layernorm | rmsnorm`
- `mlp: gelu | swiglu` (swiglu hidden = round to multiple of 64 of 8/3·d)
- `qk_norm: bool`
- `bias: bool`
- `tie_embeddings: bool`
Attention uses `torch.nn.functional.scaled_dot_product_attention(is_causal=True)`.
GPT-2 init (0.02, residual projections scaled by 1/sqrt(2·n_layer)).

**Export rule:** all-GPT-2 settings → `GPT2LMHeadModel`. `rope + rmsnorm + swiglu + bias=False`
→ `LlamaForCausalLM` (or `Qwen3ForCausalLM` when `qk_norm=True`). Any other mix trains fine but
`export_hf.py` refuses with a clear message ("ablation-only combination"). Export verifies
logits match the native model on a random batch (max abs diff < 1e-3 in fp32) and writes the
tokenizer files so `AutoTokenizer.from_pretrained(out)` works.

## Training (`TrainConfig`, `python -m mygpt.train`)

Launch: `torchrun --standalone --nproc_per_node=8 -m mygpt.train --config configs/X.yaml [key=value ...]`
(overrides like `train.lr=6e-4 model.qk_norm=true`). Also runs single-process on CPU/GPU.

Fields: `data_dir, train_tokens, total_batch_tokens, micro_batch_size, seq_len, lr, min_lr_ratio,
warmup_tokens, schedule: cosine|wsd (decay_frac), optimizer: adamw|muon (muon_lr),
weight_decay, grad_clip, dtype: bf16|fp32, compile: bool, seed, out_dir, run_name,
eval_every, eval_tokens, hellaswag_every (0=off), ckpt_every, keep_last, ckpt_remote (rclone
remote path or empty), wandb_project (empty = off), log_every`.

Behavior: grad accumulation derived from total_batch_tokens; DDP with `no_sync` on accum
steps; auto-resume from the latest checkpoint in out_dir (or pulled from ckpt_remote) incl.
optimizer, step, data position, RNG; JSONL log `out_dir/log.jsonl` (step, loss, lr, grad_norm,
tokens/s, MFU, val_loss, val_bpb, hellaswag); prints a one-line summary every log_every;
detects NaN/inf loss and stops with the last good checkpoint named; SIGTERM → save then exit.

## Shard format

`DIR/train_000000.bin`, …, `DIR/val_000000.bin`: 256 int32 header (magic 20261113, version 1,
token count, vocab size) then uint16 tokens. Documents separated by the tokenizer's EOT id at
the start of each document. `DIR/meta.json`: tokenizer, vocab_size, eot_id, total tokens,
total bytes (for bits-per-byte).

## Tests (must pass on CPU: `cd labs/common && pip install -e .[dev] && pytest -q`)

model forward/backward for every switch combination; export round-trip for GPT-2 and Llama
mappings (logits match); tokenizer train → encode/decode round-trip; shard write/read; loader
determinism + resume position; train smoke (smoke_cpu.yaml, 20 steps, loss decreases,
checkpoint resume reproduces the next loss); Muon step runs; bench runs on CPU.
