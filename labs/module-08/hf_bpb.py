"""Bits per byte of a Hugging Face causal LM on mygpt val shards, with exactly the windows
`python -m mygpt.evaluate` uses (lesson 08.2).

    # OpenAI GPT-2 on the common val text, GPT-2 tokenization (the whole split)
    python tools/hf_bpb.py --model openai-community/gpt2 --data-dir data/tok/common-val-gpt2 --eval-tokens all
    # your exported model on the same text, own tokenization (cross-check vs mygpt.evaluate)
    python tools/hf_bpb.py --model hf/my-gpt-350m-base --data-dir data/tok/common-val-bpe32k --eval-tokens all

Windows: contiguous, non-overlapping blocks of --seq-len tokens from the start of the val split
(each document starts with EOT), next-token loss averaged over all positions, read with
mygpt.data.DistributedTokenLoader, as in mygpt.evaluate. bpb = loss / ln 2 * tokens / bytes, with
tokens and bytes of the whole split from meta.json, as in mygpt.evaluate.

--eval-tokens all evaluates every full window of the split and prints the token count to pass to
`mygpt.evaluate --eval-tokens` so a native checkpoint covers the very same windows (the count is a
multiple of 64 x seq_len, so micro-batches 64, 32 and 16 all stop at the same place).

The shards must have been made with the model's tokenizer: the script encodes a probe sentence with
both and refuses to run if they differ (a mismatched tokenizer gives a meaningless bpb).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import nullcontext

PROBE = "The quick brown fox jumps over the lazy dog. Photosynthesis converts light into chemical energy."


def full_window_tokens(lengths: list[int], seq_len: int, group: int = 64) -> int:
    """Tokens covered by the loader when every shard is read in chunks of group*seq_len."""
    chunk = group * seq_len
    return sum((n - 1) // chunk for n in lengths) * chunk


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF model id or local dir")
    p.add_argument("--data-dir", required=True, help="mygpt shards dir with val_*.bin and meta.json")
    p.add_argument("--eval-tokens", default="10485760", help="number of tokens, or 'all' (default 10485760, as mygpt.evaluate)")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch", type=int, default=16, help="sequences per forward pass")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16", help="autocast dtype on CUDA (CPU: always fp32)")
    p.add_argument("--device", default=None)
    p.add_argument("--no-tokenizer-check", action="store_true")
    p.add_argument("--json", help="also write the result to this JSON file")
    a = p.parse_args(argv)

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mygpt.data import DistributedTokenLoader, ShardDataset, read_meta, resolve_tokenizer
    from mygpt.evaluate import bits_per_byte, val_bytes_per_token
    from mygpt.tokenizer import get_tokenizer

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    meta = read_meta(a.data_dir)
    if not meta:
        raise SystemExit(f"{a.data_dir}/meta.json missing")
    if not a.no_tokenizer_check:
        hf_tok = AutoTokenizer.from_pretrained(a.model)
        shard_tok = get_tokenizer(resolve_tokenizer(a.data_dir, meta))
        mine, theirs = list(shard_tok.encode(PROBE)), hf_tok(PROBE, add_special_tokens=False)["input_ids"]
        if mine != theirs:
            raise SystemExit(f"tokenizer mismatch: shards use {meta.get('tokenizer')!r}, the model's tokenizer encodes "
                             f"the probe differently ({mine[:8]}... vs {theirs[:8]}...). Use the shards made with "
                             "this model's tokenizer.")

    ds = ShardDataset(a.data_dir, "val")
    T = a.seq_len
    per_step = a.batch * T
    if a.eval_tokens == "all":
        n_tokens = full_window_tokens(ds.lengths, T)
    else:
        n_tokens = int(float(a.eval_tokens))
    if n_tokens % per_step:
        raise SystemExit(f"--eval-tokens {n_tokens} must be a multiple of batch*seq_len = {per_step}")
    steps = n_tokens // per_step
    loader = DistributedTokenLoader(a.data_dir, "val", a.batch, T, device=device)

    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(device).eval()
    if T > getattr(model.config, "max_position_embeddings", getattr(model.config, "n_positions", T)):
        raise SystemExit(f"--seq-len {T} exceeds the model's context")
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if (device.startswith("cuda") and a.dtype == "bf16") else nullcontext()

    t0, total = time.time(), 0.0
    with torch.no_grad():
        for i in range(steps):
            x, y = loader.next_batch()
            with ctx:
                logits = model(input_ids=x).logits
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))
            total += loss.item()
            if (i + 1) % max(1, steps // 10) == 0:
                print(f"[hf_bpb] {i + 1}/{steps} batches, running loss {total / (i + 1):.4f}", file=sys.stderr, flush=True)
    loss = total / steps
    vt, vb = val_bytes_per_token(meta)
    out = {"model": a.model, "data_dir": a.data_dir, "tokenizer": meta.get("tokenizer"), "eval_tokens": n_tokens,
           "seq_len": T, "dtype": a.dtype if device.startswith("cuda") else "fp32", "val_loss": round(loss, 5),
           "val_bpb": round(bits_per_byte(loss, vt, vb), 5), "bytes_per_token": round(vb / vt, 4) if vt else None,
           "split_tokens": sum(ds.lengths), "seconds": round(time.time() - t0, 1),
           "mygpt_evaluate_eval_tokens": full_window_tokens(ds.lengths, T)}
    print(json.dumps(out, indent=2))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
