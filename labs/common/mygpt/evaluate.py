"""In-training evaluation: validation loss, bits-per-byte and a quick HellaSwag (acc_norm).

The final, reportable numbers come from lm-eval-harness on the exported HF model (Module 8);
these are cheap signals to watch during training.

CLI: python -m mygpt.evaluate --ckpt runs/x/ckpt_00010000.pt [--data-dir DIR] [--hellaswag 1000]
         [--eval-tokens N | -1 | all] [--batch N]
     (--ckpt may also be a run dir: its newest ckpt_*.pt is used)

Windows: contiguous seq_len blocks from the start of the val split, read --batch sequences per
forward pass (default: the checkpoint's training micro_batch_size; lower it to fit a smaller GPU).
--eval-tokens -1 (or `all`) evaluates every full window of the split, grouped by lcm(64, batch) x
seq_len as labs/module-08/hf_bpb.py does, so batches 64, 32, 16, 8 ... evaluate the same token count
(and the very same windows when the val split is one shard, as the common val set is).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import urllib.request
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F

HELLASWAG_URL = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"


def _all_reduce_mean(x: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(x)  # SUM works on both NCCL and gloo (AVG is NCCL-only)
        x /= dist.get_world_size()
    return x


@torch.no_grad()
def val_loss(model, loader, steps: int, ctx=nullcontext()) -> float:
    """Mean next-token loss (nats) over `steps` batches from the start of the val split."""
    was_training = model.training
    model.eval()
    loader.reset()
    total = torch.zeros((), device=loader.device)
    for _ in range(steps):
        x, y = loader.next_batch()
        with ctx:
            _, loss = model(x, y)
        total += loss.detach().float()
    model.train(was_training)
    return _all_reduce_mean(total / steps).item()


def bits_per_byte(loss_nats: float, tokens: int, nbytes: int) -> float:
    """Tokenizer-independent: (nats/token -> bits/token) * tokens/byte."""
    if not tokens or not nbytes:
        return float("nan")
    return loss_nats / math.log(2) * tokens / nbytes


def full_window_tokens(lengths: list[int], seq_len: int, group: int = 64) -> int:
    """Tokens the val loader covers when every shard is read in chunks of group*seq_len
    (a chunk needs one extra token for the shifted targets)."""
    chunk = group * seq_len
    return sum((n - 1) // chunk for n in lengths) * chunk


def parse_eval_tokens(value: str) -> int:
    """--eval-tokens: a token count (1e7 style accepted), or -1 / `all` for the whole split."""
    if str(value).strip().lower() in ("all", "-1"):
        return -1
    n = int(float(value))
    if n <= 0:
        raise argparse.ArgumentTypeError(f"--eval-tokens must be positive, -1 or 'all' (got {value})")
    return n


def val_bytes_per_token(meta: dict) -> tuple[int, int]:
    t = meta.get("val_tokens") or meta.get("total_tokens") or 0
    b = meta.get("val_bytes") or meta.get("total_bytes") or 0
    return t, b


# ------------------------------------------------------------------ HellaSwag
def load_hellaswag(path: str | None = None, cache_dir: str | None = None) -> list[dict]:
    if path is None:
        cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "mygpt")
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, "hellaswag_val.jsonl")
        if not os.path.exists(path):
            # Download to a per-process temp file and rename atomically, so concurrent ranks
            # never read a half-written file or interleave writes into the same one.
            tmp = f"{path}.{os.getpid()}.tmp"
            try:
                urllib.request.urlretrieve(HELLASWAG_URL, tmp)
                os.replace(tmp, path)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _render(example: dict, tok, max_len: int):
    ctx = tok.encode(example["ctx"])
    rows, masks = [], []
    for ending in example["endings"]:
        end = tok.encode(" " + ending)
        row = (ctx + end)[-max_len:]
        rows.append(row)
        masks.append([0] * (len(row) - len(end)) + [1] * len(end))
    L = max(len(r) for r in rows)
    ids = torch.zeros(4, L, dtype=torch.long)
    mask = torch.zeros(4, L, dtype=torch.long)
    for i, (r, m) in enumerate(zip(rows, masks)):
        ids[i, :len(r)] = torch.tensor(r)
        mask[i, :len(m)] = torch.tensor(m)
    return ids, mask, int(example["label"])


@torch.no_grad()
def hellaswag(model, tok, device, ctx=nullcontext(), limit: int | None = None,
              examples: list[dict] | None = None, rank: int = 0, world: int = 1) -> float:
    """acc_norm: pick the ending with the lowest mean per-token loss. Examples are split
    across ranks and the counts all-reduced."""
    was_training = model.training
    model.eval()
    examples = examples if examples is not None else load_hellaswag()
    if limit:
        examples = examples[:limit]
    block = model.cfg.block_size if hasattr(model, "cfg") else 1024
    correct = torch.zeros((), device=device)
    total = torch.zeros((), device=device)
    for i, ex in enumerate(examples):
        if i % world != rank:
            continue
        ids, mask, label = _render(ex, tok, block)
        ids, mask = ids.to(device), mask.to(device)
        with ctx:
            logits, _ = model(ids)
        logits = logits[:, :-1].float()
        tgt, m = ids[:, 1:], mask[:, 1:].float()
        losses = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="none").view(4, -1)
        avg = (losses * m).sum(1) / m.sum(1)
        correct += int(avg.argmin().item() == label)
        total += 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(correct)
        dist.all_reduce(total)
    model.train(was_training)
    return (correct / total.clamp(min=1)).item()


def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate a mygpt checkpoint (val loss, bpb, HellaSwag).")
    p.add_argument("--ckpt", required=True, help="checkpoint file, or a run dir (newest ckpt_*.pt)")
    p.add_argument("--data-dir", default=None, help="shards dir (default: the run's data_dir)")
    p.add_argument("--eval-tokens", type=parse_eval_tokens, default=10_485_760,
                   help="val tokens to evaluate (default 10485760), or -1 / all for every full window of the split")
    p.add_argument("--batch", type=int, default=None,
                   help="sequences per forward pass (default: the checkpoint's training micro_batch_size)")
    p.add_argument("--hellaswag", type=int, default=0, help="number of HellaSwag examples (0 = skip, -1 = all)")
    p.add_argument("--tokenizer", default=None, help="gpt2 | DIR (default: from shards meta.json)")
    args = p.parse_args(argv)

    from .checkpoint import latest_checkpoint, load_checkpoint
    from .config import config_from_dict
    from .data import DistributedTokenLoader, read_meta, resolve_tokenizer
    from .model import GPT
    from .tokenizer import get_tokenizer

    ckpt = args.ckpt
    if os.path.isdir(ckpt):
        ckpt = latest_checkpoint(ckpt)
        if ckpt is None:
            raise SystemExit(f"no ckpt_*.pt in {args.ckpt}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = load_checkpoint(ckpt)
    cfg = config_from_dict(ck["config"])
    model = GPT(cfg.model).to(device)
    model.load_state_dict(ck["model"])
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
    data_dir = args.data_dir or cfg.train.data_dir
    meta = read_meta(data_dir)
    out = {"ckpt": ckpt, "step": ck["step"]}
    if os.path.isdir(data_dir):
        B, T = args.batch or cfg.train.micro_batch_size, cfg.train.seq_len
        if B <= 0:
            raise SystemExit(f"--batch must be positive (got {B})")
        loader = DistributedTokenLoader(data_dir, "val", B, T, device=device)
        n_tokens = args.eval_tokens
        if n_tokens < 0:
            n_tokens = full_window_tokens(loader.ds.lengths, T, math.lcm(64, B))
            if n_tokens == 0:
                raise SystemExit(f"val split too small for one {math.lcm(64, B)} x {T} chunk; pass --eval-tokens N")
        elif n_tokens % (B * T):
            print(f"[evaluate] --eval-tokens {n_tokens} is not a multiple of batch x seq_len = {B * T}; "
                  f"evaluating {max(1, n_tokens // (B * T)) * B * T} tokens")
        steps = max(1, n_tokens // (B * T))
        out.update(batch=B, seq_len=T, eval_tokens=steps * B * T)
        out["val_loss"] = val_loss(model, loader, steps, ctx)
        out["val_bpb"] = bits_per_byte(out["val_loss"], *val_bytes_per_token(meta))
    if args.hellaswag:
        tok = get_tokenizer(args.tokenizer or resolve_tokenizer(data_dir, meta) or "gpt2")
        out["hellaswag"] = hellaswag(model, tok, device, ctx, None if args.hellaswag < 0 else args.hellaswag)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
