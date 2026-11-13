"""Throughput benchmark: tokens/s, MFU and $ per 1B tokens for a config on this box.

  python scripts/bench.py --config configs/baseline_124m.yaml --steps 30 --price-per-gpu-hour 2.99
  torchrun --standalone --nproc_per_node=8 scripts/bench.py --config configs/baseline_124m.yaml --price-per-gpu-hour 3.99

Uses random tokens (no data needed) and the same model/optimizer/autocast/compile/DDP path as
training. The first --warmup steps (compilation, allocator warm-up) are not timed.
Extra key=value overrides work as in mygpt.train (e.g. train.micro_batch_size=32).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--steps", type=int, default=30, help="timed optimizer steps")
    p.add_argument("--warmup", type=int, default=5, help="untimed steps first")
    p.add_argument("--price-per-gpu-hour", type=float, default=4.0)
    p.add_argument("--accum", type=int, default=1, help="micro-steps per optimizer step (default 1: measures raw throughput)")
    p.add_argument("--out", default=None, help="append the result as one JSON line to this file")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args(argv)

    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    from mygpt.config import load_config
    from mygpt.model import GPT
    from mygpt.train import autocast_ctx, build_optimizer, peak_flops, setup_dist

    cfg = load_config(args.config, args.overrides)
    m, t = cfg.model, cfg.train
    rank, local_rank, world, device, ddp = setup_dist()
    torch.manual_seed(0)
    if device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")
    model = GPT(m).to(device)
    raw = model
    opts = build_optimizer(model, t, device)
    if t.compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[local_rank] if device.startswith("cuda") else None)
    ctx = autocast_ctx(device, t.dtype)
    B, T = t.micro_batch_size, t.seq_len
    x = torch.randint(0, m.vocab_size, (B, T + 1), device=device)
    inp, tgt = x[:, :-1], x[:, 1:]

    def step():
        for i in range(args.accum):
            sync = model.no_sync() if (ddp and i < args.accum - 1) else nullcontext()
            with sync:
                with ctx:
                    _, loss = model(inp, tgt)
                (loss / args.accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)
        return loss

    def sync_dev():
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    for _ in range(args.warmup):
        step()
    sync_dev()
    if ddp:
        dist.barrier()
    t0 = time.time()
    for _ in range(args.steps):
        loss = step()
    loss.item()
    sync_dev()
    dt = time.time() - t0
    tokens = args.steps * args.accum * B * T * world
    tok_s = tokens / dt
    peak = peak_flops(device)
    mfu = raw.flops_per_token(T) * tok_s / (peak * world) if peak else None
    gpus = world if device.startswith("cuda") else 0
    usd_per_1b = (args.price_per_gpu_hour * max(gpus, 1)) / (tok_s * 3600) * 1e9
    result = {"config": args.config, "overrides": args.overrides, "device": torch.cuda.get_device_name() if gpus else "cpu",
              "world": world, "params_m": round(raw.num_params() / 1e6, 1), "micro_batch": B, "seq_len": T,
              "dtype": t.dtype, "compile": t.compile, "tokens_per_s": round(tok_s, 1),
              "mfu": None if mfu is None else round(mfu, 4),
              "price_per_gpu_hour": args.price_per_gpu_hour, "usd_per_1b_tokens": round(usd_per_1b, 2),
              "hours_per_1b_tokens": round(1e9 / tok_s / 3600, 3)}
    if device.startswith("cuda"):
        result["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    if rank == 0:
        print(json.dumps(result, indent=2))
        if args.out:
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")
    if ddp:
        dist.destroy_process_group()
    return result


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    main()
