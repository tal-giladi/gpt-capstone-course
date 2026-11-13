"""Throughput benchmark on ONE GPU: per-GPU tokens/s, MFU and $ per 1B tokens for a config.

  python scripts/bench.py --config configs/baseline_124m.yaml --price-per-gpu-hour 2.99
  python scripts/bench.py --config configs/main_350m.yaml --steps 30 --json bench.jsonl train.compile=false

Runs single-process with the config's micro_batch_size and seq_len (even if the config is
written for 8 GPUs) and reports per-GPU numbers; multiply tokens/s by the GPU count for a
node (DDP on one node scales close to linearly; check with a short real run). The first
--warmup steps (torch.compile, allocator warm-up) are excluded from the timing. Reads real
shards from the config's data_dir if present, else random tokens (throughput does not depend
on content). MFU is against the dense BF16 peak of the detected GPU (mygpt.train.PEAK_FLOPS).
key=value overrides work as in mygpt.train. (Also works under torchrun: numbers stay per GPU.)
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
    p.add_argument("--warmup", type=int, default=5, help="untimed steps first (compile happens here)")
    p.add_argument("--price-per-gpu-hour", type=float, default=4.0)
    p.add_argument("--accum", type=int, default=1, help="micro-steps per optimizer step")
    p.add_argument("--json", "--out", dest="json_out", default=None, help="append the result as one JSON line here")
    p.add_argument("overrides", nargs="*", help="section.key=value, e.g. train.compile=false")
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

    loader, source = None, "random tokens"
    try:
        from mygpt.data import DistributedTokenLoader
        loader = DistributedTokenLoader(t.data_dir, "train", B, T, rank, world, device)
        source = f"shards in {t.data_dir}"
    except (FileNotFoundError, ValueError):
        if rank == 0:
            print(f"[bench] no shards at {t.data_dir!r}: using random tokens (throughput does not depend on content)")
    fixed = torch.randint(0, m.vocab_size, (B, T + 1), device=device)

    def batch():
        if loader is not None:
            return loader.next_batch()
        return fixed[:, :-1], fixed[:, 1:]

    def step():
        for i in range(args.accum):
            x, y = batch()
            sync = model.no_sync() if (ddp and i < args.accum - 1) else nullcontext()
            with sync:
                with ctx:
                    _, loss = model(x, y)
                (loss / args.accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)
        return loss

    def sync_dev():
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    if rank == 0:
        print(f"[bench] {raw.num_params() / 1e6:.1f}M params | micro {B}x{T} | {t.dtype} | compile={t.compile} | "
              f"{args.warmup} warm-up steps (untimed) + {args.steps} timed steps | data: {source}", flush=True)
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
    tok_s = args.steps * args.accum * B * T / dt  # per GPU
    peak = peak_flops(device)
    mfu = raw.flops_per_token(T) * tok_s / peak if peak else None
    usd_per_1b = args.price_per_gpu_hour / (tok_s * 3600) * 1e9
    gpu = torch.cuda.get_device_name() if device.startswith("cuda") else "cpu"
    result = {"config": args.config, "overrides": args.overrides, "gpu": gpu, "device": gpu,
              "params_m": round(raw.num_params() / 1e6, 1), "micro_batch": B, "seq_len": T,
              "dtype": t.dtype, "compile": t.compile, "data": source,
              "warmup_steps": args.warmup, "timed_steps": args.steps,
              "tokens_per_s": round(tok_s, 1), "tokens_per_s_per_gpu": round(tok_s, 1),
              "peak_tflops": None if peak is None else peak / 1e12,
              "mfu": None if mfu is None else round(mfu, 4),
              "price_per_gpu_hour": args.price_per_gpu_hour, "usd_per_1b_tokens": round(usd_per_1b, 2),
              "gpu_hours_per_1b_tokens": round(1e9 / tok_s / 3600, 3)}
    if device.startswith("cuda"):
        result["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    if rank == 0:
        print(json.dumps(result, indent=2))
        mfu_s = "n/a" if mfu is None else f"{mfu * 100:.1f}%"
        print(f"[bench] {tok_s:,.0f} tokens/s per GPU | MFU {mfu_s} | "
              f"${usd_per_1b:.2f} per 1B tokens at ${args.price_per_gpu_hour}/GPU-h")
        if args.json_out:
            with open(args.json_out, "a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")
    if ddp:
        dist.destroy_process_group()
    return result


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    main()
