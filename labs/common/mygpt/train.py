"""Pretraining entry point.

  torchrun --standalone --nproc_per_node=8 -m mygpt.train --config configs/baseline_124m.yaml
  python -m mygpt.train --config configs/smoke_cpu.yaml train.lr=1e-3 model.qk_norm=true

Single process (CPU or one GPU) or DDP under torchrun. Auto-resumes from the newest
checkpoint in out_dir (pulling it from ckpt_remote first if the box is fresh). SIGTERM ->
checkpoint, then exit 0. NaN/inf loss -> exit 1 naming the last good checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .checkpoint import latest_checkpoint, load_checkpoint, rclone_pull_latest, rclone_push, save_checkpoint
from .config import Config, load_config
from .data import DistributedTokenLoader, make_synthetic_shards, read_meta, resolve_tokenizer
from .evaluate import bits_per_byte, hellaswag, load_hellaswag, val_bytes_per_token, val_loss
from .model import GPT
from .muon import Muon

# Dense bf16 tensor-core peak per GPU (no sparsity), for MFU. First substring match of the
# CUDA device name wins, so specific variants come before "H100" (= SXM). Unknown -> no MFU.
PEAK_FLOPS = [("H100 NVL", 835e12), ("H100 PCIe", 756e12), ("H100", 989e12), ("H200", 989e12),
              ("B200", 2250e12), ("A100", 312e12), ("L40S", 362e12), ("A6000", 155e12),
              ("4090", 165e12), ("L4", 121e12), ("A10", 125e12)]


def peak_flops(device: str) -> float | None:
    if not device.startswith("cuda"):
        return None
    name = torch.cuda.get_device_name(torch.device(device))
    for key, val in PEAK_FLOPS:
        if key in name:
            return val
    return None


def setup_dist():
    """Returns (rank, local_rank, world_size, device, ddp)."""
    if "RANK" in os.environ:
        rank, local_rank, world = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device, backend = f"cuda:{local_rank}", "nccl"
        else:
            device, backend = "cpu", "gloo"
        dist.init_process_group(backend=backend)
        return rank, local_rank, world, device, True
    return 0, 0, 1, ("cuda" if torch.cuda.is_available() else "cpu"), False


def autocast_ctx(device: str, dtype: str):
    if dtype == "bf16":
        return torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16)
    return nullcontext()


def build_optimizer(model: GPT, t, device: str) -> list[torch.optim.Optimizer]:
    """AdamW for everything, or Muon for 2D block matrices + AdamW for the rest
    (embeddings, lm_head, norms, biases). In AdamW, `weight_decay` applies to >=2D tensors
    only. Muon groups use their own `muon_weight_decay`: Muon decays as p *= 1 - muon_lr * wd,
    and with muon_lr ~0.02 the AdamW value of 0.1 would shrink the blocks far too hard."""
    fused = device.startswith("cuda")
    params = dict(model.named_parameters())  # tied lm_head/wte appears once
    if t.optimizer == "muon":
        muon_p = [p for n, p in params.items() if p.ndim == 2 and n.startswith("transformer.h.")]
        rest = [(n, p) for n, p in params.items() if not (p.ndim == 2 and n.startswith("transformer.h."))]
    else:
        muon_p, rest = [], list(params.items())
    decay = [p for _, p in rest if p.ndim >= 2]
    no_decay = [p for _, p in rest if p.ndim < 2]
    opts = [torch.optim.AdamW([{"params": decay, "weight_decay": t.weight_decay},
                               {"params": no_decay, "weight_decay": 0.0}],
                              lr=t.lr, betas=(0.9, 0.95), eps=1e-8, fused=fused)]
    if muon_p:
        opts.append(Muon(muon_p, lr=t.muon_lr, momentum=0.95, weight_decay=t.muon_weight_decay))
    for opt in opts:
        for g in opt.param_groups:
            g["base_lr"] = g["lr"]
    return opts


def lr_mult(step: int, total: int, warmup: int, t) -> float:
    if step < warmup:
        return (step + 1) / warmup
    if t.schedule == "cosine":
        p = (step - warmup) / max(1, total - warmup)
        return t.min_lr_ratio + (1 - t.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    decay_start = int(total * (1 - t.decay_frac))  # wsd: warmup, stable, linear decay
    if step < decay_start:
        return 1.0
    return 1.0 - (1.0 - t.min_lr_ratio) * (step - decay_start) / max(1, total - decay_start)


class _ModuleRun:
    """Adapts a module-level tracker API (trackio.log / trackio.finish) to wandb's run object."""

    def __init__(self, mod):
        self._mod = mod

    def log(self, metrics: dict, step: int | None = None):
        self._mod.log(metrics, step=step)

    def finish(self):
        self._mod.finish()


def _rng_state(device: str) -> dict:
    s = {"torch": torch.get_rng_state()}
    if device.startswith("cuda"):
        s["cuda"] = torch.cuda.get_rng_state(torch.device(device))
    return s


def _set_rng_state(s: dict, device: str):
    torch.set_rng_state(s["torch"])
    if device.startswith("cuda") and "cuda" in s:
        torch.cuda.set_rng_state(s["cuda"], torch.device(device))


def train(cfg: Config) -> dict:
    m, t = cfg.model, cfg.train
    rank, local_rank, world, device, ddp = setup_dist()
    master = rank == 0
    # Seeding with seed + rank makes each rank initialise *different* weights below. That is
    # fine: DDP broadcasts rank 0's parameters (and buffers) to all ranks when it wraps the
    # model, so training starts from identical weights everywhere. The per-rank seed only
    # matters for things that should differ per rank (e.g. dropout masks).
    torch.manual_seed(t.seed + rank)
    if device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")
    os.makedirs(t.out_dir, exist_ok=True)

    def barrier():
        if ddp:
            dist.barrier()

    # ---- data
    data_dir = t.data_dir
    if data_dir == "synthetic":
        data_dir = os.path.join(t.out_dir, "synthetic_data")
        if master and not os.path.exists(os.path.join(data_dir, "meta.json")):
            make_synthetic_shards(data_dir, vocab_size=m.vocab_size, seed=t.seed)
        barrier()
    meta = read_meta(data_dir)
    if meta.get("vocab_size", 0) > m.vocab_size:
        raise ValueError(f"shards use vocab {meta['vocab_size']} > model.vocab_size {m.vocab_size}")
    B, T = t.micro_batch_size, t.seq_len
    assert T <= m.block_size, "train.seq_len must be <= model.block_size"
    per_micro = B * T * world
    if t.total_batch_tokens % per_micro:
        raise ValueError(f"total_batch_tokens {t.total_batch_tokens} not divisible by micro*seq*world = {per_micro}")
    accum = t.total_batch_tokens // per_micro
    total_steps = t.train_tokens // t.total_batch_tokens
    warmup = t.warmup_tokens // t.total_batch_tokens
    train_loader = DistributedTokenLoader(data_dir, "train", B, T, rank, world, device)
    try:
        val_loader = DistributedTokenLoader(data_dir, "val", B, T, rank, world, device)
    except (FileNotFoundError, ValueError) as e:
        val_loader = None
        if master:
            print(f"[train] no validation: {e}")
    eval_steps = max(1, t.eval_tokens // per_micro)

    # ---- model / optimizer / resume
    model = GPT(m).to(device)
    raw_model = model
    optimizers = build_optimizer(model, t, device)
    if master and t.ckpt_remote and latest_checkpoint(t.out_dir) is None:
        rclone_pull_latest(t.ckpt_remote, t.out_dir)
    barrier()
    start_step, last_good, skipped = 0, None, 0
    ck_path = latest_checkpoint(t.out_dir)
    if ck_path:
        ck = load_checkpoint(ck_path, map_location="cpu")
        if ck["config"]["model"] != cfg.to_dict()["model"]:
            raise ValueError(f"{ck_path} was trained with a different model config; use a new out_dir")
        raw_model.load_state_dict(ck["model"])
        for opt, sd in zip(optimizers, ck["optimizers"]):
            opt.load_state_dict(sd)
        train_loader.load_state_dict(ck["loader"])
        _set_rng_state(ck["rng"], device)
        start_step, last_good = ck["step"], ck_path
        if master:
            print(f"[train] resumed from {ck_path} at step {start_step}")
        # skip_batches: jump the data past a stretch that keeps producing NaNs. Applied once:
        # if the checkpoint was written by a run that already used this value, don't skip again.
        skip_done = ck["config"]["train"].get("skip_batches", 0)
        if t.skip_batches and t.skip_batches != skip_done:
            for _ in range(t.skip_batches * accum):
                train_loader.next_batch()
            skipped = t.skip_batches
        del ck
    if t.compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[local_rank] if device.startswith("cuda") else None)
    eval_model = model.module if ddp else model
    ctx = autocast_ctx(device, t.dtype)

    if master:
        n = raw_model.num_params()
        print(f"[train] params {n / 1e6:.1f}M | world {world} | micro {B}x{T} | accum {accum} | "
              f"steps {total_steps} | device {device} | dtype {t.dtype}")
    tok, hs_examples = None, None
    tok_name = resolve_tokenizer(data_dir, meta)
    if t.hellaswag_every and tok_name not in (None, "synthetic"):
        from .tokenizer import get_tokenizer
        tok = get_tokenizer(tok_name)
        # Rank 0 downloads (if needed) first; the others read the finished file after the barrier.
        if master:
            hs_examples = load_hellaswag()
        barrier()
        if not master:
            hs_examples = load_hellaswag()
    vt, vb = val_bytes_per_token(meta)

    run = None  # an object with .log(metrics, step=...) and .finish()
    if master and t.wandb_project and t.tracker != "none":
        if t.tracker == "trackio":
            # Trackio (free, Hugging Face) mirrors the wandb init/log/finish API; no alerts.
            import trackio
            trackio.init(project=t.wandb_project, name=t.run_name, config=cfg.to_dict(), resume="allow")
            run = _ModuleRun(trackio)
        elif t.tracker == "wandb":
            import wandb
            run = wandb.init(project=t.wandb_project, name=t.run_name, id=t.run_name, resume="allow",
                             config=cfg.to_dict())
        else:
            raise ValueError(f"train.tracker must be wandb, trackio or none, not {t.tracker!r}")
    log_path = os.path.join(t.out_dir, "log.jsonl")
    log_f = open(log_path, "a", encoding="utf-8") if master else None

    stop = {"flag": False}

    def on_sigterm(signum, frame):
        stop["flag"] = True
        if master:
            print("[train] SIGTERM received: will checkpoint after this step and exit")

    old_handler = signal.signal(signal.SIGTERM, on_sigterm)
    flops_tok, peak = raw_model.flops_per_token(T), peak_flops(device)
    uploads: list = []

    def checkpoint(step_done: int) -> str | None:
        nonlocal uploads, last_good
        path = None
        if master:
            state = {"model": raw_model.state_dict(), "optimizers": [o.state_dict() for o in optimizers],
                     "loader": train_loader.state_dict(), "rng": _rng_state(device),
                     "config": cfg.to_dict(), "meta": {**meta, "tokenizer_resolved": tok_name}}
            path = save_checkpoint(t.out_dir, step_done, state, t.keep_last)
            last_good = path
            if t.ckpt_remote:
                for proc in uploads:
                    proc.wait()
                # The log goes up with every checkpoint, so the bucket's log.jsonl never lags
                # its newest checkpoint by more than one interval (e.g. after a SIGKILL).
                uploads = [p for p in (rclone_push(path, t.ckpt_remote),
                                       rclone_push(log_path, t.ckpt_remote)) if p is not None]
        barrier()
        return path

    def alert(title: str, text: str, level: str = "INFO"):
        """W&B alert (email/Slack per the user's W&B settings); never lets a failure stop training."""
        if run is None or not t.wandb_alerts or t.tracker != "wandb":
            return
        try:
            import wandb
            lvl = getattr(getattr(wandb, "AlertLevel", None), level, None)
            run.alert(title=title, text=text, **({"level": lvl} if lvl is not None else {}))
        except Exception as e:  # noqa: BLE001 - alerts are best effort
            print(f"[train] wandb alert failed: {e}")

    def log(rec: dict):
        if master:
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            if run is not None:
                run.log(rec, step=rec["step"])

    if skipped:
        if master:
            print(f"[train] skip_batches={skipped}: advanced the train loader {skipped} global batches "
                  f"({skipped * t.total_batch_tokens} tokens) past the step-{start_step} position")
        log({"step": start_step, "skip_batches": skipped, "skipped_tokens": skipped * t.total_batch_tokens})

    t_last, steps_since = time.time(), 0
    exit_code, stopped = 0, False
    last_loss = float("nan")
    try:
        for step in range(start_step, total_steps):
            step_done = step + 1
            last = step_done == total_steps
            lrm = lr_mult(step, total_steps, warmup, t)
            for opt in optimizers:
                for g in opt.param_groups:
                    g["lr"] = g["base_lr"] * lrm
            loss_acc = torch.zeros((), device=device)
            for micro in range(accum):
                x, y = train_loader.next_batch()
                sync_ctx = model.no_sync() if (ddp and micro < accum - 1) else nullcontext()
                with sync_ctx:
                    with ctx:
                        _, loss = model(x, y)
                    loss = loss / accum
                    loss_acc += loss.detach()
                    loss.backward()
            if ddp:
                dist.all_reduce(loss_acc)
                loss_acc /= world
            clip = t.grad_clip if t.grad_clip > 0 else float("inf")
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            last_loss, gn = loss_acc.item(), grad_norm.item()
            if not (math.isfinite(last_loss) and math.isfinite(gn)):
                if master:
                    print(f"[train] non-finite loss/grad at step {step_done} (loss={last_loss}, grad_norm={gn}). "
                          f"Stopping. Last good checkpoint: {last_good or 'none'}")
                    alert(f"{t.run_name}: non-finite loss at step {step_done}",
                          f"loss={last_loss}, grad_norm={gn}. Training stopped. "
                          f"Last good checkpoint: {last_good or 'none'}", "ERROR")
                exit_code = 1
                break
            for opt in optimizers:
                opt.step()
                opt.zero_grad(set_to_none=True)
            steps_since += 1

            rec = None
            if step_done % t.log_every == 0 or last or step == start_step:
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                dt = time.time() - t_last
                tok_s = steps_since * t.total_batch_tokens / max(dt, 1e-9)
                mfu = flops_tok * tok_s / (peak * world) if peak else 0.0
                rec = {"step": step_done, "loss": last_loss, "lr": optimizers[0].param_groups[0]["lr"],
                       "grad_norm": gn, "tokens_per_s": tok_s, "mfu": mfu,
                       "tokens": step_done * t.total_batch_tokens}
                if master:
                    print(f"step {step_done:>6}/{total_steps} | loss {last_loss:.4f} | lr {rec['lr']:.2e} | "
                          f"gnorm {gn:.2f} | {tok_s / 1e3:.1f}k tok/s | "
                          f"mfu {f'{mfu * 100:.1f}%' if peak else 'n/a'}", flush=True)
                t_last, steps_since = time.time(), 0
            if val_loader is not None and t.eval_every and (step_done % t.eval_every == 0 or last):
                rec = rec or {"step": step_done}
                rec["val_loss"] = val_loss(eval_model, val_loader, eval_steps, ctx)
                rec["val_bpb"] = bits_per_byte(rec["val_loss"], vt, vb)
                if master:
                    print(f"step {step_done:>6} | val_loss {rec['val_loss']:.4f} | val_bpb {rec['val_bpb']:.4f}")
            if tok is not None and (step_done % t.hellaswag_every == 0 or last):
                rec = rec or {"step": step_done}
                rec["hellaswag"] = hellaswag(raw_model, tok, device, ctx, examples=hs_examples, rank=rank, world=world)
                if master:
                    print(f"step {step_done:>6} | hellaswag acc_norm {rec['hellaswag']:.4f}")
            if rec is not None:
                log(rec)

            stopping = stop["flag"]
            if ddp:  # all ranks must agree, otherwise one rank waits forever in a collective
                f = torch.tensor(float(stopping), device=device)
                dist.all_reduce(f, op=dist.ReduceOp.MAX)
                stopping = f.item() > 0
            if (t.ckpt_every and step_done % t.ckpt_every == 0) or last or stopping:
                path = checkpoint(step_done)
                if master and stopping:
                    print(f"[train] saved {path} after SIGTERM; exiting")
                t_last = time.time()  # don't count checkpoint time in tokens/s
            if stopping:
                stopped = True
                break
            if last and master:
                alert(f"{t.run_name}: finished", f"{step_done} steps, final loss {last_loss:.4f}. "
                      f"Last checkpoint: {last_good or 'none'}")
    finally:
        signal.signal(signal.SIGTERM, old_handler)
        if master:
            for proc in uploads:
                proc.wait()
            if t.ckpt_remote:
                rclone_push(log_path, t.ckpt_remote, wait=True)
            log_f.close()
            if run is not None:
                run.finish()
        if ddp:
            dist.destroy_process_group()
    return {"exit_code": exit_code, "stopped": stopped, "last_checkpoint": last_good, "last_loss": last_loss,
            "log": log_path}


def main(argv=None):
    p = argparse.ArgumentParser(description="Train a mygpt model (single process or torchrun).")
    p.add_argument("--config", required=True, help="YAML config, e.g. configs/smoke_cpu.yaml")
    p.add_argument("overrides", nargs="*", help="section.key=value, e.g. train.lr=6e-4 model.qk_norm=true")
    args = p.parse_args(argv)
    result = train(load_config(args.config, args.overrides))
    if result["exit_code"]:
        raise SystemExit(result["exit_code"])
    return result


if __name__ == "__main__":
    main()
