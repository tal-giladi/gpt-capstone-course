"""Pre-flight checks before an expensive launch (lesson 07.1). Run on the node, from the repo root,
venv active, after setup_box.sh and after pulling the shards:

    python tools/preflight.py --config configs/main_350m.yaml --gpus 8 --tag recipe-v1 \
        --budget-left 336 --estimate 187

Prints one PASS / WARN / FAIL line per check and exits 1 if anything FAILs. It changes nothing
except one tiny test file it copies to <bucket root>/tmp/ to prove the bucket is writable.

Checks: git (clean tree, HEAD at or after the tag), config (ckpt_remote, wandb, run name,
out_dir, batch divisibility, tokens vs data), data (shards present and complete, tokenizer and
vocab match, same shard count as the bucket), bucket (readable and writable), W&B key, GPUs
(count, model, NVLink, nothing else running), disk space, the stop guards (autoshutdown and a
hard stop), and the money rule from 06.4 (budget left >= 1.25 x estimate).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

RESULTS: list[tuple[str, str, str]] = []


def add(level: str, name: str, msg: str) -> None:
    RESULTS.append((level, name, msg))
    print(f"{level:<4}  {name:<12} {msg}", flush=True)


def run(cmd: list[str] | str, timeout: int = 60) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return 127, str(e)


def load_cfg(path: str, overrides: list[str]):
    from mygpt.config import load_config  # the project package, installed in the venv
    return load_config(path, overrides)


# ------------------------------------------------------------------ checks
def check_git(tag: str | None) -> None:
    code, out = run(["git", "status", "--porcelain", "--untracked-files=no"])
    if code:
        add("FAIL", "git", f"not a git repo or git missing: {out[:120]}")
        return
    if out:
        add("FAIL", "git", f"uncommitted changes to tracked files ({len(out.splitlines())}); commit or revert first")
    else:
        add("PASS", "git", "working tree clean")
    _, head = run(["git", "rev-parse", "--short", "HEAD"])
    if tag:
        code, _ = run(["git", "merge-base", "--is-ancestor", tag, "HEAD"])
        if code == 0:
            _, desc = run(["git", "describe", "--tags", "--always"])
            add("PASS", "git", f"HEAD {head} contains tag {tag} ({desc}); write this commit into budget.md")
        else:
            add("FAIL", "git", f"tag {tag} missing or not an ancestor of HEAD {head}: launch from the frozen recipe")


def check_config(c, gpus: int) -> None:
    t, m = c.train, c.model
    if not t.ckpt_remote:
        add("FAIL", "config", "train.ckpt_remote is empty: a dead node would lose the run")
    elif not t.ckpt_remote.rstrip("/").endswith("/" + t.run_name):
        add("WARN", "config", f"ckpt_remote {t.ckpt_remote} does not end in /{t.run_name}")
    else:
        add("PASS", "config", f"ckpt_remote {t.ckpt_remote}")
    add("PASS" if t.wandb_project else "WARN", "config",
        f"wandb_project {t.wandb_project!r}" + ("" if t.wandb_project else " (no live curves; log.jsonl only)"))
    per = t.micro_batch_size * t.seq_len * gpus
    if t.total_batch_tokens % per:
        add("FAIL", "config", f"total_batch_tokens {t.total_batch_tokens} not divisible by micro*seq*gpus = {per}")
    else:
        add("PASS", "config", f"{t.train_tokens // t.total_batch_tokens} steps, accumulation "
                              f"{t.total_batch_tokens // per} on {gpus} GPUs, ckpt every {t.ckpt_every} steps")
    ck = sorted(glob.glob(os.path.join(t.out_dir, "ckpt_*.pt")))
    if ck:
        add("WARN", "config", f"{t.out_dir} already holds {len(ck)} checkpoint(s), newest {os.path.basename(ck[-1])}: "
                              "this launch RESUMES; use a new out_dir for a fresh run")
    else:
        add("PASS", "config", f"out_dir {t.out_dir} is empty (fresh start, or auto-pull from the bucket)")
    if t.compile is False:
        add("WARN", "config", "compile: false (slower than your benchmark assumed)")


def shard_complete(path: str) -> bool:
    import numpy as np
    head = np.fromfile(path, dtype=np.int32, count=4)
    return len(head) == 4 and head[0] == 20261113 and os.path.getsize(path) == 1024 + 2 * int(head[2])


def check_data(c, remote_root: str | None) -> None:
    t, m = c.train, c.model
    d = t.data_dir
    if not os.path.isdir(d):
        add("FAIL", "data", f"{d} missing: bash scripts/sync.sh pull-tok <name> {d}")
        return
    meta_p = os.path.join(d, "meta.json")
    meta = json.load(open(meta_p, encoding="utf-8")) if os.path.exists(meta_p) else {}
    train = sorted(glob.glob(os.path.join(d, "train_*.bin")))
    val = sorted(glob.glob(os.path.join(d, "val_*.bin")))
    bad = [os.path.basename(p) for p in train + val if not shard_complete(p)]
    if not train or not val or bad:
        add("FAIL", "data", f"{len(train)} train / {len(val)} val shards; incomplete: {bad[:5]}")
    else:
        add("PASS", "data", f"{len(train)} train + {len(val)} val shards, all complete")
    if meta:
        tok, vocab = meta.get("tokenizer"), meta.get("vocab_size", 0)
        lvl = "PASS" if vocab <= m.vocab_size and vocab > m.vocab_size - 128 else "FAIL"
        add(lvl, "data", f"tokenizer {tok}, vocab {vocab} vs model.vocab_size {m.vocab_size}")
        have = meta.get("train_tokens") or 0
        if have:
            ratio = t.train_tokens / have
            lvl = "PASS" if ratio <= 0.99 else ("WARN" if ratio <= 1.0 else "FAIL")
            add(lvl, "data", f"train_tokens {t.train_tokens / 1e9:.2f}B = {ratio:.1%} of the {have / 1e9:.2f}B "
                             "available" + ("" if ratio <= 1.0 else " (would repeat data)"))
    else:
        add("FAIL", "data", f"{meta_p} missing")
    if remote_root:
        name = os.path.basename(os.path.normpath(d))
        code, out = run(["rclone", "lsf", f"{remote_root}/tok/{name}", "--include", "train_*.bin"], timeout=120)
        if code == 0:
            n = len([x for x in out.splitlines() if x.strip()])
            add("PASS" if n == len(train) else "FAIL", "data", f"bucket tok/{name}/ has {n} train shards, local {len(train)}")
        else:
            add("WARN", "data", f"could not list {remote_root}/tok/{name}: {out[:100]}")


def check_bucket(remote_root: str | None, ckpt_remote: str) -> None:
    if not shutil.which("rclone"):
        add("FAIL", "bucket", "rclone not installed (setup_box.sh did not finish)")
        return
    root = remote_root or ckpt_remote.rsplit("/ckpt/", 1)[0]
    code, out = run(["rclone", "lsf", root, "--max-depth", "1"], timeout=60)
    if code:
        add("FAIL", "bucket", f"cannot list {root}: {out[:120]}")
        return
    code, out = run(["rclone", "lsf", ckpt_remote], timeout=60)
    existing = [x for x in out.splitlines() if x.startswith("ckpt_")] if code == 0 else []
    if existing:
        add("WARN", "bucket", f"{ckpt_remote} already has {len(existing)} checkpoint(s): a fresh out_dir will pull and "
                              "RESUME the newest one")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(f"preflight {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    dest = f"{root}/tmp/"
    code, out = run(["rclone", "copy", f.name, dest], timeout=60)
    os.unlink(f.name)
    add("PASS" if code == 0 else "FAIL", "bucket", f"write test to {dest}" + ("" if code == 0 else f": {out[:120]}"))


def check_wandb(c) -> None:
    if not c.train.wandb_project or getattr(c.train, "tracker", "wandb") != "wandb":
        return  # no tracker, or Trackio (uses your HF login, no W&B key)
    key = os.environ.get("WANDB_API_KEY", "")
    add("PASS" if len(key) >= 40 else "FAIL", "wandb",
        "WANDB_API_KEY set" if len(key) >= 40 else "WANDB_API_KEY missing or too short (template secret?)")


def check_gpus(gpus: int, model_substr: str) -> None:
    code, out = run(["nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu,temperature.gpu",
                     "--format=csv,noheader,nounits"])
    if code:
        add("FAIL", "gpus", f"nvidia-smi failed: {out[:100]}")
        return
    rows = [r.split(", ") for r in out.splitlines() if r.strip()]
    names = {r[0] for r in rows}
    ok = len(rows) == gpus and all(model_substr in n for n in names)
    add("PASS" if ok else "FAIL", "gpus", f"{len(rows)} x {', '.join(sorted(names))} (want {gpus} x {model_substr})")
    busy = [i for i, r in enumerate(rows) if float(r[1]) > 1024 or float(r[2]) > 5]
    add("FAIL" if busy else "PASS", "gpus",
        f"GPUs {busy} already in use: another process is running" if busy else "all GPUs idle and empty")
    code, topo = run(["nvidia-smi", "topo", "-m"])
    if gpus > 1 and code == 0:
        nv = "NV" in topo.split("\n", 2)[1] if "\n" in topo else False
        add("PASS" if nv else "FAIL", "gpus", "NVLink between GPUs (NVxx in topo -m)" if nv
            else "no NVLink in nvidia-smi topo -m: PCIe node, all-reduce will be slow")


def check_disk(path: str, min_gb: float) -> None:
    os.makedirs(path, exist_ok=True)
    free = shutil.disk_usage(path).free / 1e9
    add("PASS" if free >= min_gb else "FAIL", "disk", f"{free:.0f} GB free at {path} (need >= {min_gb:.0f})")


def check_guards() -> None:
    if os.name == "nt":
        add("WARN", "guards", "not checked on Windows")
        return
    code, ps = run(["ps", "-eo", "args"])
    lines = ps.splitlines()
    auto = any("autoshutdown.sh" in line for line in lines)
    hard = any("sleep" in line and ("stop" in line or "terminate" in line or "shutdown" in line) for line in lines)
    add("PASS" if auto else "FAIL", "guards", "autoshutdown.sh running" if auto else "autoshutdown.sh NOT running")
    add("PASS" if hard else "FAIL", "guards", "hard stop (sleep ...; stop) armed" if hard else "no hard stop armed")


def check_money(left: float | None, estimate: float | None) -> None:
    if left is None or estimate is None:
        add("WARN", "budget", "pass --budget-left and --estimate to apply the 06.4 go rule")
        return
    r = left / estimate
    lvl = "PASS" if r >= 1.25 else ("WARN" if r >= 1.10 else "FAIL")
    add(lvl, "budget", f"money left ${left:.0f} = {r:.2f} x estimate ${estimate:.0f} "
                       f"({'GO' if r >= 1.25 else 'cut tokens or use a cheaper host' if r >= 1.10 else 'shrink'})")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("overrides", nargs="*", help="the same section.key=value overrides you will launch with")
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--gpu-model", default="H100", help="substring every GPU name must contain")
    p.add_argument("--tag", default="recipe-v1")
    p.add_argument("--remote-root", default=os.environ.get("MYGPT_REMOTE"), help="bucket root (default $MYGPT_REMOTE)")
    p.add_argument("--min-free-gb", type=float, default=60)
    p.add_argument("--budget-left", type=float)
    p.add_argument("--estimate", type=float)
    p.add_argument("--skip", default="", help="comma list of checks to skip, e.g. gpus,guards (laptop test)")
    a = p.parse_args(argv)
    skip = set(filter(None, a.skip.split(",")))

    c = load_cfg(a.config, a.overrides)
    if "git" not in skip:
        check_git(a.tag)
    check_config(c, a.gpus)
    if "data" not in skip:
        check_data(c, None if "bucket" in skip else a.remote_root)
    if "bucket" not in skip and c.train.ckpt_remote:
        check_bucket(a.remote_root, c.train.ckpt_remote)
    check_wandb(c)
    if "gpus" not in skip:
        check_gpus(a.gpus, a.gpu_model)
    check_disk(c.train.out_dir, a.min_free_gb)
    if "guards" not in skip:
        check_guards()
    check_money(a.budget_left, a.estimate)

    fails = sum(1 for r in RESULTS if r[0] == "FAIL")
    warns = sum(1 for r in RESULTS if r[0] == "WARN")
    print(f"\n{'READY' if not fails else 'NOT READY'}: {fails} FAIL, {warns} WARN")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
