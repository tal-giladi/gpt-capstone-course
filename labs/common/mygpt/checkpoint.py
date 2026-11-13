"""Checkpoints: atomic save, keep-last-N, find latest, optional rclone mirror to a bucket.

File name: OUT_DIR/ckpt_00012000.pt (step = optimizer steps completed). A checkpoint holds
model, optimizer, step, data-loader position, RNG states and the full config.
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess

import torch

_PAT = re.compile(r"ckpt_(\d{8})\.pt$")


def ckpt_path(out_dir: str, step: int) -> str:
    return os.path.join(out_dir, f"ckpt_{step:08d}.pt")


def list_checkpoints(out_dir: str) -> list[str]:
    return sorted(p for p in glob.glob(os.path.join(out_dir, "ckpt_*.pt")) if _PAT.search(p))


def latest_checkpoint(out_dir: str) -> str | None:
    ckpts = list_checkpoints(out_dir)
    return ckpts[-1] if ckpts else None


def save_checkpoint(out_dir: str, step: int, state: dict, keep_last: int = 3) -> str:
    """Write to a temp file then rename, so a kill mid-write never leaves a corrupt 'latest'."""
    os.makedirs(out_dir, exist_ok=True)
    path = ckpt_path(out_dir, step)
    tmp = path + ".tmp"
    torch.save({**state, "step": step}, tmp)
    os.replace(tmp, path)
    if keep_last > 0:
        for old in list_checkpoints(out_dir)[:-keep_last]:
            os.remove(old)
    return path


def load_checkpoint(path: str, map_location="cpu") -> dict:
    # weights_only=False: the file also holds RNG states / config dicts we wrote ourselves.
    return torch.load(path, map_location=map_location, weights_only=False)


# ---------------------------------------------------------------- rclone mirror
def have_rclone() -> bool:
    return shutil.which("rclone") is not None


def rclone_push(path: str, remote: str, wait: bool = False):
    """`rclone copy FILE REMOTE` in the background (returns the Popen) or blocking."""
    if not remote:
        return None
    if not have_rclone():
        print(f"[ckpt] WARNING: ckpt_remote={remote!r} set but rclone is not installed; skipping upload")
        return None
    cmd = ["rclone", "copy", path, remote, "--retries", "5", "--low-level-retries", "20"]
    proc = subprocess.Popen(cmd)
    if wait:
        proc.wait()
        if proc.returncode != 0:
            print(f"[ckpt] WARNING: rclone exited {proc.returncode} uploading {path}")
    return proc


def rclone_pull_latest(remote: str, out_dir: str) -> str | None:
    """Download the newest ckpt_*.pt from REMOTE into out_dir (used when the box is new)."""
    if not remote or not have_rclone():
        return None
    res = subprocess.run(["rclone", "lsf", remote, "--include", "ckpt_*.pt"], capture_output=True, text=True)
    names = sorted(n.strip() for n in res.stdout.splitlines() if _PAT.search(n.strip()))
    if res.returncode != 0 or not names:
        return None
    os.makedirs(out_dir, exist_ok=True)
    print(f"[ckpt] pulling {names[-1]} from {remote}")
    subprocess.run(["rclone", "copy", f"{remote.rstrip('/')}/{names[-1]}", out_dir], check=True)
    return os.path.join(out_dir, names[-1])
