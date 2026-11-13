"""Summarise a training run from its log.jsonl (stdlib only).

    python tools/run_summary.py runs/baseline_124m
    python tools/run_summary.py runs/baseline_124m --gpus 8 --single-gpu-tps 480000 \
        --price-per-gpu-hour 4.39 --target-tokens 10e9
    python tools/run_summary.py runs/drill --compare runs/drill_ref      # resumed vs uninterrupted
    python tools/run_summary.py runs/baseline_124m --markdown >> results.md

Accepts a run directory (reads DIR/log.jsonl) or a log.jsonl path. After a resume, mygpt.train
logs the steps between the last checkpoint and the crash a second time; this script keeps the
LAST record for each step (the one the surviving run wrote) and reports how many resumes it saw.

Prints: steps and tokens reached, last train loss, last and best val_loss / val_bpb, last and best
HellaSwag, median tokens/s (node) and MFU, and with --gpus / --single-gpu-tps the per-GPU speed
and multi-GPU scaling efficiency; with --target-tokens and --price-per-gpu-hour the projected
wall clock and cost of the whole run at the measured speed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys


def load_log(path: str) -> tuple[list[dict], int]:
    """Return (records sorted by step, last record per step wins, number of resumes seen)."""
    if os.path.isdir(path):
        path = os.path.join(path, "log.jsonl")
    by_step: dict[int, dict] = {}
    resumes, prev = 0, 0
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"[run_summary] skipping unreadable line {n} of {path} (killed mid-write?)", file=sys.stderr)
                continue
            step = int(rec.get("step", 0))
            if step < prev:
                resumes += 1
            prev = step
            # a step logged again after a resume: newer values replace older ones field by field
            by_step[step] = {**by_step.get(step, {}), **rec}
    if not by_step:
        raise SystemExit(f"no records in {path}")
    return [by_step[s] for s in sorted(by_step)], resumes


def tokens_per_step(recs: list[dict]) -> float | None:
    for r in recs:
        if r.get("tokens") and r.get("step"):
            return r["tokens"] / r["step"]
    return None


def last_with(recs: list[dict], key: str) -> dict | None:
    for r in reversed(recs):
        if key in r and r[key] is not None and not (isinstance(r[key], float) and math.isnan(r[key])):
            return r
    return None


def best_with(recs: list[dict], key: str, lower_is_better: bool) -> dict | None:
    vals = [r for r in recs if isinstance(r.get(key), (int, float)) and math.isfinite(r[key])]
    if not vals:
        return None
    return (min if lower_is_better else max)(vals, key=lambda r: r[key])


def summarize(path: str, gpus: int | None = None, single_gpu_tps: float | None = None,
              price: float | None = None, target_tokens: float | None = None) -> dict:
    recs, resumes = load_log(path)
    tps_step = tokens_per_step(recs)
    last = recs[-1]
    out: dict = {"run": path.rstrip("/\\"), "records": len(recs), "resumes": resumes,
                 "last_step": last["step"],
                 "tokens": round(last["step"] * tps_step) if tps_step else None}
    r = last_with(recs, "loss")
    out["last_loss"] = r and round(r["loss"], 4)
    for key, low in (("val_loss", True), ("val_bpb", True), ("hellaswag", False)):
        lr, br = last_with(recs, key), best_with(recs, key, low)
        out[f"last_{key}"] = lr and round(lr[key], 4)
        out[f"last_{key}_step"] = lr and lr["step"]
        out[f"best_{key}"] = br and round(br[key], 4)
        out[f"best_{key}_step"] = br and br["step"]
    # speed: drop the first two speed records, they include torch.compile and warm-up time
    speed = [rr["tokens_per_s"] for rr in recs if rr.get("tokens_per_s")]
    speed = speed[2:] if len(speed) > 4 else speed
    if speed:
        node = statistics.median(speed)
        out["median_tokens_per_s"] = round(node)
        mfus = [rr["mfu"] for rr in recs[1:] if rr.get("mfu")]
        out["median_mfu"] = round(statistics.median(mfus), 4) if mfus else None
        if gpus:
            out["tokens_per_s_per_gpu"] = round(node / gpus)
            if single_gpu_tps:
                out["scaling_efficiency"] = round(node / (gpus * single_gpu_tps), 3)
        if target_tokens:
            hours = target_tokens / node / 3600
            out["projected_train_hours"] = round(hours, 2)
            if price and gpus:
                out["projected_cost_usd"] = round(hours * gpus * price, 2)
    return out


def compare(a: str, b: str) -> dict:
    """Train-loss difference at steps logged in both runs (resume drill check)."""
    ra, _ = load_log(a)
    rb, _ = load_log(b)
    la = {r["step"]: r["loss"] for r in ra if "loss" in r}
    lb = {r["step"]: r["loss"] for r in rb if "loss" in r}
    common = sorted(set(la) & set(lb))
    if not common:
        return {"common_steps": 0}
    diffs = [abs(la[s] - lb[s]) for s in common]
    worst = common[diffs.index(max(diffs))]
    return {"common_steps": len(common), "max_abs_loss_diff": round(max(diffs), 5),
            "at_step": worst, "mean_abs_loss_diff": round(sum(diffs) / len(diffs), 5),
            "final_step": common[-1], "final_loss_a": round(la[common[-1]], 4),
            "final_loss_b": round(lb[common[-1]], 4)}


def fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:,.4f}" if abs(v) < 100 else f"{v:,.0f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def markdown(s: dict) -> str:
    rows = [("Steps", s["last_step"]), ("Tokens", s["tokens"]), ("Resumes seen in log", s["resumes"]),
            ("Last train loss", s["last_loss"]),
            ("Final val loss", s["last_val_loss"]), ("Best val loss (step)", f"{fmt(s['best_val_loss'])} ({fmt(s['best_val_loss_step'])})"),
            ("Final val bpb", s["last_val_bpb"]),
            ("Final HellaSwag acc_norm (in-training)", s["last_hellaswag"]),
            ("Best HellaSwag (step)", f"{fmt(s['best_hellaswag'])} ({fmt(s['best_hellaswag_step'])})"),
            ("Median tokens/s (node)", s.get("median_tokens_per_s")), ("Median MFU", s.get("median_mfu"))]
    for k, label in (("tokens_per_s_per_gpu", "Tokens/s per GPU"), ("scaling_efficiency", "Scaling efficiency"),
                     ("projected_train_hours", "Projected training hours"), ("projected_cost_usd", "Projected cost (USD)")):
        if k in s:
            rows.append((label, s[k]))
    lines = [f"### Run `{os.path.basename(s['run']) or s['run']}`", "", "| Metric | Value |", "|---|---|"]
    lines += [f"| {k} | {fmt(v)} |" for k, v in rows]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="run dir (with log.jsonl) or a log.jsonl file")
    p.add_argument("--gpus", type=int, help="GPUs the run used (for per-GPU speed and scaling)")
    p.add_argument("--single-gpu-tps", type=float, help="per-GPU tokens/s from scripts/bench.py (lesson 04.3)")
    p.add_argument("--price-per-gpu-hour", type=float, help="$ per GPU-hour you pay")
    p.add_argument("--target-tokens", type=float, help="planned training tokens, e.g. 10e9")
    p.add_argument("--compare", help="second run dir/log: report train-loss differences at common steps")
    p.add_argument("--markdown", action="store_true", help="print a markdown table for results.md")
    p.add_argument("--json", action="store_true", help="print JSON instead of text")
    a = p.parse_args(argv)

    s = summarize(a.run, a.gpus, a.single_gpu_tps, a.price_per_gpu_hour, a.target_tokens)
    if a.compare:
        s["compare"] = compare(a.run, a.compare)
    if a.json:
        print(json.dumps(s, indent=2))
    elif a.markdown:
        print(markdown(s))
    else:
        width = max(len(k) for k in s)
        for k, v in s.items():
            print(f"{k:<{width}}  {fmt(v) if not isinstance(v, dict) else json.dumps(v)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
