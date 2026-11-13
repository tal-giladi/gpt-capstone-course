"""Build the ablation table from the runs' log.jsonl files (stdlib only, Module 6).

    python tools/ablation_table.py runs/abl_* --queue tools/ablations.txt
    python tools/ablation_table.py runs/abl_* --queue tools/ablations.txt --control abl_qwen3 \
        --price-per-gpu-hour 2.69 > results/ablations.md

For every run it takes the metric (default val_bpb) at the SAME token count for all runs: the
largest evaluated step that every run reached (or --at-step). A run that has not reached it is
listed as unfinished, never compared at an earlier point. Then:

  delta  = run - mean(control runs)       (negative = better for val_bpb / val_loss)
  noise  = spread (max - min) of the metric over the control runs at the same step,
           or --noise (e.g. the 06.1 seed spread when the control is a single run)
  verdict: "better" / "worse" if |delta| > 2 x noise (and > --min-effect), else "within noise"

Controls: runs whose name starts with --control (default "abl_control"); a comma-separated list
of names also works. With one control run the noise is unknown and every verdict says so.
--queue reads NAME OVERRIDES lines (tools/ablations.txt) to fill the "change" column.
Cost: tokens / median tokens/s -> GPU-hours (1 GPU) x --price-per-gpu-hour.
Logs written again after a resume are de-duplicated (last record per step wins).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics
import sys


def load_log(path: str) -> list[dict]:
    if os.path.isdir(path):
        path = os.path.join(path, "log.jsonl")
    by_step: dict[int, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line cut off by a kill
            s = int(rec.get("step", 0))
            by_step[s] = {**by_step.get(s, {}), **rec}
    return [by_step[s] for s in sorted(by_step)]


def read_queue(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    macros, changes = {}, {}
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("#"):
                # commented-out queue lines still describe runs you may have done earlier
                cand = line.lstrip("#").strip()
                line = cand if re.match(r"^\S+(\s+\S+=\S+)+$", cand) else ""
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith("@"):
                key, _, val = line[1:].partition(" ")
                macros[key] = val.strip()
                continue
            words = line.split()
            parts = [macros.get(w, w) for w in words[1:]]
            changes[words[0]] = " ".join(parts) or "(none)"
    return changes


def metric_at(recs: list[dict], key: str, step: int) -> float | None:
    for r in recs:
        if r["step"] == step and isinstance(r.get(key), (int, float)) and math.isfinite(r[key]):
            return float(r[key])
    return None


def eval_steps(recs: list[dict], key: str) -> set[int]:
    return {r["step"] for r in recs if isinstance(r.get(key), (int, float)) and math.isfinite(r[key])}


def tokens_per_step(recs: list[dict]) -> float | None:
    for r in recs:
        if r.get("tokens") and r.get("step"):
            return r["tokens"] / r["step"]
    return None


def median_speed(recs: list[dict]) -> float | None:
    sp = [r["tokens_per_s"] for r in recs if r.get("tokens_per_s")]
    sp = sp[2:] if len(sp) > 4 else sp  # first records include compile time
    return statistics.median(sp) if sp else None


def build(runs: list[str], key: str, control: str, queue: dict[str, str], at_step: int | None,
          price: float | None, min_effect: float, noise_override: float | None = None) -> tuple[list[dict], dict]:
    data = {}
    for r in runs:
        name = os.path.basename(os.path.normpath(r))
        try:
            data[name] = load_log(r)
        except FileNotFoundError:
            print(f"[ablation_table] no log.jsonl in {r}, skipped", file=sys.stderr)
    if not data:
        raise SystemExit("no runs with a log.jsonl")
    ctrl_names = [n.strip() for n in control.split(",")] if "," in control else \
        [n for n in data if n == control or n.startswith(control)]
    ctrl_names = [n for n in ctrl_names if n in data]
    if not ctrl_names:
        raise SystemExit(f"no control run matches {control!r}; runs: {sorted(data)}")

    if at_step is None:
        # compare at the control's final evaluated step; every run shares the token budget and
        # schedule, so a run that did not reach it is reported as unfinished, not compared early
        at_step = max(eval_steps(data[ctrl_names[0]], key) or {0})
        if not at_step:
            raise SystemExit(f"control has no {key} records")
    ctrl_vals = [v for v in (metric_at(data[n], key, at_step) for n in ctrl_names) if v is not None]
    if not ctrl_vals:
        raise SystemExit(f"no control has {key} at step {at_step}")
    ctrl_mean = sum(ctrl_vals) / len(ctrl_vals)
    noise = (max(ctrl_vals) - min(ctrl_vals)) if len(ctrl_vals) > 1 else None
    if noise_override is not None:
        noise = noise_override  # e.g. the seed spread measured by the 06.1 controls
    lower_better = key != "hellaswag"

    rows = []
    for name, recs in sorted(data.items(), key=lambda kv: (kv[0] not in ctrl_names, kv[0])):
        v = metric_at(recs, key, at_step)
        tps = tokens_per_step(recs)
        speed = median_speed(recs)
        row = {"run": name, "change": queue.get(name, ""), "control": name in ctrl_names,
               "value": v, "delta": None, "verdict": "", "tokens": at_step * tps if tps else None,
               "tokens_per_s": speed, "gpu_h": None, "usd": None}
        last = max(r["step"] for r in recs)
        if tps and speed:
            row["gpu_h"] = last * tps / speed / 3600
            row["usd"] = row["gpu_h"] * price if price else None
        if v is None:
            row["verdict"] = f"unfinished (last step {last})"
        elif name in ctrl_names:
            row["delta"] = v - ctrl_mean
            row["verdict"] = "control"
        else:
            d = v - ctrl_mean
            row["delta"] = d
            if noise is None:
                row["verdict"] = "noise unknown (1 control)"
            elif abs(d) > max(2 * noise, min_effect):
                good = (d < 0) == lower_better
                row["verdict"] = "better" if good else "worse"
            else:
                row["verdict"] = "within noise"
        rows.append(row)
    info = {"metric": key, "at_step": at_step, "controls": ctrl_names, "control_mean": ctrl_mean,
            "noise": noise, "threshold": None if noise is None else max(2 * noise, min_effect)}
    return rows, info


def to_markdown(rows: list[dict], info: dict) -> str:
    def f(v, nd=4):
        return "-" if v is None else f"{v:.{nd}f}"
    tok = next((r["tokens"] for r in rows if r["tokens"]), None)
    head = [f"Metric **{info['metric']}** at step {info['at_step']:,}"
            + (f" ({tok / 1e9:.2f}B tokens)" if tok else "")
            + f"; control mean {info['control_mean']:.4f} over {len(info['controls'])} run(s)"
            + (f", noise {info['noise']:.4f}, effect threshold {info['threshold']:.4f}."
               if info["noise"] is not None else "; noise unknown (run a second control seed)."), ""]
    lines = ["| Run | Change vs control | " + info["metric"] + " | Delta | Verdict | Tokens/s | GPU-h | Cost |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        delta = "-" if r["delta"] is None else f"{r['delta']:+.4f}"
        speed = "-" if not r["tokens_per_s"] else f"{r['tokens_per_s']:,.0f}"
        usd = "-" if r["usd"] is None else f"${r['usd']:.2f}"
        lines.append(f"| {r['run']} | {r['change'] or '-'} | {f(r['value'])} | {delta} | {r['verdict']} | "
                     f"{speed} | {f(r['gpu_h'], 2)} | {usd} |")
    total_usd = sum(r["usd"] for r in rows if r["usd"])
    total_h = sum(r["gpu_h"] for r in rows if r["gpu_h"])
    foot = ["", f"Total: {total_h:.2f} GPU-h" + (f", ${total_usd:.2f}" if total_usd else "") + " of training time "
            "(excludes box setup and idle time)."]
    return "\n".join(head + lines + foot) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="run dirs (each with log.jsonl); globs allowed")
    p.add_argument("--queue", help="queue file with NAME OVERRIDES lines (fills the change column)")
    p.add_argument("--control", default="abl_control", help="control name prefix, or comma-separated names")
    p.add_argument("--metric", default="val_bpb", choices=["val_bpb", "val_loss", "hellaswag"])
    p.add_argument("--at-step", type=int, help="compare at this step (default: the control's last eval)")
    p.add_argument("--min-effect", type=float, default=0.0, help="smallest delta you care about")
    p.add_argument("--noise", type=float, help="use this seed spread instead of the controls' (e.g. from 06.1)")
    p.add_argument("--price-per-gpu-hour", type=float, help="$ per GPU-hour of the ablation box")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    runs = []
    for r in a.runs:
        runs += sorted(glob.glob(r)) or [r]
    rows, info = build(runs, a.metric, a.control, read_queue(a.queue), a.at_step, a.price_per_gpu_hour, a.min_effect, a.noise)
    if a.json:
        print(json.dumps({"info": info, "rows": rows}, indent=2))
    else:
        print(to_markdown(rows, info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
