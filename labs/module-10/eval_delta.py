"""Did alignment break base skills? Compare lm-eval results of base, SFT and DPO models.

  python tools/eval_delta.py evals/base evals/sft evals/dpo --names base,sft,dpo --out evals/regression.md

Each argument is an lm-eval --output_path directory (or a results_*.json file); the newest
results_*.json under it is used. Prints a markdown table of acc_norm (acc if a task has no
acc_norm) per task, the change against the FIRST model, and flags a change as a regression when it
is below -2 combined standard errors (sqrt(se1^2 + se2^2)). Stdlib only.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os


def load(path: str) -> dict:
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "**", "results_*.json"), recursive=True), key=os.path.getmtime)
        if not files:
            raise SystemExit(f"no results_*.json under {path}")
        path = files[-1]
    with open(path, encoding="utf-8") as f:
        res = json.load(f)["results"]
    out = {}
    for task, m in res.items():
        key = "acc_norm" if "acc_norm,none" in m else "acc" if "acc,none" in m else None
        if key:
            out[task] = (key, m[f"{key},none"], m.get(f"{key}_stderr,none") or 0.0)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+")
    p.add_argument("--names", default=None, help="comma-separated labels, same order as paths")
    p.add_argument("--out", default=None, help="also write the table to this markdown file")
    args = p.parse_args(argv)
    names = args.names.split(",") if args.names else [os.path.basename(os.path.normpath(x)) for x in args.paths]
    runs = [load(x) for x in args.paths]
    tasks = sorted(set.intersection(*(set(r) for r in runs)))
    head = "| task | metric | " + " | ".join(names) + " | " + " | ".join(f"Δ {n}" for n in names[1:]) + " |"
    lines = [head, "|" + "---|" * (2 + 2 * len(names) - 1)]
    flags = []
    for t in tasks:
        key, base, se0 = runs[0][t]
        cells = [f"{r[t][1]:.3f} ± {r[t][2]:.3f}" for r in runs]
        deltas = []
        for n, r in zip(names[1:], runs[1:]):
            d = r[t][1] - base
            z = d / math.sqrt(se0 ** 2 + r[t][2] ** 2) if (se0 or r[t][2]) else 0.0
            mark = " ⚠" if z < -2 else ""
            if mark:
                flags.append(f"{n} {t}: {d:+.3f} ({z:.1f} SE)")
            deltas.append(f"{d:+.3f}{mark}")
        lines.append(f"| {t} | {key} | " + " | ".join(cells) + " | " + " | ".join(deltas) + " |")
    table = "\n".join(lines)
    summary = ("Regressions beyond 2 standard errors: " + "; ".join(flags)) if flags else \
        "No change beyond 2 standard errors against the first model."
    print(table + "\n\n" + summary)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(table + "\n\n" + summary + "\n")


if __name__ == "__main__":
    main()
