"""Compare lm-eval-harness results of several models in one markdown table, with standard errors and
paired bootstrap confidence intervals for the differences that matter (lessons 08.1-08.3).

    python tools/eval_table.py evals/main evals/baseline evals/gpt2 evals/gpt2-medium \
        --names mine-350m,baseline-124m,gpt2-124m,gpt2-355m \
        --compare mine-350m:gpt2-355m --compare baseline-124m:gpt2-124m --out reports/eval_table.md

Each positional argument is an lm-eval --output_path folder; the newest results_*.json below it is
used, and the samples_<task>_*.jsonl files next to it if lm-eval ran with --log_samples.

Metric per task (override with --metric task=metric): acc_norm where lm-eval reports it
(hellaswag, arc_easy, arc_challenge, piqa), acc for winogrande and lambada_openai (plus LAMBADA
perplexity as an extra column). Values are shown in points (x100) +- lm-eval's standard error.

--compare A:B prints, per task, A - B with a 95% interval:
  paired bootstrap over the same items (needs --log_samples for both; 10,000 resamples), or
  if samples are missing, an unpaired normal interval from the two standard errors (wider).
"Yes" in the significant column means the interval excludes 0.

The script also checks that all runs used the same settings (n-shot, --limit, lm-eval version,
task versions) and prints a warning line for each difference: numbers from different settings
must not share a table.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

DEFAULT_METRIC = {"hellaswag": "acc_norm", "arc_easy": "acc_norm", "arc_challenge": "acc_norm",
                  "piqa": "acc_norm", "winogrande": "acc", "lambada_openai": "acc", "openbookqa": "acc_norm",
                  "sciq": "acc_norm"}


def newest(pattern: str) -> str | None:
    files = sorted(glob.glob(pattern, recursive=True), key=os.path.getmtime)
    return files[-1] if files else None


def load_run(folder: str) -> dict:
    res = newest(os.path.join(folder, "**", "results_*.json"))
    if not res:
        raise SystemExit(f"no results_*.json under {folder}")
    with open(res, encoding="utf-8") as f:
        d = json.load(f)
    d["_path"] = res
    d["_dir"] = os.path.dirname(res)
    return d


def metric_for(task: str, run: dict, overrides: dict) -> str:
    m = overrides.get(task) or DEFAULT_METRIC.get(task)
    r = run["results"].get(task, {})
    if m and f"{m},none" in r:
        return m
    for cand in ("acc_norm", "acc", "exact_match"):
        if f"{cand},none" in r:
            return cand
    return m or "acc"


def value(run: dict, task: str, metric: str) -> tuple[float | None, float | None]:
    r = run["results"].get(task)
    if not r:
        return None, None
    v, se = r.get(f"{metric},none"), r.get(f"{metric}_stderr,none")
    return (v if isinstance(v, (int, float)) else None), (se if isinstance(se, (int, float)) else None)


def samples(run: dict, task: str, metric: str) -> dict[int, float] | None:
    f = newest(os.path.join(run["_dir"], f"samples_{task}_*.jsonl"))
    if not f:
        return None
    out = {}
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                s = json.loads(line)
                if metric in s:
                    out[int(s["doc_id"])] = float(s[metric])
    return out or None


def paired_bootstrap(a: dict[int, float], b: dict[int, float], iters: int, seed: int) -> tuple[float, float, float, int]:
    import numpy as np
    ids = sorted(set(a) & set(b))
    d = np.array([a[i] - b[i] for i in ids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    means, n, done = [], len(d), 0
    while done < iters:  # chunks keep memory small for 10k items x 10k resamples
        k = min(200, iters - done)
        idx = rng.integers(0, n, size=(k, n))
        means.append(d[idx].mean(axis=1))
        done += k
    m = np.concatenate(means)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), n


def settings(run: dict) -> dict:
    cfg = run.get("config", {})
    return {"n-shot": json.dumps(run.get("n-shot", {}), sort_keys=True), "limit": cfg.get("limit"),
            "lm_eval": run.get("lm_eval_version"), "versions": json.dumps(run.get("versions", {}), sort_keys=True)}


def pts(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folders", nargs="+", help="lm-eval --output_path folders, one per model")
    p.add_argument("--names", help="comma list of model names (default: folder names)")
    p.add_argument("--metric", action="append", default=[], help="task=metric override, e.g. arc_easy=acc")
    p.add_argument("--compare", action="append", default=[], help="A:B, report A minus B per task")
    p.add_argument("--iters", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="also write the markdown to this file")
    a = p.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # the ± sign on a Windows console
    except (AttributeError, ValueError):
        pass

    names = a.names.split(",") if a.names else [os.path.basename(os.path.normpath(f)) for f in a.folders]
    if len(names) != len(a.folders):
        raise SystemExit("--names must have one name per folder")
    runs = {n: load_run(f) for n, f in zip(names, a.folders)}
    overrides = dict(m.split("=", 1) for m in a.metric)
    tasks: list[str] = []
    for r in runs.values():
        tasks += [t for t in r["results"] if t not in tasks]

    lines = []
    first = runs[names[0]]
    cfg = first.get("config", {})
    lines.append(f"lm-eval {first.get('lm_eval_version', '?')}, n-shot "
                 f"{', '.join(sorted(set(str(v) for v in first.get('n-shot', {}).values())))}, "
                 f"model_args of first run: `{json.dumps(cfg.get('model_args'))}`, limit {cfg.get('limit')}.")
    base = settings(first)
    for n in names[1:]:
        s = settings(runs[n])
        for k in base:
            if s[k] != base[k]:
                msg = f"WARNING: {n} differs from {names[0]} in {k}: {s[k]} vs {base[k]}"
                lines.append(msg)
                print(msg, file=sys.stderr)
    if any(r.get("config", {}).get("limit") for r in runs.values()):
        lines.append("WARNING: at least one run used --limit; these are not full-benchmark numbers.")
    lines.append("")

    head = "| Task (metric) | items | " + " | ".join(names) + " |"
    lines += [head, "|" + "---|" * (len(names) + 2)]
    avg = {n: [] for n in names}
    for t in tasks:
        m = metric_for(t, first if t in first["results"] else next(r for r in runs.values() if t in r["results"]), overrides)
        nitems = next((r.get("n-samples", {}).get(t, {}).get("effective") for r in runs.values()
                       if r.get("n-samples", {}).get(t)), None)
        cells = []
        for n in names:
            v, se = value(runs[n], t, m)
            cells.append("-" if v is None else f"{pts(v)} ± {pts(se)}")
            if v is not None and all(t in r["results"] for r in runs.values()):
                avg[n].append(v)
        lines.append(f"| {t} ({m}) | {nitems or '-'} | " + " | ".join(cells) + " |")
        if t == "lambada_openai":
            cells = []
            for n in names:
                v, se = value(runs[n], t, "perplexity")
                cells.append("-" if v is None else f"{v:.1f} ± {se:.1f}" if se is not None else f"{v:.1f}")
            lines.append(f"| {t} (perplexity, lower is better) | {nitems or '-'} | " + " | ".join(cells) + " |")
    lines.append("| **Average accuracy (tasks all models have)** | | " +
                 " | ".join(f"**{pts(sum(v) / len(v))}**" if v else "-" for v in avg.values()) + " |")
    lines.append("")
    lines.append("Values in points (accuracy x 100) ± lm-eval's standard error.")

    for comp in a.compare:
        A, B = comp.split(":")
        lines += ["", f"### {A} minus {B}", "", "| Task (metric) | difference | 95% interval | method | significant |",
                  "|---|---|---|---|---|"]
        for t in tasks:
            m = metric_for(t, runs[A], overrides)
            sa, sb = samples(runs[A], t, m), samples(runs[B], t, m)
            if sa and sb:
                d, lo, hi, n = paired_bootstrap(sa, sb, a.iters, a.seed)
                method = f"paired bootstrap, {n} items"
            else:
                va, ea = value(runs[A], t, m)
                vb, eb = value(runs[B], t, m)
                if va is None or vb is None:
                    continue
                d = va - vb
                half = 1.96 * math.sqrt((ea or 0) ** 2 + (eb or 0) ** 2)
                lo, hi, method = d - half, d + half, "unpaired, from stderr"
            sig = "yes" if (lo > 0 or hi < 0) else "no"
            lines.append(f"| {t} ({m}) | {100 * d:+.1f} | [{100 * lo:+.1f}, {100 * hi:+.1f}] | {method} | {sig} |")

    md = "\n".join(lines) + "\n"
    print(md)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
