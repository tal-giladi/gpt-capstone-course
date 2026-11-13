"""Training-curve plots for the evaluation report, as standalone SVG files (stdlib only, lesson 08.3).

    python tools/plot_curves.py runs/main_350m runs/baseline_124m --labels "my 350M,baseline 124M" \
        --out-dir reports/figures --hline "hellaswag:GPT-2 124M (Karpathy eval)=0.2945" \
        --hline "hellaswag:GPT-2 medium (Karpathy eval)=0.3752"

Each run is a run dir with log.jsonl (or the log file itself), e.g. pulled with
`bash scripts/sync.sh pull-ckpt main_350m runs/main_350m` or just the log from the bucket.
Steps logged twice after a resume keep their last record (as tools/run_summary.py does).

Writes one SVG per metric that at least one run has, with training tokens (billions) on the x axis:
  val_bpb.svg      validation bits per byte (comparable across tokenizers)
  hellaswag.svg    in-training HellaSwag acc_norm (mygpt scoring; not the lm-eval number)
  train_loss.svg   train loss in nats/token (compare only runs with the same tokenizer)
  tokens_per_s.svg node throughput
  lr.svg           learning rate (shows warmup, stable phase and decay)
--hline METRIC:LABEL=VALUE draws a dashed reference line on that metric's plot.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import sys

COLORS = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2"]
METRICS = [("val_bpb", "Validation bits per byte (lower is better)"),
           ("hellaswag", "HellaSwag acc_norm, in-training eval (higher is better)"),
           ("loss", "Train loss, nats per token (same tokenizer only)"),
           ("tokens_per_s", "Node throughput, tokens per second"),
           ("lr", "Learning rate (first optimizer group)")]
FILES = {"loss": "train_loss.svg"}


def load(path: str) -> list[dict]:
    if os.path.isdir(path):
        path = os.path.join(path, "log.jsonl")
    by_step: dict[int, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "skip_batches" in r:
                continue
            s = int(r.get("step", 0))
            by_step[s] = {**by_step.get(s, {}), **r}
    recs = [by_step[s] for s in sorted(by_step)]
    tps = next((r["tokens"] / r["step"] for r in recs if r.get("tokens") and r.get("step")), None)
    for r in recs:
        r["_x"] = r["step"] * tps / 1e9 if tps else r["step"]
    return recs


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=10 * mag)
    start = math.ceil(lo / step) * step
    out, v = [], start
    while v <= hi + 1e-12:
        out.append(round(v, 10))
        v += step
    return out


def fmt(v: float) -> str:
    if v == 0:
        return "0"
    if abs(v) >= 1e6:
        return f"{v / 1e6:g}M"
    if abs(v) >= 1e3:
        return f"{v / 1e3:g}k"
    if abs(v) < 1e-2:
        return f"{v:.0e}"
    return f"{v:g}"


def svg_plot(series: list[tuple[str, list[tuple[float, float]]]], title: str, xlabel: str,
             hlines: list[tuple[str, float]], log_y: bool = False) -> str:
    W, H, L, R, T, B = 760, 440, 70, 20, 40, 60
    pw, ph = W - L - R, H - T - B
    xs = [x for _, pts in series for x, _ in pts]
    ys = [y for _, pts in series for _, y in pts] + [v for _, v in hlines]
    x0, x1 = 0.0, max(xs) * 1.02
    y0, y1 = min(ys), max(ys)
    pad = (y1 - y0) * 0.05 or abs(y1) * 0.05 or 1
    y0, y1 = y0 - pad, y1 + pad

    def X(x):
        return L + (x - x0) / (x1 - x0) * pw

    def Y(y):
        return T + (1 - (y - y0) / (y1 - y0)) * ph

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
         f'font-family="Helvetica, Arial, sans-serif" font-size="12">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         f'<text x="{W / 2}" y="22" text-anchor="middle" font-size="15" fill="#111">{html.escape(title)}</text>']
    for t in nice_ticks(y0, y1):
        o.append(f'<line x1="{L}" x2="{L + pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="#e5e7eb"/>')
        o.append(f'<text x="{L - 6}" y="{Y(t) + 4:.1f}" text-anchor="end" fill="#374151">{fmt(t)}</text>')
    for t in nice_ticks(x0, x1):
        o.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{T}" y2="{T + ph}" stroke="#f3f4f6"/>')
        o.append(f'<text x="{X(t):.1f}" y="{T + ph + 16}" text-anchor="middle" fill="#374151">{fmt(t)}</text>')
    o.append(f'<rect x="{L}" y="{T}" width="{pw}" height="{ph}" fill="none" stroke="#9ca3af"/>')
    o.append(f'<text x="{L + pw / 2}" y="{H - 18}" text-anchor="middle" fill="#111">{xlabel}</text>')
    for i, (label, v) in enumerate(hlines):
        o.append(f'<line x1="{L}" x2="{L + pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#6b7280" '
                 f'stroke-dasharray="6 4"/>')
        o.append(f'<text x="{L + pw - 4}" y="{Y(v) - 4:.1f}" text-anchor="end" fill="#4b5563">{html.escape(label)} {v:g}</text>')
    for i, (label, pts) in enumerate(series):
        c = COLORS[i % len(COLORS)]
        path = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in pts)
        o.append(f'<polyline fill="none" stroke="{c}" stroke-width="1.8" points="{path}"/>')
        if len(pts) <= 40:
            o += [f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="2.5" fill="{c}"/>' for x, y in pts]
        o.append(f'<rect x="{L + 10}" y="{T + 10 + 18 * i}" width="14" height="3" fill="{c}"/>')
        o.append(f'<text x="{L + 30}" y="{T + 15 + 18 * i}" fill="#111">{html.escape(label)}</text>')
    o.append("</svg>")
    return "\n".join(o) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+")
    p.add_argument("--labels", help="comma list, one per run (default: folder names)")
    p.add_argument("--out-dir", default="reports/figures")
    p.add_argument("--hline", action="append", default=[], help="METRIC:LABEL=VALUE")
    p.add_argument("--smooth", type=int, default=1, help="moving average over N records for loss/tokens_per_s")
    a = p.parse_args(argv)
    labels = a.labels.split(",") if a.labels else [os.path.basename(os.path.normpath(r)) for r in a.runs]
    if len(labels) != len(a.runs):
        raise SystemExit("--labels needs one label per run")
    logs = [load(r) for r in a.runs]
    hl: dict[str, list[tuple[str, float]]] = {}
    for h in a.hline:
        metric, rest = h.split(":", 1)
        label, val = rest.rsplit("=", 1)
        hl.setdefault(metric, []).append((label, float(val)))
    os.makedirs(a.out_dir, exist_ok=True)
    written = []
    for key, title in METRICS:
        series = []
        for label, recs in zip(labels, logs):
            pts = [(r["_x"], r[key]) for r in recs if isinstance(r.get(key), (int, float)) and math.isfinite(r[key])]
            if key in ("loss", "tokens_per_s") and a.smooth > 1 and pts:
                ys = [y for _, y in pts]
                ys = [sum(ys[max(0, i - a.smooth + 1):i + 1]) / len(ys[max(0, i - a.smooth + 1):i + 1])
                      for i in range(len(ys))]
                pts = [(x, y) for (x, _), y in zip(pts, ys)]
            if key == "tokens_per_s":
                pts = pts[2:]  # torch.compile warm-up
            if pts:
                series.append((label, pts))
        if not series:
            continue
        name = FILES.get(key, f"{key}.svg")
        with open(os.path.join(a.out_dir, name), "w", encoding="utf-8") as f:
            f.write(svg_plot(series, title, "training tokens (billions)", hl.get(key, [])))
        written.append(name)
    print(f"wrote {', '.join(written)} to {a.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
