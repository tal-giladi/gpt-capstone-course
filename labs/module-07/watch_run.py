"""Watch a long training run from its log.jsonl (stdlib only).

    # on the node, in its own tmux window: a status line every 2 minutes, phone alerts via ntfy
    python tools/watch_run.py runs/main_350m --target-tokens 20e9 --price-per-hour 35.12 \
        --started 2026-11-17T09:05 --plan-hours 5.3 --interval 120 --ntfy mygpt-7f3a9c2e

    # on your laptop, against a log pulled from the bucket: one status, exit code 2 if anything is wrong
    python tools/watch_run.py main_350m.log.jsonl --target-tokens 20e9 --once

    # the incident list for reports/main-run-log.md
    python tools/watch_run.py runs/main_350m --target-tokens 20e9 --markdown

What it checks (each check becomes an ALERT line, and a phone notification with --ntfy):
  stale     no new log line for --stale-minutes (default 10): hung run, dead node or finished run
  nonfinite the last loss or grad norm is NaN/inf
  spike     a train loss more than --spike nats (default 0.25) above the median of the previous
            --window records; reported as recovered once loss is back within half the threshold
  gradnorm  a grad norm more than --gn-factor x (default 5) the rolling median
  slow      the last --slow-records records (default 3) all below --slow-frac (default 0.85) of
            the run's median tokens/s (single dips right after an eval or HellaSwag are normal and
            ignored)
  cost      projected total node cost more than --over-plan (default 1.15) x the planned cost
It also reports resumes (a step lower than the previous one) and skip_batches events.

After a resume, steps between the last checkpoint and the crash appear twice in log.jsonl; the
LAST record for each step wins (same rule as tools/run_summary.py from Module 5).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import statistics
import sys
import time
import urllib.request


# ------------------------------------------------------------------ reading the log
def read_log(path: str) -> tuple[list[dict], list[dict], float]:
    """Return (records by step, last record per step wins), events, file mtime.
    Events: {"kind": "resume"|"skip", "step": ..., ...} in file order."""
    if os.path.isdir(path):
        path = os.path.join(path, "log.jsonl")
    by_step: dict[int, dict] = {}
    events: list[dict] = []
    prev = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # half-written last line after a kill
            step = int(rec.get("step", 0))
            if "skip_batches" in rec:
                events.append({"kind": "skip", "step": step, "batches": rec["skip_batches"],
                               "tokens": rec.get("skipped_tokens")})
                continue
            if step < prev:
                events.append({"kind": "resume", "step": step, "from_step": prev})
            prev = step
            by_step[step] = {**by_step.get(step, {}), **rec}
    return [by_step[s] for s in sorted(by_step)], events, os.path.getmtime(path)


def finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


# ------------------------------------------------------------------ analysis
def find_spikes(recs: list[dict], thr: float, window: int) -> list[dict]:
    """Loss spikes: loss > rolling median (previous `window` train records) + thr.
    A spike ends when loss is back below median + thr/2."""
    train = [r for r in recs if finite(r.get("loss"))]
    spikes, cur = [], None
    for i, r in enumerate(train):
        hist = [x["loss"] for x in train[max(0, i - window):i]]
        if len(hist) < max(5, window // 5):
            continue
        base = statistics.median(hist) if cur is None else cur["baseline"]
        excess = r["loss"] - base
        if cur is None and excess > thr:
            cur = {"start": r["step"], "baseline": round(base, 4), "peak": r["loss"], "peak_step": r["step"],
                   "recovered_step": None}
        elif cur is not None:
            if r["loss"] > cur["peak"]:
                cur["peak"], cur["peak_step"] = r["loss"], r["step"]
            if excess < thr / 2:
                cur["recovered_step"] = r["step"]
                spikes.append(cur)
                cur = None
    if cur is not None:
        spikes.append(cur)
    for s in spikes:
        s["peak"] = round(s["peak"], 4)
        s["excess"] = round(s["peak"] - s["baseline"], 4)
    return spikes


def find_gn_spikes(recs: list[dict], factor: float, window: int) -> list[dict]:
    train = [r for r in recs if finite(r.get("grad_norm"))]
    out = []
    for i, r in enumerate(train):
        hist = [x["grad_norm"] for x in train[max(0, i - window):i]]
        if len(hist) < max(5, window // 5):
            continue
        med = statistics.median(hist)
        if med > 0 and r["grad_norm"] > factor * med:
            out.append({"step": r["step"], "grad_norm": round(r["grad_norm"], 3), "median": round(med, 3)})
    return out


def speed_records(recs: list[dict]) -> list[dict]:
    sp = [r for r in recs if finite(r.get("tokens_per_s")) and r["tokens_per_s"] > 0]
    return sp[2:] if len(sp) > 4 else sp  # first records include torch.compile


def find_slow_stretches(recs: list[dict], frac: float, min_records: int) -> list[dict]:
    sp = speed_records(recs)
    if len(sp) < 5:
        return []
    med = statistics.median(r["tokens_per_s"] for r in sp)
    out, run = [], []
    for r in sp + [None]:
        if r is not None and r["tokens_per_s"] < frac * med:
            run.append(r)
            continue
        if len(run) >= min_records:
            out.append({"from_step": run[0]["step"], "to_step": run[-1]["step"], "records": len(run),
                        "min_ratio": round(min(x["tokens_per_s"] for x in run) / med, 3)})
        run = []
    return out


def parse_time(s: str) -> float:
    t = dt.datetime.fromisoformat(s)
    if t.tzinfo is None:
        t = t.astimezone()  # local time
    return t.timestamp()


def status(path: str, a, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    recs, events, mtime = read_log(path)
    if not recs:
        return {"alerts": [("stale", "log has no records yet")], "step": 0}
    last = recs[-1]
    tps_step = next((r["tokens"] / r["step"] for r in recs if r.get("tokens") and r.get("step")), None)
    tokens = last["step"] * tps_step if tps_step else None
    s: dict = {"step": last["step"], "tokens": tokens, "events": events}
    last_loss_rec = next((r for r in reversed(recs) if "loss" in r), None)
    s["loss"] = last_loss_rec and last_loss_rec["loss"]
    s["grad_norm"] = last_loss_rec and last_loss_rec.get("grad_norm")
    trail = [r["loss"] for r in recs[-20:] if finite(r.get("loss"))]
    s["loss_median20"] = statistics.median(trail) if trail else None
    for k in ("val_loss", "val_bpb", "hellaswag"):
        r = next((r for r in reversed(recs) if finite(r.get(k))), None)
        s[k] = r and (r[k], r["step"])
    sp = speed_records(recs)
    s["tps_run"] = statistics.median(r["tokens_per_s"] for r in sp) if sp else None
    s["tps_recent"] = statistics.median(r["tokens_per_s"] for r in sp[-5:]) if sp else None
    s["age_min"] = (now - mtime) / 60
    s["spikes"] = find_spikes(recs, a.spike, a.window)
    s["gn_spikes"] = find_gn_spikes(recs, a.gn_factor, a.window)
    s["slow"] = find_slow_stretches(recs, a.slow_frac, a.slow_records)

    alerts = []
    if s["age_min"] > a.stale_minutes:
        alerts.append(("stale", f"no new log line for {s['age_min']:.0f} min (hung, dead node, or finished?)"))
    if last_loss_rec and not (finite(last_loss_rec["loss"]) and finite(last_loss_rec.get("grad_norm", 0.0))):
        alerts.append(("nonfinite", f"non-finite loss/grad at step {last_loss_rec['step']}"))
    open_spikes = [x for x in s["spikes"] if x["recovered_step"] is None]
    if open_spikes:
        x = open_spikes[-1]
        alerts.append(("spike", f"loss spike since step {x['start']}: peak {x['peak']:.3f} vs baseline "
                                f"{x['baseline']:.3f} (+{x['excess']:.3f}), not recovered"))
    if s["gn_spikes"] and s["gn_spikes"][-1]["step"] == last_loss_rec["step"]:
        g = s["gn_spikes"][-1]
        alerts.append(("gradnorm", f"grad norm {g['grad_norm']} at step {g['step']} = "
                                   f"{g['grad_norm'] / g['median']:.1f}x median"))
    if s["slow"] and sp and s["slow"][-1]["to_step"] == sp[-1]["step"]:
        x = s["slow"][-1]
        alerts.append(("slow", f"tokens/s below {a.slow_frac:.0%} of run median for the last {x['records']} "
                               f"records (min {x['min_ratio']:.0%})"))

    if a.target_tokens and tokens is not None:
        s["done_frac"] = tokens / a.target_tokens
        if s["tps_recent"]:
            s["eta_h"] = max(0.0, a.target_tokens - tokens) / s["tps_recent"] / 3600
    if a.started:
        s["elapsed_h"] = (now - parse_time(a.started)) / 3600
        if a.price_per_hour:
            s["cost_so_far"] = s["elapsed_h"] * a.price_per_hour
            if "eta_h" in s:
                s["cost_projected"] = (s["elapsed_h"] + s["eta_h"]) * a.price_per_hour
                if a.plan_hours:
                    plan = a.plan_hours * a.price_per_hour
                    s["cost_plan"] = plan
                    if s["cost_projected"] > a.over_plan * plan:
                        alerts.append(("cost", f"projected ${s['cost_projected']:.0f} > {a.over_plan:.2f} x plan "
                                               f"${plan:.0f}"))
    s["alerts"] = alerts
    return s


# ------------------------------------------------------------------ output
def f4(x):
    return "-" if x is None else f"{x:.4f}"


def status_line(s: dict) -> str:
    ts = dt.datetime.now().strftime("%H:%M")
    parts = [f"{ts} step {s['step']:,}"]
    if s.get("done_frac") is not None:
        parts.append(f"{100 * s['done_frac']:.1f}%")
    parts.append(f"loss {f4(s.get('loss'))} (med20 {f4(s.get('loss_median20'))})")
    if s.get("val_bpb"):
        parts.append(f"val_bpb {s['val_bpb'][0]:.4f}@{s['val_bpb'][1]}")
    if s.get("hellaswag"):
        parts.append(f"hs {s['hellaswag'][0]:.4f}@{s['hellaswag'][1]}")
    if s.get("tps_recent"):
        parts.append(f"{s['tps_recent'] / 1e6:.2f}M tok/s ({s['tps_recent'] / s['tps_run']:.0%} of median)")
    if s.get("eta_h") is not None:
        parts.append(f"ETA {s['eta_h']:.1f} h")
    if s.get("cost_so_far") is not None:
        c = f"${s['cost_so_far']:.0f} so far"
        if s.get("cost_projected") is not None:
            c += f", ~${s['cost_projected']:.0f} total"
            if s.get("cost_plan"):
                c += f" (plan ${s['cost_plan']:.0f})"
        parts.append(c)
    parts.append(f"log age {s['age_min']:.1f} min")
    lines = [" | ".join(parts)]
    lines += [f"ALERT {k}: {msg}" for k, msg in s["alerts"]]
    return "\n".join(lines)


def markdown(s: dict) -> str:
    out = ["### Health summary (tools/watch_run.py)", "",
           f"Last step {s['step']:,}" + (f" ({100 * s['done_frac']:.1f}% of target)" if s.get("done_frac") else "")
           + f", last train loss {f4(s.get('loss'))}, median tokens/s {s.get('tps_run') or 0:,.0f}.", ""]
    out += ["| Kind | Steps | Detail |", "|---|---|---|"]
    rows = []
    for e in s["events"]:
        if e["kind"] == "resume":
            rows.append(("resume", f"{e['from_step']:,} -> {e['step']:,}",
                         "log restarted from an earlier step (checkpoint resume)"))
        else:
            rows.append(("skip_batches", f"{e['step']:,}", f"{e['batches']} batches ({e['tokens'] or 0:,} tokens) skipped"))
    for x in s["spikes"]:
        rec = f"recovered by step {x['recovered_step']:,}" if x["recovered_step"] else "NOT recovered"
        rows.append(("loss spike", f"{x['start']:,}-{x['peak_step']:,}",
                     f"peak {x['peak']:.3f} vs baseline {x['baseline']:.3f} (+{x['excess']:.3f}), {rec}"))
    for g in s["gn_spikes"]:
        rows.append(("grad-norm spike", f"{g['step']:,}", f"{g['grad_norm']} vs median {g['median']}"))
    for x in s["slow"]:
        rows.append(("slowdown", f"{x['from_step']:,}-{x['to_step']:,}",
                     f"{x['records']} records below threshold, min {x['min_ratio']:.0%} of median"))
    rows.sort(key=lambda r: int(r[1].split("-")[0].split(" ")[0].replace(",", "")))
    out += [f"| {k} | {st} | {d} |" for k, st, d in rows]
    if not rows:
        out.append("| none | - | no incidents detected |")
    return "\n".join(out) + "\n"


def notify(topic: str, title: str, msg: str, server: str = "https://ntfy.sh") -> None:
    req = urllib.request.Request(f"{server.rstrip('/')}/{topic}", data=msg.encode("utf-8"), method="POST",
                                 headers={"Title": title, "Priority": "4", "Tags": "warning"})
    try:
        urllib.request.urlopen(req, timeout=10).read()
    except Exception as e:  # never let a notification failure stop the watcher
        print(f"[watch_run] ntfy failed: {e}", file=sys.stderr)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="run dir (with log.jsonl) or a log.jsonl file")
    p.add_argument("--target-tokens", type=float, help="planned training tokens, e.g. 20e9")
    p.add_argument("--started", help="node start time, ISO format, e.g. 2026-11-17T09:05 (local time)")
    p.add_argument("--price-per-hour", type=float, help="node price per hour (all GPUs), e.g. 35.12")
    p.add_argument("--plan-hours", type=float, help="billed hours you planned in 07.1, e.g. 5.3")
    p.add_argument("--interval", type=int, default=120, help="seconds between checks (default 120)")
    p.add_argument("--once", action="store_true", help="check once and exit (exit code 2 if any alert)")
    p.add_argument("--markdown", action="store_true", help="print the incident table for the run log and exit")
    p.add_argument("--json", action="store_true", help="print the full status as JSON and exit")
    p.add_argument("--ntfy", help="ntfy.sh topic for phone alerts (use a long random name)")
    p.add_argument("--ntfy-server", default="https://ntfy.sh")
    p.add_argument("--stale-minutes", type=float, default=10)
    p.add_argument("--spike", type=float, default=0.25, help="loss spike threshold in nats above rolling median")
    p.add_argument("--window", type=int, default=50, help="records in the rolling median")
    p.add_argument("--gn-factor", type=float, default=5.0)
    p.add_argument("--slow-frac", type=float, default=0.85)
    p.add_argument("--slow-records", type=int, default=3)
    p.add_argument("--over-plan", type=float, default=1.15)
    p.add_argument("--repeat-minutes", type=float, default=30, help="re-send the same alert after this long")
    a = p.parse_args(argv)

    if a.markdown or a.json or a.once:
        s = status(a.run, a)
        if a.json:
            print(json.dumps(s, indent=2, default=str))
        elif a.markdown:
            print(markdown(s))
        else:
            print(status_line(s))
        return 2 if (a.once and s["alerts"]) else 0

    sent: dict[str, float] = {}
    if a.ntfy:
        notify(a.ntfy, "watch_run started", f"watching {a.run}", a.ntfy_server)
    while True:
        try:
            s = status(a.run, a)
            print(status_line(s), flush=True)
            for kind, msg in s["alerts"]:
                if a.ntfy and time.time() - sent.get(kind, 0) > a.repeat_minutes * 60:
                    notify(a.ntfy, f"main run: {kind}", f"step {s['step']}: {msg}", a.ntfy_server)
                    sent[kind] = time.time()
            for kind in list(sent):
                if kind not in {k for k, _ in s["alerts"]}:
                    del sent[kind]  # cleared: a new occurrence alerts immediately
        except FileNotFoundError:
            print(f"[watch_run] {a.run}: no log.jsonl yet", flush=True)
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
