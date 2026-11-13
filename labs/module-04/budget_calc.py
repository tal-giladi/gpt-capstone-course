"""Turn measured tokens/s into GPU-hours and dollars for every planned run (stdlib only).

Measured mode (after lesson 04.3's benchmarks, per-GPU tokens/s from scripts/bench.py):

    python tools/budget_calc.py --tps-baseline 520000 --tps-ablation 560000 --tps-main 190000 \
        --price-1gpu 2.69 --price-8gpu 4.39

FLOP mode (before you have measurements; assumes an MFU):

    python tools/budget_calc.py --mfu 0.40 --price-1gpu 2.69 --price-8gpu 4.39

Prints a markdown table you paste into budget.md. Every number is an estimate: multi-GPU scaling
is checked in Module 5, overhead covers compile, evals, checkpoints and restarts.
"""

from __future__ import annotations

import argparse

PEAK = {"h100": 989e12, "h200": 989e12, "b200": 2250e12}  # dense BF16 FLOP/s per GPU


def flops_tps(n_params: float, mfu: float, peak: float) -> float:
    """Tokens/s per GPU from C = 6*N*D (attention FLOPs ignored, ~5-8% at these sizes)."""
    return mfu * peak / (6 * n_params)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tps-baseline", type=float, help="per-GPU tokens/s, configs/baseline_124m.yaml")
    p.add_argument("--tps-ablation", type=float, help="per-GPU tokens/s, configs/ablation_base.yaml")
    p.add_argument("--tps-main", type=float, help="per-GPU tokens/s, configs/main_350m.yaml")
    p.add_argument("--mfu", type=float, help="use 6ND FLOPs at this MFU instead of measurements")
    p.add_argument("--gpu", default="h100", choices=PEAK)
    p.add_argument("--n-baseline", type=float, default=124e6)
    p.add_argument("--n-ablation", type=float, default=124e6)
    p.add_argument("--n-main", type=float, default=350e6)
    p.add_argument("--d-baseline", type=float, default=10e9, help="training tokens")
    p.add_argument("--d-ablation", type=float, default=1e9, help="tokens per ablation run")
    p.add_argument("--d-main", type=float, default=20e9)
    p.add_argument("--ablations", type=int, default=10)
    p.add_argument("--price-1gpu", type=float, required=True, help="$/GPU-hour for 1-GPU boxes")
    p.add_argument("--price-8gpu", type=float, required=True, help="$/GPU-hour on the 8-GPU node")
    p.add_argument("--scaling", type=float, default=0.90, help="8-GPU efficiency vs 8x one GPU")
    p.add_argument("--overhead", type=float, default=1.25, help="compile, evals, ckpts, restarts")
    p.add_argument("--fixed", type=float, default=40 + 10 + 12 + 25 + 8,
                   help="$ for dev box, CPU boxes, eval, SFT+DPO, storage (course plan)")
    p.add_argument("--cap", type=float, default=500)
    a = p.parse_args()

    if a.mfu:
        peak = PEAK[a.gpu]
        tps = {k: flops_tps(n, a.mfu, peak) for k, n in
               (("baseline", a.n_baseline), ("ablation", a.n_ablation), ("main", a.n_main))}
        src = f"6ND at MFU {a.mfu:.0%} on {a.gpu.upper()}"
    else:
        if not (a.tps_baseline and a.tps_ablation and a.tps_main):
            p.error("give all three --tps-* values, or --mfu")
        tps = {"baseline": a.tps_baseline, "ablation": a.tps_ablation, "main": a.tps_main}
        src = "measured tokens/s"

    rows = []
    # (name, tokens, per-GPU tokens/s, GPUs, $/GPU-h, scaling)
    plan = [
        ("Baseline 124M x %.0fB" % (a.d_baseline / 1e9), a.d_baseline, tps["baseline"], 8, a.price_8gpu, a.scaling),
        ("Ablations %d x %.1fB" % (a.ablations, a.d_ablation / 1e9), a.ablations * a.d_ablation, tps["ablation"], 1, a.price_1gpu, 1.0),
        ("Main 350M x %.0fB" % (a.d_main / 1e9), a.d_main, tps["main"], 8, a.price_8gpu, a.scaling),
    ]
    total = 0.0
    for name, tokens, t, gpus, price, scale in plan:
        gpu_h = tokens / (t * scale) / 3600 * a.overhead
        wall_h = gpu_h / gpus
        usd = gpu_h * price
        per_b = price * 1e9 / (t * scale) / 3600
        total += usd
        rows.append(f"| {name} | {t:,.0f} | {gpus} | {wall_h:.1f} h | {gpu_h:.1f} | ${per_b:.2f} | ${usd:.0f} |")

    print(f"Source: {src}; scaling {a.scaling:.0%} on 8 GPUs; overhead x{a.overhead}\n")
    print("| Run | Tokens/s per GPU | GPUs | Wall clock | GPU-h | $ per 1B tokens | Cost |")
    print("|---|---|---|---|---|---|---|")
    print("\n".join(rows))
    grand = total + a.fixed
    print(f"| Other items from the course plan | | | | | | ${a.fixed:.0f} |")
    print(f"| **Subtotal** | | | | | | **${grand:.0f}** |")
    print(f"| Contingency to the ${a.cap:.0f} cap | | | | | | ${a.cap - grand:.0f} ({(a.cap - grand) / grand:.0%}) |")
    if grand > a.cap * 0.8:
        print("\nWARNING: less than 25% contingency left. Shrink the main run (fewer tokens or a smaller model) now.")


if __name__ == "__main__":
    main()
