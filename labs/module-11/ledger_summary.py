"""Sum the budget.md ledger: planned vs actual per module and in total (lesson 11.3).

    python tools/ledger_summary.py budget.md                  # prints a markdown table
    python tools/ledger_summary.py budget.md --cap 500 > reports/cost_table.md

Reads every table row shaped `| date | lesson | item | estimate | actual |` (the ledger format
used since 01.1). Amounts like `~$3.25`, `$201`, `€12.40` or `$____` are accepted; a row whose
actual is still blank is counted as "missing" and listed, so you can fill it from the provider's
billing page before you publish the report. Currencies are not converted: write EUR rows in USD.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import OrderedDict

ROW = re.compile(r"^\|\s*(\d{4}-\d{2}-\d{2})\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*$")
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def amount(cell: str) -> float | None:
    m = NUM.search(cell.replace(" ", ""))
    return float(m.group(0).replace(",", "")) if m else None


def module_of(lesson: str) -> str:
    m = re.match(r"(\d{2})", lesson.strip())
    return m.group(1) if m else "other"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("budget")
    p.add_argument("--cap", type=float, default=500.0)
    a = p.parse_args(argv)

    per = OrderedDict()
    missing = []
    for line in open(a.budget, encoding="utf-8"):
        m = ROW.match(line.strip())
        if not m:
            continue
        date, lesson, item, est, act = m.groups()
        e, x = amount(est), amount(act)
        if e is None and x is None:
            continue  # header or empty row
        mod = module_of(lesson)
        d = per.setdefault(mod, {"est": 0.0, "act": 0.0, "rows": 0})
        d["est"] += e or 0.0
        d["rows"] += 1
        if x is None:
            missing.append(f"{date} {lesson} {item}")
            d["act"] += e or 0.0  # count the estimate until the real number is in
        else:
            d["act"] += x

    if not per:
        print("no ledger rows found (expected | date | lesson | item | estimate | actual |)")
        return 1
    print("| Module | Rows | Planned | Actual | Difference |")
    print("|---|---|---|---|---|")
    te = ta = 0.0
    for mod, d in per.items():
        te += d["est"]
        ta += d["act"]
        print(f"| {mod} | {d['rows']} | ${d['est']:.2f} | ${d['act']:.2f} | {d['act'] - d['est']:+.2f} |")
    print(f"| **Total** | | **${te:.2f}** | **${ta:.2f}** | **{ta - te:+.2f}** |")
    print(f"\nCap ${a.cap:.0f}: used {100 * ta / a.cap:.0f}%, left ${a.cap - ta:.2f}.")
    if missing:
        print(f"\n{len(missing)} row(s) without an actual amount (estimate used above):")
        for r in missing:
            print(f"- {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
