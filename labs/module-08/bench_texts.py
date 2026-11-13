"""Write the WinoGrande and LAMBADA items lm-eval scores as a --benchmark-file for
`scripts/decontam.py` (lesson 08.2). Module 3 checked HellaSwag, ARC and PIQA; these two were not.
`decontam.py --benchmarks winogrande,lambada_openai` now loads the same items itself; this file is
for scanning on a box without Hub access (write the JSONL elsewhere, copy it over).

    python tools/bench_texts.py --out reports/bench_wg_lambada.jsonl
    python scripts/decontam.py --shards-or-parquet data/tok/bpe32k-100BT-30 --tokenizer data/tok/bpe32k-100BT-30 \
        --benchmark-file reports/bench_wg_lambada.jsonl --out reports/decontam-wg-lambada-bpe32k.json

Items (the same splits lm-eval 0.4.13 uses):
  winogrande      allenai/winogrande, winogrande_xl, validation (1,267): sentence with the blank
                  filled by the correct option
  lambada_openai  EleutherAI/lambada_openai, default, test (5,153): the full passage including
                  the target word
"""
from __future__ import annotations

import argparse
import json
import sys


def items() -> list[dict]:
    from datasets import load_dataset
    out = []
    wg = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    for r in wg:
        opt = r["option1"] if r["answer"] == "1" else r["option2"]
        out.append({"benchmark": "winogrande", "text": r["sentence"].replace("_", opt, 1)})
    lb = load_dataset("EleutherAI/lambada_openai", "default", split="test")
    out += [{"benchmark": "lambada_openai", "text": r["text"]} for r in lb]
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    rows = items()
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["benchmark"]] = counts.get(r["benchmark"], 0) + 1
    print(f"wrote {len(rows)} items to {a.out}: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
