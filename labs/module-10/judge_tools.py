"""Helpers around scripts/judge_winrate.py: reuse answers, estimate the judge bill, report honestly.

  # 1. on the GPU box: generate both models' answers once (dry run = no API calls, no API cost)
  python scripts/judge_winrate.py --model-a hf/my-gpt-350m-sft --model-b hf/my-gpt-350m-dpo \
      --n 300 --dry-run --out judge/gen
  python tools/judge_tools.py answers judge/gen/judgments.jsonl --out judge/answers
  #    -> judge/answers/answers_a.jsonl, answers_b.jsonl (prompt/answer), copy them to your laptop

  # 2. anywhere: what will judging cost? (2 judge calls per prompt, both answer orders)
  python tools/judge_tools.py cost judge/gen/judgments.jsonl

  # 3. after the real run: win rate with CIs, position bias, length bias -> winrate.json + notes.md
  python tools/judge_tools.py report judge/dpo_vs_sft/judgments.jsonl --out judge/dpo_vs_sft

Stdlib only. Token counts in `cost` are an ESTIMATE (characters / 4 for English text); prices are
the list prices checked on 2026-11-13 and must be re-checked on the providers' pricing pages.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random

# $ per 1M tokens (input, output), standard tier, checked 2026-11-13. Batch APIs: 50% off both.
PRICES = {
    "anthropic:claude-opus-5-5": (4.00, 20.00),
    "anthropic:claude-sonnet-5-5": (2.00, 10.00),
    "anthropic:claude-haiku-5-5": (0.10, 0.50),
    "openai:gpt-6.1-sol": (2.00, 10.00),
    "openai:gpt-6-luna": (0.10, 0.50),
    "openai:gpt-5-nano": (0.05, 0.40),
}
TEMPLATE_TOKENS = 110      # judge instructions around the prompt and answers (judge_winrate.JUDGE_TEMPLATE)
VISIBLE_OUT_TOKENS = 60    # one-sentence reason + verdict


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def tokens(text: str) -> float:
    return len(text) / 4.0


def cmd_answers(args):
    rows = read_jsonl(args.judgments)
    os.makedirs(args.out, exist_ok=True)
    write_jsonl(os.path.join(args.out, "answers_a.jsonl"), [{"prompt": r["prompt"], "answer": r["answer_a"]} for r in rows])
    write_jsonl(os.path.join(args.out, "answers_b.jsonl"), [{"prompt": r["prompt"], "answer": r["answer_b"]} for r in rows])
    empty_a = sum(not r["answer_a"].strip() for r in rows)
    empty_b = sum(not r["answer_b"].strip() for r in rows)
    print(f"wrote {len(rows)} prompts to {args.out}/answers_a.jsonl and answers_b.jsonl "
          f"(empty answers: A {empty_a}, B {empty_b})")


def cmd_cost(args):
    rows = read_jsonl(args.judgments)
    n_calls = 2 * len(rows)
    inp = sum(2 * (TEMPLATE_TOKENS + tokens(r["prompt"]) + tokens(r["answer_a"]) + tokens(r["answer_b"])) for r in rows)
    out = n_calls * (VISIBLE_OUT_TOKENS + args.thinking_tokens)
    print(f"{len(rows)} prompts -> {n_calls} judge calls, ~{inp / 1e3:.0f}k input and ~{out / 1e3:.0f}k output tokens "
          f"(ESTIMATE: chars/4, {args.thinking_tokens} thinking tokens per call)\n")
    print("| judge | $ in / out per 1M | standard | batch (-50%) |")
    print("|---|---|---|---|")
    for name, (pin, pout) in PRICES.items():
        usd = inp / 1e6 * pin + out / 1e6 * pout
        print(f"| {name} | {pin:.2f} / {pout:.2f} | ${usd:.2f} | ${usd / 2:.2f} |")
    print("\nPrices checked 2026-11-13; re-check https://platform.claude.com/docs/en/about-claude/pricing "
          "and https://developers.openai.com/api/docs/pricing before you run.")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def sign_test_p(wins, losses):
    """Two-sided exact binomial test of wins vs losses (ties ignored)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def cmd_report(args):
    rows = [r for r in read_jsonl(args.judgments) if r["outcome"] != "invalid"]
    invalid = sum(1 for r in read_jsonl(args.judgments) if r["outcome"] == "invalid")
    n = len(rows)
    if n == 0:
        raise SystemExit("no valid judgments")
    score = {"b_wins": 1.0, "tie": 0.5, "a_wins": 0.0}
    s = [score[r["outcome"]] for r in rows]
    wins = sum(r["outcome"] == "b_wins" for r in rows)
    losses = sum(r["outcome"] == "a_wins" for r in rows)
    ties = n - wins - losses
    rate = sum(s) / n
    rng = random.Random(args.seed)
    boots = sorted(sum(rng.choice(s) for _ in range(n)) / n for _ in range(args.bootstrap))
    ci = (boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1])
    # verdicts are [v1, v2]: v1 with A shown first, v2 with B shown first, already mapped back to A/B.
    first_pos = sum((r["verdicts"][0] == "A") + (r["verdicts"][1] == "B") for r in rows)
    decided = sum(sum(v in ("A", "B") for v in r["verdicts"]) for r in rows)
    consistent = sum(r["verdicts"][0] == r["verdicts"][1] for r in rows)
    len_a = [len(r["answer_a"].split()) for r in rows]
    len_b = [len(r["answer_b"].split()) for r in rows]
    b_longer_wins = sum(r["outcome"] == "b_wins" and lb > la for r, la, lb in zip(rows, len_a, len_b))
    report = {
        "judgments_file": args.judgments,
        "n_judged": n, "invalid": invalid, "b_wins": wins, "a_wins": losses, "ties": ties,
        "b_win_rate_ties_half": rate, "ci95_bootstrap": list(ci),
        "b_wins_among_decided": wins / (wins + losses) if wins + losses else float("nan"),
        "ci95_wilson_among_decided": list(wilson(wins, wins + losses)),
        "sign_test_p_two_sided": sign_test_p(wins, losses),
        "order_consistency": consistent / n,
        "first_position_pick_rate": first_pos / decided if decided else float("nan"),
        "mean_words_a": sum(len_a) / n, "mean_words_b": sum(len_b) / n,
        "share_b_wins_where_b_longer": b_longer_wins / wins if wins else float("nan"),
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "winrate.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    verdict = ("B is better" if ci[0] > 0.5 else "A is better" if ci[1] < 0.5 else "no detectable difference")
    notes = f"""# Win-rate result

- Comparison: B vs A, judged in both orders on {n} prompts ({invalid} invalid verdicts dropped).
- B wins {wins}, A wins {losses}, ties {ties} (a tie includes order-dependent verdicts).
- **B win rate (ties = 0.5): {rate:.1%}, 95% bootstrap CI {ci[0]:.1%}-{ci[1]:.1%} -> {verdict}.**
- Among decided prompts: B {report['b_wins_among_decided']:.1%} (Wilson 95% CI {report['ci95_wilson_among_decided'][0]:.1%}-{report['ci95_wilson_among_decided'][1]:.1%}), sign test p = {report['sign_test_p_two_sided']:.3g}.
- Judge order consistency: {report['order_consistency']:.1%}; picks the first-shown answer {report['first_position_pick_rate']:.1%} of the time (50% = no position bias).
- Mean answer length: A {report['mean_words_a']:.0f} words, B {report['mean_words_b']:.0f} words; {report['share_b_wins_where_b_longer']:.0%} of B's wins are prompts where B was longer.
"""
    with open(os.path.join(args.out, "notes.md"), "w", encoding="utf-8") as f:
        f.write(notes)
    print(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}/winrate.json and {args.out}/notes.md")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("answers", help="split a (dry-run) judgments.jsonl into answers_a/answers_b JSONL")
    a.add_argument("judgments")
    a.add_argument("--out", required=True)
    c = sub.add_parser("cost", help="estimate the judge bill from a judgments.jsonl with real answers")
    c.add_argument("judgments")
    c.add_argument("--thinking-tokens", type=int, default=300,
                   help="ESTIMATE of hidden reasoning tokens per call (billed as output); 0 for non-reasoning judges")
    r = sub.add_parser("report", help="win rate with CIs, position and length bias -> winrate.json + notes.md")
    r.add_argument("judgments")
    r.add_argument("--out", required=True)
    r.add_argument("--bootstrap", type=int, default=10000)
    r.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    {"answers": cmd_answers, "cost": cmd_cost, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
