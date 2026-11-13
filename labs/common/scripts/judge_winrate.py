"""Pairwise win-rate of model B vs model A, judged by a strong LLM through an API.

  # see exactly what would be sent, no API calls, no cost:
  python scripts/judge_winrate.py --model-a hf/my-gpt-350m-sft --model-b hf/my-gpt-350m-dpo --n 5 --dry-run
  # real run (Anthropic; key from ANTHROPIC_API_KEY):
  python scripts/judge_winrate.py --model-a hf/my-gpt-350m-sft --model-b hf/my-gpt-350m-dpo --n 200 --out judge/dpo_vs_sft
  # any OpenAI-compatible endpoint (key from OPENAI_API_KEY):
  python scripts/judge_winrate.py ... --judge openai --judge-model MODEL [--base-url https://.../v1]

Each prompt is judged twice with the answer order swapped (cancels position bias): B wins a
prompt if it wins both orders, ties if the verdicts disagree. Instead of models you can pass
pre-generated answers (--answers-a/--answers-b JSONL with prompt/answer fields).
A prompt is `invalid` when either call is declined by the judge (refusal) or has no parseable
verdict; both kinds are counted separately and left out of the win rate. summary.json also has
the judge's real input/output token counts (from the API's usage field) and the cost they imply
at list prices (checked 2026-11-13; override with --price-in/--price-out), and a Wilson 95% CI
(ties count 0.5), which always lies in [0, 1].
Writes OUT/judgments.jsonl and OUT/summary.json. Install: pip install -e .[post]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))

JUDGE_TEMPLATE = """You are judging two AI assistant answers to the same user prompt.
Judge helpfulness, correctness and clarity. Ignore answer length unless it hurts quality.
Do not let the order of the answers influence you.

[User prompt]
{prompt}

[Answer A]
{a}

[Answer B]
{b}

Reply with a one-sentence reason, then exactly one verdict on the last line: [[A]], [[B]] or [[TIE]]."""

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"
# $ per 1M tokens (input, output), standard (non-batch) tier, checked 2026-11-13 -- re-check
# https://platform.claude.com/docs/en/about-claude/pricing and https://developers.openai.com/api/docs/pricing
PRICES = {
    "anthropic:claude-opus-5-5": (4.00, 20.00),
    "anthropic:claude-sonnet-5-5": (2.00, 10.00),
    "anthropic:claude-haiku-5-5": (0.10, 0.50),
    "openai:gpt-6.1-sol": (2.00, 10.00),
    "openai:gpt-6-luna": (0.10, 0.50),
    "openai:gpt-5-nano": (0.05, 0.40),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-a", help="baseline HF model dir (e.g. the SFT model)")
    p.add_argument("--model-b", help="candidate HF model dir (e.g. the DPO model)")
    p.add_argument("--answers-a", help="JSONL with prompt/answer instead of --model-a")
    p.add_argument("--answers-b", help="JSONL with prompt/answer instead of --model-b")
    p.add_argument("--prompts", help="prompts file: .jsonl with a 'prompt' field, or .txt one per line")
    p.add_argument("--dataset", default="HuggingFaceH4/no_robots", help="used when --prompts is not given")
    p.add_argument("--dataset-split", default="test")
    p.add_argument("--dataset-field", default="prompt")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--judge", choices=["anthropic", "openai"], default="anthropic")
    p.add_argument("--judge-model", default=None, help="default for anthropic: claude-opus-5-5; required for openai")
    p.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint base URL")
    p.add_argument("--out", default="judge_out")
    p.add_argument("--price-in", type=float, default=None, help="$ per 1M input tokens (default: built-in list price)")
    p.add_argument("--price-out", type=float, default=None, help="$ per 1M output tokens (default: built-in list price)")
    p.add_argument("--dry-run", action="store_true", help="no API calls: print one judge prompt, use a coin-flip judge")
    return p.parse_args(argv)


def load_prompts(args) -> list[str]:
    if args.prompts:
        with open(args.prompts, encoding="utf-8") as f:
            if args.prompts.endswith(".jsonl"):
                prompts = [json.loads(line)["prompt"] for line in f if line.strip()]
            else:
                prompts = [line.strip() for line in f if line.strip()]
    else:
        from datasets import load_dataset
        ds = load_dataset(args.dataset, split=args.dataset_split)
        prompts = [r[args.dataset_field] for r in ds]
    random.Random(args.seed).shuffle(prompts)
    return prompts[:args.n]


def load_answers(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        return {r["prompt"]: r["answer"] for r in map(json.loads, filter(str.strip, f))}


def generate_answers(model_dir: str, prompts: list[str], max_new_tokens: int) -> list[str]:
    from chat import generate, load
    tok, model, device = load(model_dir)
    gen_args = argparse.Namespace(max_new_tokens=max_new_tokens, temperature=0.0, top_p=1.0, repetition_penalty=1.1)
    out = []
    for i, prompt in enumerate(prompts):
        out.append(generate(tok, model, device, [{"role": "user", "content": prompt}], gen_args))
        if (i + 1) % 20 == 0:
            print(f"  {model_dir}: {i + 1}/{len(prompts)} answers", flush=True)
    return out


def judge_name(args) -> str:
    if args.dry_run:
        return "dry-run"
    return f"{args.judge}:{args.judge_model or DEFAULT_ANTHROPIC_MODEL}"


def make_judge(args):
    """Returns judge(text) -> {"text", "declined", "input_tokens", "output_tokens"}.
    SDKs are imported only when actually used."""
    if args.dry_run:
        rng = random.Random(args.seed)
        return lambda text: {"text": rng.choice(["dry run [[A]]", "dry run [[B]]", "dry run [[TIE]]"]),
                             "declined": False, "input_tokens": 0, "output_tokens": 0}
    if args.judge == "anthropic":
        import anthropic
        client = anthropic.Anthropic()  # ANTHROPIC_API_KEY (or an `ant auth login` profile)
        model = args.judge_model or DEFAULT_ANTHROPIC_MODEL

        def judge(text: str) -> dict:
            resp = client.beta.messages.create(
                model=model, max_tokens=4000,
                output_config={"effort": "low"},  # a pairwise verdict needs little reasoning
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                messages=[{"role": "user", "content": text}])
            usage = getattr(resp, "usage", None)
            return {"text": "".join(b.text for b in resp.content if b.type == "text"),
                    "declined": resp.stop_reason == "refusal",
                    "input_tokens": getattr(usage, "input_tokens", 0) or 0,
                    "output_tokens": getattr(usage, "output_tokens", 0) or 0}
        return judge
    if not args.judge_model:
        raise SystemExit("--judge openai needs --judge-model")
    from openai import OpenAI
    client = OpenAI(base_url=args.base_url) if args.base_url else OpenAI()  # OPENAI_API_KEY

    def judge(text: str) -> dict:
        resp = client.chat.completions.create(model=args.judge_model, messages=[{"role": "user", "content": text}])
        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        return {"text": choice.message.content or "",
                "declined": bool(getattr(choice.message, "refusal", None)) or choice.finish_reason == "content_filter",
                "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "output_tokens": getattr(usage, "completion_tokens", 0) or 0}
    return judge


def verdict(reply: str) -> str | None:
    found = re.findall(r"\[\[(A|B|TIE)\]\]", reply)
    return found[-1] if found else None


def call_status(res: dict) -> str:
    """'declined' (the judge refused), 'no_verdict' (no [[A]]/[[B]]/[[TIE]] in the reply) or 'ok'."""
    if res["declined"]:
        return "declined"
    return "ok" if verdict(res["text"]) is not None else "no_verdict"


def wilson(k: float, n: int, z: float = 1.96) -> list[float]:
    """Wilson score interval for k successes in n trials (k may be fractional: ties = 0.5).
    Always inside [0, 1], unlike the normal approximation on small n."""
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(max(p * (1 - p), 0.0) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def estimate_cost(args, input_tokens: int, output_tokens: int):
    """(usd, [price_in, price_out]) at $/1M tokens, or (None, None) when the price is unknown."""
    if args.dry_run:
        return 0.0, [0.0, 0.0]
    pin, pout = PRICES.get(judge_name(args), (None, None))
    pin = args.price_in if args.price_in is not None else pin
    pout = args.price_out if args.price_out is not None else pout
    if pin is None or pout is None:
        return None, None
    return input_tokens / 1e6 * pin + output_tokens / 1e6 * pout, [pin, pout]


def main(argv=None):
    args = parse_args(argv)
    if not (args.model_a or args.answers_a) or not (args.model_b or args.answers_b):
        raise SystemExit("need --model-a/--answers-a and --model-b/--answers-b")
    prompts = load_prompts(args)
    if args.answers_a:
        amap = load_answers(args.answers_a)
        prompts = [p for p in prompts if p in amap]
    if args.answers_b:
        bmap = load_answers(args.answers_b)
        prompts = [p for p in prompts if p in bmap]
    ans_a = [amap[p] for p in prompts] if args.answers_a else generate_answers(args.model_a, prompts, args.max_new_tokens)
    ans_b = [bmap[p] for p in prompts] if args.answers_b else generate_answers(args.model_b, prompts, args.max_new_tokens)

    judge = make_judge(args)
    if args.dry_run:
        print("---- example judge request (dry run, nothing sent) ----")
        print(JUDGE_TEMPLATE.format(prompt=prompts[0], a=ans_a[0], b=ans_b[0]))
        print("-------------------------------------------------------")
    os.makedirs(args.out, exist_ok=True)
    wins = losses = ties = 0
    invalid = {"declined": 0, "no_verdict": 0}
    calls = {"ok": 0, "declined": 0, "no_verdict": 0}
    tok_in = tok_out = 0
    with open(os.path.join(args.out, "judgments.jsonl"), "w", encoding="utf-8") as f:
        for p, a, b in zip(prompts, ans_a, ans_b):
            r1 = judge(JUDGE_TEMPLATE.format(prompt=p, a=a, b=b))                # B shown second
            r2 = judge(JUDGE_TEMPLATE.format(prompt=p, a=b, b=a))                # B shown first
            statuses = [call_status(r1), call_status(r2)]
            for r, st in zip((r1, r2), statuses):
                calls[st] += 1
                tok_in += r["input_tokens"]
                tok_out += r["output_tokens"]
            v1 = verdict(r1["text"]) if statuses[0] == "ok" else None
            v2 = verdict(r2["text"]) if statuses[1] == "ok" else None
            v2 = {"A": "B", "B": "A"}.get(v2, v2)                                # map back to A=model A
            row = {"prompt": p, "answer_a": a, "answer_b": b, "verdicts": [v1, v2], "judge_status": statuses}
            if v1 is None or v2 is None:
                outcome = "invalid"
                reason = "declined" if "declined" in statuses else "no_verdict"
                invalid[reason] += 1
                row["invalid_reason"] = reason
                row["judge_replies"] = [r1["text"], r2["text"]]                  # to inspect what went wrong
            elif v1 == v2 == "B":
                outcome, wins = "b_wins", wins + 1
            elif v1 == v2 == "A":
                outcome, losses = "a_wins", losses + 1
            else:
                outcome, ties = "tie", ties + 1
            row["outcome"] = outcome
            f.write(json.dumps(row) + "\n")
    n = wins + losses + ties
    rate = (wins + 0.5 * ties) / n if n else float("nan")
    cost, prices = estimate_cost(args, tok_in, tok_out)
    n_invalid = invalid["declined"] + invalid["no_verdict"]
    summary = {"model_a": args.model_a or args.answers_a, "model_b": args.model_b or args.answers_b,
               "judge": judge_name(args),
               "n_judged": n, "b_wins": wins, "a_wins": losses, "ties": ties,
               "invalid": n_invalid, "invalid_declined": invalid["declined"],
               "invalid_no_verdict": invalid["no_verdict"],
               "b_win_rate": rate, "ci95": wilson(wins + 0.5 * ties, n),
               "ci95_method": "wilson, ties count 0.5, clipped to [0, 1]",
               "judge_calls": sum(calls.values()), "judge_calls_declined": calls["declined"],
               "judge_calls_no_verdict": calls["no_verdict"],
               "judge_input_tokens": tok_in, "judge_output_tokens": tok_out,
               "price_usd_per_mtok_in_out": prices,
               "estimated_cost_usd": round(cost, 6) if cost is not None else None,
               "cost_note": ("list prices checked 2026-11-13, standard tier; the provider bill is authoritative"
                             if cost is not None else
                             "no built-in price for this judge model: pass --price-in/--price-out")}
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    if n_invalid:
        print(f"WARNING: {n_invalid} of {len(prompts)} prompts invalid ({invalid['declined']} declined by the judge, "
              f"{invalid['no_verdict']} without a parseable verdict); they are not in the win rate. "
              f"See invalid_reason / judge_replies in judgments.jsonl.", flush=True)
    return summary


if __name__ == "__main__":
    main()
