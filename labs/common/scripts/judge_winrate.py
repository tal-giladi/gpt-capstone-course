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


def make_judge(args):
    """Returns judge(text) -> raw reply. SDKs are imported only when actually used."""
    if args.dry_run:
        rng = random.Random(args.seed)
        return lambda text: rng.choice(["dry run [[A]]", "dry run [[B]]", "dry run [[TIE]]"])
    if args.judge == "anthropic":
        import anthropic
        client = anthropic.Anthropic()  # ANTHROPIC_API_KEY (or an `ant auth login` profile)
        model = args.judge_model or "claude-opus-5-5"

        def judge(text: str) -> str:
            resp = client.beta.messages.create(
                model=model, max_tokens=4000,
                output_config={"effort": "low"},  # a pairwise verdict needs little reasoning
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                messages=[{"role": "user", "content": text}])
            if resp.stop_reason == "refusal":
                return ""
            return "".join(b.text for b in resp.content if b.type == "text")
        return judge
    if not args.judge_model:
        raise SystemExit("--judge openai needs --judge-model")
    from openai import OpenAI
    client = OpenAI(base_url=args.base_url) if args.base_url else OpenAI()  # OPENAI_API_KEY

    def judge(text: str) -> str:
        resp = client.chat.completions.create(model=args.judge_model, messages=[{"role": "user", "content": text}])
        return resp.choices[0].message.content or ""
    return judge


def verdict(reply: str) -> str | None:
    found = re.findall(r"\[\[(A|B|TIE)\]\]", reply)
    return found[-1] if found else None


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
    wins = losses = ties = invalid = 0
    with open(os.path.join(args.out, "judgments.jsonl"), "w", encoding="utf-8") as f:
        for p, a, b in zip(prompts, ans_a, ans_b):
            v1 = verdict(judge(JUDGE_TEMPLATE.format(prompt=p, a=a, b=b)))       # B shown second
            v2 = verdict(judge(JUDGE_TEMPLATE.format(prompt=p, a=b, b=a)))       # B shown first
            v2 = {"A": "B", "B": "A"}.get(v2, v2)                                # map back to A=model A
            if v1 is None or v2 is None:
                outcome = "invalid"
                invalid += 1
            elif v1 == v2 == "B":
                outcome, wins = "b_wins", wins + 1
            elif v1 == v2 == "A":
                outcome, losses = "a_wins", losses + 1
            else:
                outcome, ties = "tie", ties + 1
            f.write(json.dumps({"prompt": p, "answer_a": a, "answer_b": b, "verdicts": [v1, v2], "outcome": outcome}) + "\n")
    n = wins + losses + ties
    rate = (wins + 0.5 * ties) / n if n else float("nan")
    se = math.sqrt(rate * (1 - rate) / n) if n else float("nan")
    summary = {"model_a": args.model_a or args.answers_a, "model_b": args.model_b or args.answers_b,
               "judge": "dry-run" if args.dry_run else f"{args.judge}:{args.judge_model or 'claude-opus-5-5'}",
               "n_judged": n, "b_wins": wins, "a_wins": losses, "ties": ties, "invalid": invalid,
               "b_win_rate": rate, "ci95": [rate - 1.96 * se, rate + 1.96 * se]}
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
