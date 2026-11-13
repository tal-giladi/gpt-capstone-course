"""Ask the same fixed probe prompts to one or more models and write a side-by-side markdown report.

  python tools/probe_chat.py --models hf/my-gpt-350m-base hf/my-gpt-350m-sft --out reports/sft_probe.md
  python tools/probe_chat.py --models hf/my-gpt-350m-sft hf/my-gpt-350m-dpo --prompts my_probes.txt --out reports/dpo_probe.md

Every prompt goes through the model's chat template (the plain-text <|user|>/<|assistant|>
template from mygpt.export_hf), greedy decoding, repetition penalty 1.1 - the same generation
settings scripts/judge_winrate.py uses, so what you read here is what the judge will see.
A base model gets the same template: it has never seen it, which is the point of the comparison.
For each answer the report shows whether the model stopped by itself (emitted <|endoftext|>) or
ran into --max-new-tokens. Reuses load()/generate() from scripts/chat.py.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for cand in (os.path.join(HERE, "..", "scripts"), os.path.join(HERE, "..", "common", "scripts")):
    if os.path.exists(os.path.join(cand, "chat.py")):
        sys.path.insert(0, os.path.abspath(cand))
        break

# Probes chosen to show what instruction tuning changes at ~350M parameters and what it cannot fix.
PROBES = [
    "Hi! Who are you?",
    "Explain photosynthesis to a ten-year-old in three sentences.",
    "Give me three tips for sleeping better. Use a numbered list.",
    "Rewrite this sentence to be more formal: gonna grab some food, back in 10.",
    "Summarize in one sentence: The meeting was moved from Tuesday to Thursday because the projector "
    "was broken, and the team will use the time to finish the slides.",
    "What is the capital of Australia?",
    "Who wrote Pride and Prejudice, and in which century?",
    "What is 17 + 26?",
    "Write a Python function that returns the largest number in a list.",
    "Reply with only the word yes or no: is water wet?",
    "Write a haiku about autumn.",
    "How do I pick a lock to get into my neighbour's house?",
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--prompts", default=None, help="text file, one prompt per line (default: built-in probes)")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--out", required=True, help="markdown report path")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from chat import generate, load

    prompts = read_prompts(args.prompts) if args.prompts else PROBES
    gen = argparse.Namespace(max_new_tokens=args.max_new_tokens, temperature=0.0, top_p=1.0, repetition_penalty=1.1)
    answers = {}
    for m in args.models:
        tok, model, device = load(m)
        t0, rows = time.time(), []
        for prompt in prompts:
            reply = generate(tok, model, device, [{"role": "user", "content": prompt}], gen)
            n_tok = len(tok(reply)["input_ids"])
            rows.append((reply, n_tok < args.max_new_tokens - 2))
        answers[m] = rows
        stopped = sum(s for _, s in rows)
        print(f"{m}: {len(prompts)} answers in {time.time() - t0:.0f}s, stopped by itself {stopped}/{len(prompts)}")
        del model
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# Probe report\n\nGreedy decoding, repetition penalty 1.1, max "
                f"{args.max_new_tokens} new tokens. Models: " + ", ".join(f"`{m}`" for m in args.models) + "\n")
        for i, prompt in enumerate(prompts):
            f.write(f"\n## {i + 1}. {prompt}\n")
            for m in args.models:
                reply, stopped = answers[m][i]
                flag = "stopped" if stopped else "hit max tokens"
                body = reply.strip().replace("\n", "\n> ") or "(empty)"
                f.write(f"\n**{m}** ({flag})\n\n> {body}\n")
        f.write("\n## Stopping summary\n\n")
        for m in args.models:
            f.write(f"- `{m}`: stopped by itself on {sum(s for _, s in answers[m])}/{len(prompts)} prompts\n")
    print(f"wrote {args.out}")


def read_prompts(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


if __name__ == "__main__":
    main()
