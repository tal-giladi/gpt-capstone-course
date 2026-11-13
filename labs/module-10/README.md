# Module 10 lab files — preference tuning (DPO) and the judge

Helpers for lessons 10.1 and 10.2. Copy them into the `tools/` folder of your project repo and
commit them. The code they work with (`scripts/dpo.py`, `scripts/judge_winrate.py`, lm-eval) is
in `labs/common/`; lesson 10.1 also uses `labs/module-09/probe_chat.py` and
`labs/module-09/sft_data_stats.py`.

| File | Used in | What it does |
|---|---|---|
| `judge_tools.py answers` | 10.2 | Splits a `judgments.jsonl` (from a `judge_winrate.py --dry-run`) into `answers_a.jsonl` / `answers_b.jsonl`, so the GPU generates answers once and the paid judging runs later from your laptop with `--answers-a/--answers-b`. |
| `judge_tools.py cost` | 10.2 | ESTIMATES the judge bill for those exact answers (2 calls per prompt, characters/4 tokens, an assumed number of hidden reasoning tokens) for several Anthropic and OpenAI judges, standard and batch prices (checked 2026-11-13). |
| `judge_tools.py report` | 10.2 | Win rate with ties as 0.5 and a 95% bootstrap CI, win share among decided prompts with a Wilson CI, a two-sided sign test, the judge's order consistency and first-position pick rate, and answer-length bias. Writes `winrate.json` and `notes.md`. |
| `eval_delta.py` | 10.2 | Compares lm-eval result folders of base, SFT and DPO models: a markdown table of per-task accuracy and deltas, flagging drops larger than 2 combined standard errors. |

Examples:

```bash
python tools/judge_tools.py answers judge/gen/judgments.jsonl --out judge/answers
python tools/judge_tools.py cost judge/gen/judgments.jsonl
python tools/judge_tools.py report judge/dpo_vs_sft/judgments.jsonl --out judge/dpo_vs_sft
python tools/eval_delta.py evals/base evals/sft evals/dpo --names base,sft,dpo --out evals/regression.md
```

Requirements: Python 3.10+, standard library only. No API keys are read by these helpers; only
`scripts/judge_winrate.py` calls a paid API.

Tested on 2026-11-13 (Windows 11, Python 3.14): `answers` and `report` on a real dry-run
`judgments.jsonl` from `judge_winrate.py` and on a synthetic 200-prompt file; `cost` on the same
files; `eval_delta.py` on lm-eval 0.4.13 result folders of three tiny exported models. The judge
API itself was not called.
