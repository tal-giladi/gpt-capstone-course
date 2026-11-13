# Build progress — gpt-capstone-course

Goal: guided real-world capstone (train your own ~350M GPT on rented GPUs, < $500), built to
`tals-academy/docs/new-course-instructions.md`. Plan: `curriculum/course-plan.md`.
Spec for writers: `curriculum/lesson-spec.md`. Code contract: `curriculum/code-contract.md`.

## Resume procedure
1. Read this file and `curriculum/status/*.log`; skip anything marked done.
2. Max two writing agents at once, one module each; only the main session edits shared files
   (`_sidebar.md`, `glossary.md`, `README.md`, `labs/common/` after the code unit, `BUILD_PROGRESS.md`).
3. After each module: merge glossary inbox, run the dry-run import
   (`cd ../../tals-academy && npm run check-course -- ../course-creator/gpt-capstone-course`), commit, push.

## Units
- [x] Foundations (plan, spec, contract, sidebar, README, glossary stub, course details)
- [x] Lab code `labs/common/` + CPU tests
- [x] Module 1 — Plan, accounts and guardrails
- [x] Module 2 — Data: find it, license it, store it
- [x] Module 3 — Your tokenizer and token shards
- [x] Module 4 — GPUs: pick, rent, benchmark
- [x] Module 5 — Baseline: reproduce GPT-2 124M
- [x] Module 6 — Make it yours: ablations
- [ ] Module 7 — The main run
- [ ] Module 8 — Evaluate honestly
- [x] Module 9 — Instruction tuning (SFT)
- [x] Module 10 — Preference tuning (DPO)
- [ ] Module 11 — Ship it
- [ ] Final QA (dry-run import 0 problems, links, git clean)

## Now working
- Agent G: Module 7, then Module 8
- Agent H: labs/common fixes from code-requests 09 and 10

## Decisions
- Modules numbered 1–11 (plan's 0–10 shifted by one).
- Main run ~350M / ~20B tokens / $500 cap; own BPE tokenizer; slug gpt-capstone-course; public repo.
- No reserved chat tokens in the own tokenizer: export uses plain-text <|user|>/<|assistant|> tags + EOT.
- Decontamination filters finished shards in place (labs/module-03/filter_shards.py).
- Val-only sets tok/common-val-gpt2/ and tok/common-val-bpe32k/ for same-text bpb comparison (evaluate --data-dir).
- Improvements limited to combinations exportable to HF GPT-2 / Llama / Qwen3 formats.

## Open questions
- (none)
