# Code requests from Module 10 (for labs/common owner)

1. **BLOCKING — fp32 master weights in `scripts/dpo.py`.** Same issue as Module 9 item 1: with the
   model loaded in bf16 and lr 5e-7 (the default) or 1e-6 (the SmolLM2-360M recipe the lesson uses),
   only 1-2% of bf16 weights change per AdamW step. Load with `dtype=torch.float32`; TRL then also
   builds the reference model in fp32 (`ref_model_init_kwargs.setdefault("dtype", self.model.dtype)`).
   Lesson 10.1 assumes the fixed behaviour.
   DONE 2026-11-13 by main maintainer: same fix as Module 9 item 1 (fp32 load, bf16 autocast); TRL builds the reference in fp32, checked in the test.
2. **`scripts/dpo.py`: keep only the columns DPO needs.** Datasets such as
   `argilla/ultrafeedback-binarized-preferences-cleaned` have a string `prompt` column next to
   conversational `chosen`/`rejected` lists, which TRL rejects as mixed formats. A
   `remove_columns` to `{"chosen", "rejected"}` when `chosen` is a message list (as `sft.py` does
   for `messages`) would let the lesson use these alternatives without a conversion step. Lesson
   10.1 currently converts them to a local JSONL with a short Python snippet.
   DONE 2026-11-13 by main maintainer: dpo.py keeps only prompt/chosen/rejected; a string `prompt` next to message lists is dropped when `chosen` already starts with the user turn (H4, Argilla, verified on the real datasets) or turned into a user message when `chosen` is answer-only. Lesson 10.1 step 4 and its quiz q1 no longer use the JSONL conversion.
3. **`scripts/judge_winrate.py`: record token usage.** The summary has no input/output token counts,
   so the learner cannot reconcile the bill with the estimate. Suggest summing `resp.usage`
   (Anthropic) / `resp.usage` (OpenAI) into `summary.json` as `judge_input_tokens`,
   `judge_output_tokens`. Lesson 10.2 tells the learner to read the bill in the provider console.
   DONE 2026-11-13 by main maintainer: summary.json has judge_calls, judge_input_tokens, judge_output_tokens (from resp.usage), price_usd_per_mtok_in_out, estimated_cost_usd (list prices 2026-11-13, overridable with --price-in/--price-out). Declined (refusal / OpenAI refusal or content_filter) and no-verdict replies are counted separately (invalid_declined, invalid_no_verdict, per-call counts), printed as a WARNING, and invalid rows keep invalid_reason and judge_replies. Lesson 10.2 updated.
4. **Nice to have: `--judge anthropic` via the Message Batches API** (50% cheaper, results within
   24 h). Note the server-side `fallbacks` parameter is rejected on the Batches API, and must not be
   sent for `claude-haiku-5-5` as a list (the current `fallbacks="default"` form is accepted there
   but a declined request stays declined).
5. **Nice to have: `judge_winrate.py` CI.** `ci95` uses a normal approximation with ties counted
   as 0.5 and is not clipped to [0, 1]; lesson 10.2 reports the bootstrap and Wilson intervals from
   `labs/module-10/judge_tools.py report` instead, which is fine. Clipping would avoid confusing
   negative lower bounds on small `--n`.
   DONE 2026-11-13 by main maintainer: ci95 is now a Wilson interval (ties as 0.5) clipped to [0, 1], labelled by ci95_method. Lesson 10.2 troubleshooting row updated. (Item 4, the Batches API, is not done.)
