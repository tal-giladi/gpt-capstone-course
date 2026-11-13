# Code requests from Module 10 (for labs/common owner)

1. **BLOCKING — fp32 master weights in `scripts/dpo.py`.** Same issue as Module 9 item 1: with the
   model loaded in bf16 and lr 5e-7 (the default) or 1e-6 (the SmolLM2-360M recipe the lesson uses),
   only 1-2% of bf16 weights change per AdamW step. Load with `dtype=torch.float32`; TRL then also
   builds the reference model in fp32 (`ref_model_init_kwargs.setdefault("dtype", self.model.dtype)`).
   Lesson 10.1 assumes the fixed behaviour.
2. **`scripts/dpo.py`: keep only the columns DPO needs.** Datasets such as
   `argilla/ultrafeedback-binarized-preferences-cleaned` have a string `prompt` column next to
   conversational `chosen`/`rejected` lists, which TRL rejects as mixed formats. A
   `remove_columns` to `{"chosen", "rejected"}` when `chosen` is a message list (as `sft.py` does
   for `messages`) would let the lesson use these alternatives without a conversion step. Lesson
   10.1 currently converts them to a local JSONL with a short Python snippet.
3. **`scripts/judge_winrate.py`: record token usage.** The summary has no input/output token counts,
   so the learner cannot reconcile the bill with the estimate. Suggest summing `resp.usage`
   (Anthropic) / `resp.usage` (OpenAI) into `summary.json` as `judge_input_tokens`,
   `judge_output_tokens`. Lesson 10.2 tells the learner to read the bill in the provider console.
4. **Nice to have: `--judge anthropic` via the Message Batches API** (50% cheaper, results within
   24 h). Note the server-side `fallbacks` parameter is rejected on the Batches API, and must not be
   sent for `claude-haiku-5-5` as a list (the current `fallbacks="default"` form is accepted there
   but a declined request stays declined).
5. **Nice to have: `judge_winrate.py` CI.** `ci95` uses a normal approximation with ties counted
   as 0.5 and is not clipped to [0, 1]; lesson 10.2 reports the bootstrap and Wilson intervals from
   `labs/module-10/judge_tools.py report` instead, which is fine. Clipping would avoid confusing
   negative lower bounds on small `--n`.
