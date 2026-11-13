# Code requests from Module 6 (for labs/common owner)

Lessons 06.1–06.4 work with the code as it is today. Items are improvements; each says what the
lesson does now.

1. **MFU overstated for untied embeddings (`GPT.flops_per_token`).** It uses 6 × num_params
   (minus wpe), so with `tie_embeddings=false` the input embedding table (a lookup, no matmul) is
   counted as matmul work: the logged MFU for `abl_untied` is ~19% too high (926 vs 775 MFLOP/token
   at 1024 context). Lesson 06.2 tells learners to compare tokens/s instead. Request: subtract
   `wte` from N when untied (keep it when tied, since it is then the lm_head matmul).
2. **One `weight_decay` for both optimizers (`train.build_optimizer`).** With `optimizer: muon`,
   the same value is applied as `p *= 1 - muon_lr * wd` to block matrices and as AdamW decay to
   embeddings/head; 0.1 is far too strong for Muon at lr 0.02 (~33× AdamW's per-step shrink), so
   the lessons and `main_350m.yaml` use `weight_decay: 0.0`, which also removes AdamW's decay on
   the embeddings (a confound noted in 06.3). Request: add `train.muon_weight_decay` (default 0.0)
   and keep `weight_decay` for AdamW only.
3. **`main_350m.yaml` template comment** says "Params: ... ~= 337M" — correct for the shipped
   switches (337.2M tied, 370.7M untied, checked with `GPT` on the meta device). No change needed;
   noting it because lesson 06.4 quotes these numbers.
4. **Nice to have: `train.decay_from_step` / branching a WSD decay from a stable checkpoint.**
   Lesson 06.3 mentions that WSD lets you branch decays off a stable checkpoint (Hägele et al.);
   the current schedule derives the decay start from `train_tokens`, which can be changed on resume
   (train config may change), so branching works by resuming a copy of the run dir with a smaller
   `train.train_tokens`. If Module 7 wants to teach that, a test that a resumed run with changed
   `train_tokens` decays correctly would make it safe to recommend.
