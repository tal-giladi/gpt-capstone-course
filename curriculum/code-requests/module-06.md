# Code requests from Module 6 (for labs/common owner)

Lessons 06.1–06.4 work with the code as it is today. Items are improvements; each says what the
lesson does now.

1. **MFU overstated for untied embeddings (`GPT.flops_per_token`).** It uses 6 × num_params
   (minus wpe), so with `tie_embeddings=false` the input embedding table (a lookup, no matmul) is
   counted as matmul work: the logged MFU for `abl_untied` is ~19% too high (926 vs 775 MFLOP/token
   at 1024 context). Lesson 06.2 tells learners to compare tokens/s instead. Request: subtract
   `wte` from N when untied (keep it when tied, since it is then the lm_head matmul).
   DONE 2026-11-13 by main maintainer: GPT.flops_per_token subtracts wte from N when tie_embeddings=false (abl_untied now logs 775 MFLOP/token, same as tied). Tested. Lesson 06.2 (table row, paragraph, troubleshooting row, quiz q2) updated.
2. **One `weight_decay` for both optimizers (`train.build_optimizer`).** With `optimizer: muon`,
   the same value is applied as `p *= 1 - muon_lr * wd` to block matrices and as AdamW decay to
   embeddings/head; 0.1 is far too strong for Muon at lr 0.02 (~33× AdamW's per-step shrink), so
   the lessons and `main_350m.yaml` use `weight_decay: 0.0`, which also removes AdamW's decay on
   the embeddings (a confound noted in 06.3). Request: add `train.muon_weight_decay` (default 0.0)
   and keep `weight_decay` for AdamW only.
   DONE 2026-11-13 by main maintainer: Added train.muon_weight_decay (default 0.0, Muon groups only); weight_decay is AdamW-only. main_350m.yaml: weight_decay 0.1, muon_weight_decay 0.0; ablation_base.yaml lists muon_weight_decay 0.0. Tested. Lessons 06.3 (+quiz q2), 06.4 and labs/module-06/ablations.txt (Muon lines use train.muon_weight_decay=0.0) updated.
3. **`main_350m.yaml` template comment** says "Params: ... ~= 337M" — correct for the shipped
   switches (337.2M tied, 370.7M untied, checked with `GPT` on the meta device). No change needed;
   noting it because lesson 06.4 quotes these numbers.
4. **Nice to have: `train.decay_from_step` / branching a WSD decay from a stable checkpoint.**
   Lesson 06.3 mentions that WSD lets you branch decays off a stable checkpoint (Hägele et al.);
   the current schedule derives the decay start from `train_tokens`, which can be changed on resume
   (train config may change), so branching works by resuming a copy of the run dir with a smaller
   `train.train_tokens`. If Module 7 wants to teach that, a test that a resumed run with changed
   `train_tokens` decays correctly would make it safe to recommend.
   DONE 2026-11-13 by main maintainer: Added test_wsd_branch_with_smaller_train_tokens: a copy of a 40-step WSD run resumed from its step-20 (stable) checkpoint with train_tokens for 30 steps decays over steps 21-30 and matches an uninterrupted 30-step run exactly. Safe to recommend in Module 7 (set decay_frac for the new total). README documents it.
