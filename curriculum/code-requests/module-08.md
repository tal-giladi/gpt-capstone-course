# Code requests from Module 8 (for labs/common owner)

Lessons 08.1–08.3 work with the code as it is today. Items are improvements; each says what the
lesson does now.

1. **`mygpt.evaluate` has no micro-batch option.** It evaluates with the checkpoint's training
   `micro_batch_size` (64 for the baseline, logits 64 x 1024 x 50,304), which does not fit a 24 GB
   card. Lesson 08.2 therefore rents an 80 GB H100 for the native bpb runs. Request: `--batch N`
   (default: training micro-batch); windows stay identical as long as `--eval-tokens` is a multiple
   of N x seq_len.
   DONE 2026-11-13 by main maintainer: `mygpt.evaluate --batch N` (default: training micro_batch_size). Lessons 08.1 (4090 row) and 08.2 (step 3, OOM row) use `--batch 16` on 24 GB cards; the H100 stays recommended.
2. **`mygpt.evaluate --eval-tokens all`.** Lesson 08.2 gets the whole-split count from
   `tools/hf_bpb.py` (`mygpt_evaluate_eval_tokens`) and passes it by hand. A built-in `all` that
   covers every full window (as `hf_bpb.full_window_tokens` does, grouped by 64 x seq_len) would
   remove that step. Same as item 3 of the Module 7 requests.
   DONE 2026-11-13 by main maintainer: `--eval-tokens all` / `-1` added (same count as `hf_bpb.full_window_tokens` for batches dividing 64). Lesson 08.2 step 3 drops the grep/copy-the-count step and compares `eval_tokens` in the JSON files.
3. **`[eval]` extra without the HF backend.** `pyproject.toml` has `eval = ["lm-eval>=0.4.13,<0.5"]`;
   recent lm-eval no longer pulls transformers/accelerate itself. Lessons tell learners to install
   `.[post,eval]` (as the labs/common README does). Request: `eval = ["lm-eval[hf]>=0.4.13,<0.5"]`
   so `.[eval]` alone works.
   DONE 2026-11-13 by main maintainer: `eval = ["lm-eval[hf]>=0.4.13,<0.5"]`. Lesson 08.1 and labs/module-08/README install `.[eval]`; `.[post,eval]` in Modules 9-10 still valid.
4. **`scripts/decontam.py` covers four benchmarks.** WinoGrande and LAMBADA are reached through
   `labs/module-08/bench_texts.py` + `--benchmark-file`. Optional: add them to `load_benchmarks`.
   DONE 2026-11-13 by main maintainer: `load_benchmarks` now knows `winogrande` (winogrande_xl validation, blank filled) and `lambada_openai` (alias `lambada`), same items as `bench_texts.py`. Lesson 08.2 step 5 uses `--benchmarks winogrande,lambada_openai`; `bench_texts.py` kept as the offline route.
5. **Windows: lm-eval table printing.** On a cp1252 console lm-eval 0.4.13 crashes with
   `UnicodeEncodeError` after writing results; 08.1 says to set `PYTHONUTF8=1`. Nothing to change
   in our code; noted for the README's Windows section.
   DONE 2026-11-13 by main maintainer: Windows `PYTHONUTF8=1` note added to the labs/common README.
