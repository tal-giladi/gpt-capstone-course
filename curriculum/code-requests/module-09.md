# Code requests from Module 9 (for labs/common owner)

Lessons 09.1 and 09.2 call `scripts/sft.py` and `scripts/chat.py` exactly as they are on
2026-11-13, except for item 1, which the lessons already assume.

1. **BLOCKING — `scripts/sft.py` (and `scripts/dpo.py`) must keep fp32 master weights.**
   Today both scripts load the model with `dtype=torch.bfloat16` whenever the GPU supports bf16 and
   then train with `bf16=True`. The HF Trainer keeps no fp32 copy of the weights in that case, so
   every AdamW update is rounded to bf16. Measured on 2026-11-13 (CPU, torch 2.15): for weights
   ~N(0, 0.02), one update of size lr changes only 1% of bf16 weights at lr 5e-7 (the dpo.py
   default), 2% at 1e-6, 57% at 5e-5 (the sft.py default) and ~100% only from 2e-4 up. DPO with the
   defaults would barely move the model; SFT at small learning rates is silently degraded.
   Fix: load with `dtype=torch.float32` always and keep `bf16=bf16` (autocast mixed precision).
   Memory for a 350M model in fp32 + AdamW is ~5.4 GB (+1.4 GB for the DPO reference model), far
   below 80 GB. Lessons 09.2 and 10.1 say "the scripts load fp32 weights and train with bf16
   autocast"; the one-liner check in 09.2 (`grep -n "dtype=" scripts/sft.py`) expects
   `dtype=torch.float32`. The saved model is then fp32 (~1.35 GB per 337M model); fine.
2. **`scripts/sft.py --packing` needs a FlashAttention model to be correct.** TRL 1.17's default
   `packing_strategy="bfd"` switches on padding-free batches, which the TRL docs say are only
   supported with FlashAttention 2/3. `sft.py` loads the model with the default attention (SDPA),
   so `--packing` risks attention across packed conversations. Lessons tell learners to leave
   `--packing` off. Either add `--attn-implementation` (passed to `from_pretrained`) or make
   `--packing` refuse unless a FlashAttention implementation is set.
3. **Nice to have: `--resume` in `sft.py`/`dpo.py`** passing `resume_from_checkpoint=True` to
   `trainer.train()`. Today a crashed 2-hour SFT run restarts from step 0 (lesson 09.2 tells the
   learner to accept that; the cost is a few dollars).
4. **Nice to have: `--save-final-only`** (or delete `checkpoint-*` folders after `save_model`), so
   `--out` holds only the final model and can be pushed with `sync.sh push-hf` without uploading
   ~4 GB optimizer checkpoints. Lesson 09.2 removes them by hand before pushing.
5. **Nice to have: `--dataset-revision SHA`** in `sft.py`/`dpo.py` (passed as `revision=` to
   `load_dataset`), so a run is pinned to a dataset commit. Lesson 09.1 records the hash in the data
   card and 09.2 asks the learner to compare it by hand on the training day.
