# Course details (not imported)

- Slug: `gpt-capstone` — repo `tal-giladi/gpt-capstone-course` (public).
- Free or paid: suggest paid (Tal decides).
- Risk notice: This course spends real money on rented cloud GPUs; a forgotten machine can cost tens of dollars per hour, so set spending limits before you start.
- One-line summary: Rent real GPUs, use real data, and train, align, evaluate and publish your own GPT-2-class model for under $500.
- For: engineers who finished an LLM training course (transformers, BPE, AdamW, DDP, SFT, DPO) and want to do it for real.
- Format: 11 modules, 32 guided lessons, each ending in a real artifact; quizzes per lesson and module.
- Prerequisites: PyTorch, transformer internals, distributed training basics, a credit card and ~$500.
- What you will learn:
  - Pick, license and store a pretraining dataset (FineWeb-Edu) in object storage
  - Train your own BPE tokenizer and tokenize billions of tokens on rented CPUs
  - Compare GPU clouds by live price and benchmark real throughput and MFU
  - Reproduce GPT-2 124M on 8×H100 with checkpoint/resume
  - Choose model improvements with fair ablations and run a ~350M main run
  - Evaluate honestly against GPT-2 with lm-eval-harness
  - Instruction-tune (SFT) and preference-tune (DPO) your model
  - Publish the model, a live demo and a technical report with a full cost ledger
