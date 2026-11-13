# TODO for Tal

- Course complete 2026-11-13: 11 modules, 32 lessons, 229 quiz questions; Academy dry-run 0 problems. Ready to import from tal-giladi/gpt-capstone-course (public), commit at HEAD of main.
- Web version: https://tal-giladi.github.io/gpt-capstone-course/
- Facts the agents could not verify on the provider page are labelled in the lessons; see curriculum/audit-modules-01-06.md and audit-modules-07-11.md.

## Price sources checked 2026-11-13 (plan-level; each lesson re-verifies)
- https://www.thundercompute.com/blog/nvidia-h100-pricing
- https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison
- https://www.spheron.network/blog/lambda-cloud-h100-pricing-2026/
- https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/main/sample/10BT
- https://infratally.com/articles/cloudflare-r2-vs-aws-s3-vs-backblaze-b2-2026/

## Not verified locally (no GPU here)
- labs/common CUDA, NCCL/DDP (torchrun), torch.compile, rclone, wandb, real judge API calls, gradio.
  First real check: lesson 04.2 runs `pytest -q` and 04.3 runs `scripts/bench.py` on the first H100 box.
