# TODO for Tal

- Plan approved 2026-11-13. Next: build the lessons.

## Price sources checked 2026-11-13 (plan-level; each lesson re-verifies)
- https://www.thundercompute.com/blog/nvidia-h100-pricing
- https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison
- https://www.spheron.network/blog/lambda-cloud-h100-pricing-2026/
- https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/main/sample/10BT
- https://infratally.com/articles/cloudflare-r2-vs-aws-s3-vs-backblaze-b2-2026/

## Not verified locally (no GPU here)
- labs/common CUDA, NCCL/DDP (torchrun), torch.compile, rclone, wandb, real judge API calls, gradio.
  First real check: lesson 04.2 runs `pytest -q` and 04.3 runs `scripts/bench.py` on the first H100 box.
