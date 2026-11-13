# Module 4 lab files — renting and benchmarking GPUs

Helpers for lessons 04.2 and 04.3. Copy them into the `tools/` folder of your project repo and
commit them, so every rented box gets them with `git clone`. The project code they drive
(`scripts/bench.py`, `scripts/autoshutdown.sh`, `configs/*.yaml`) is in `labs/common/`.

| File | Used in | What it does |
|---|---|---|
| `setup_box.sh` | 04.2, every later GPU lesson | Installs git/tmux/rclone/uv, clones your repo, creates the pinned `.venv`, runs the CPU tests and `gpu_check.py`. Idempotent. |
| `gpu_check.py` | 04.2 | Driver vs CUDA check (CUDA 13 wheels need driver >= 580), GPU count, free memory, bf16 matmul TFLOP/s vs peak, NVLink topology, free disk. Exit 1 on any problem. |
| `budget_calc.py` | 04.3 | Turns measured tokens/s (or an assumed MFU) into GPU-hours, wall-clock time and dollars for baseline, ablations and main run; prints a markdown table for `budget.md`. Stdlib only. |

Requirements: Linux GPU box with an NVIDIA driver, Python 3.12 via uv, the project installed.
`budget_calc.py` runs anywhere with Python 3.10+. Checked 2026-11-13 against RunPod docs, the
NVIDIA H100 spec page and CUDA 13.0 release notes.

No secrets live in these files. Keys come from environment variables (RunPod secrets or a
`chmod 600` env file copied to the box), never from the repo.
