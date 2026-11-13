# Module 6 lab files — ablations

Helpers for lessons 06.1–06.4. Copy them into the `tools/` folder of your project repo and
commit them, so every rented box gets them with `git clone`. The training code they drive
(`mygpt/train.py`, `configs/ablation_base.yaml`, `configs/main_350m.yaml`) is in `labs/common/`.

| File | Used in | What it does |
|---|---|---|
| `ablations.txt` | 06.1–06.3 | The ablation queue: one `NAME OVERRIDES...` line per run on top of `configs/ablation_base.yaml`. Controls first; later blocks are commented out until you reach their lesson. `@ARCH ...` defines a macro for the 06.3 runs. |
| `run_ablations.sh` | 06.1–06.3 | Runs the active lines one after another on one GPU (`python -m mygpt.train ... train.out_dir=runs/NAME train.run_name=NAME`). Skips runs marked `DONE` locally or in the bucket, resumes unfinished ones, continues after a failed run, copies `log.jsonl` + `DONE` to `$MYGPT_REMOTE/ckpt/NAME/`. `DRY_RUN=1` prints the commands; `WANDB_PROJECT=...` logs to W&B; `PUSH_CKPT=1` also uploads checkpoints. |
| `ablation_table.py` | 06.1–06.4 | Reads every run's `log.jsonl`, takes `val_bpb` (or `--metric val_loss`/`hellaswag`) at the same step for all runs, subtracts the control mean, estimates noise from the spread of the control seeds and labels each run better / worse / within noise. Adds tokens/s, GPU-hours and cost per run. Prints the markdown table for `results/ablations.md`. Stdlib only. |

Examples:

```bash
DRY_RUN=1 bash tools/run_ablations.sh tools/ablations.txt         # check the commands first
WANDB_PROJECT=my-gpt-ablations bash tools/run_ablations.sh tools/ablations.txt
python tools/ablation_table.py "runs/abl_*" --queue tools/ablations.txt --price-per-gpu-hour 2.69 > results/ablations.md
python tools/ablation_table.py "runs/abl_*" --queue tools/ablations.txt --control abl_qwen3 --noise 0.0025   # 06.3: single new control, noise from 06.1
```

Requirements: bash, the project installed in `.venv`, rclone with the `r2` remote for the bucket
copies (optional; without it results stay on the box). `ablation_table.py` needs only Python
3.10+. Tested on 2026-11-13 on Windows (Git Bash) with `CONFIG=configs/smoke_cpu.yaml`, including
a deliberately diverging run (`train.lr=1e6`, exit code 1) and a rerun that skipped finished runs.
