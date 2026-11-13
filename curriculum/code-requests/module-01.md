# Code requests from Module 1 (for labs/common owner)

Lesson 01.3 builds the learner's repo by copying the contents of `labs/common/` to the repo root
and adding `tools/` (Module 1 helpers), `.gitignore` (from `labs/module-01/project.gitignore`)
and `budget.md`. It then runs, on a laptop:

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]" --torch-backend cpu
pytest -q
python -m mygpt.train --config configs/smoke_cpu.yaml
python -m mygpt.train --config configs/smoke_cpu.yaml train.train_tokens=40960   # must resume at step 21
```

1. **README.md layout statement.** `labs/common/README.md` says
   `git clone <your project repo> && cd <repo>/labs/common`. The course convention (lesson spec,
   Modules 2 and 4) is that the project repo *is* a copy of labs/common at its root. Please change
   that line to `git clone <your project repo> && cd <repo>`.
2. **Keep `README.md` in labs/common** (pyproject `readme = "README.md"`; install fails without it).
3. **Lesson quotes this output** (measured 2026-11-13, Windows 11, torch 2.15 CPU): first line
   `[train] params 0.1M | world 1 | micro 8x64 | accum 2 | steps 20 | device cpu | dtype fp32`,
   step 1 loss 5.5691, step 20 loss 4.2751, val_loss 4.2578; resume prints
   `[train] resumed from runs/smoke_cpu/ckpt_00000020.pt at step 20` and ends near loss 3.43 at
   step 40; `pytest -q` -> `100 passed in 103.46s`. If you change the smoke config, the
   seed or the log format, tell me.
4. **`.gitignore`.** `project.gitignore` contains every entry of labs/common's `.gitignore` plus
   secrets (`.env`, `*.env`, keys), `wandb/`, `*.safetensors`, `*.parquet`. If you add entries,
   tell me so the two stay in step.
