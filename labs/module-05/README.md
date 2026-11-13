# Module 5 lab files — the GPT-2 124M baseline

Helper for lessons 05.1–05.3. Copy it into the `tools/` folder of your project repo and commit
it, so every rented box gets it with `git clone`. The training code it reads the output of
(`mygpt/train.py`, `configs/baseline_124m.yaml`, `scripts/sync.sh`) is in `labs/common/`.

| File | Used in | What it does |
|---|---|---|
| `run_summary.py` | 05.1, 05.2, 05.3 | Reads a run's `log.jsonl` (de-duplicating steps logged twice after a resume) and prints steps, tokens, final and best val loss / val bpb / HellaSwag, median tokens/s and MFU. With `--gpus` and `--single-gpu-tps` it computes multi-GPU scaling efficiency; with `--target-tokens` and `--price-per-gpu-hour` it projects the full run's wall clock and cost; `--compare OTHER_RUN` reports the train-loss difference between two runs at common steps (resume drill); `--markdown` prints a table for `results.md`. Stdlib only. |

Examples:

```bash
python tools/run_summary.py runs/rehearsal --gpus 2 --target-tokens 10e9 --price-per-gpu-hour 4.39
python tools/run_summary.py runs/drill --compare runs/drill_ref
python tools/run_summary.py runs/baseline_124m --gpus 8 --single-gpu-tps 480000 --markdown >> results.md
```

Requirements: Python 3.10+, nothing else (runs on your laptop or on the box). Tested on
2026-11-13 against `log.jsonl` files from `configs/smoke_cpu.yaml` runs, including a resumed run.
