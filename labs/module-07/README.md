# Module 7 lab files — the main run

Helpers for lessons 07.1–07.3. Copy them into the `tools/` folder of your project repo and commit
them, so the 8×H100 node gets them with `git clone`. The training code they check and watch
(`mygpt/train.py`, `configs/main_350m.yaml`, `scripts/sync.sh`, `scripts/autoshutdown.sh`) is in
`labs/common/`; `tools/run_summary.py` comes from Module 5.

| File | Used in | What it does |
|---|---|---|
| `preflight.py` | 07.1 | One PASS / WARN / FAIL line per check before launch: clean git tree at or after `recipe-v1`, `ckpt_remote` and W&B set, batch divisibility, an empty or resuming `out_dir`, every shard complete (header vs file size), tokenizer and vocab match, `train_tokens` fits in one epoch, same shard count as the bucket, bucket readable and writable (copies one tiny file to `tmp/`), `WANDB_API_KEY`, GPU count / model / NVLink / nothing else running, free disk, `autoshutdown.sh` and a hard stop running, and the 06.4 money rule. Exit code 1 if anything fails. Needs the project venv (reads the config with `mygpt.config`). |
| `watch_run.py` | 07.2, 07.3 | Watches `log.jsonl`: a status line every `--interval` seconds (progress, loss, val bpb, HellaSwag, tokens/s vs the run's median, ETA, cost so far and projected vs plan) and ALERT lines for a stale log, NaN, a loss spike that has not recovered, a grad-norm spike, a sustained slowdown (single dips after evals are ignored) and a cost overrun. `--ntfy TOPIC` sends alerts to your phone through ntfy.sh; `--once` checks once (exit code 2 on alerts); `--markdown` prints the incident table (spikes, slowdowns, resumes, skipped batches) for the run log. Stdlib only. |
| `RUN_LOG.template.md` | 07.1–07.3 | The run log you keep: launch facts, first-15-minutes checks, timeline, one block per incident, close-out. Copy to `reports/main-run-log.md`. |

Examples:

```bash
python tools/preflight.py --config configs/main_350m.yaml --gpus 8 --tag recipe-v1 --budget-left 336 --estimate 187
python tools/watch_run.py runs/main_350m --target-tokens 20e9 --price-per-hour 35.12 \
  --started 2026-11-17T09:05 --plan-hours 5.3 --interval 120 --ntfy mygpt-7f3a9c2e
python tools/watch_run.py runs/main_350m --target-tokens 20e9 --markdown >> reports/main-run-log.md
```

Requirements: Python 3.10+; `preflight.py` also needs the project installed (`pip install -e .`),
`git`, `rclone`, `nvidia-smi` and `ps` on the box. Tested on 2026-11-13 on Windows: `watch_run.py`
against a smoke-run `log.jsonl` (`configs/smoke_cpu.yaml`) and a synthetic 6,000-step main-run log
with an eval dip, a recovered spike, a resume, a slowdown, a `skip_batches` event and an open
spike; `preflight.py` against `configs/smoke_cpu.yaml` with its synthetic shards (the GPU, guard
and bucket checks were not run there: no GPU, no rclone remote).
