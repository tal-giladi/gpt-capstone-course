# Main run log: main_350m

Copy to `reports/main-run-log.md` in your project repo at launch (07.1), add a line to the
timeline every time you look at the run or act on it (07.2), and close it at the end (07.3).
Times in your local time zone. Keep it factual: what you saw, what you did, what it cost.

## Launch (07.1)

| Item | Value |
|---|---|
| Commit / tag | `<git rev-parse --short HEAD>` / recipe-v1 |
| Config | configs/main_350m.yaml (+ overrides: none) |
| Provider, node, region | <RunPod Secure 8x H100 SXM, EU-RO-1> |
| Price | $<35.12>/h (checked <date time>) |
| Node started | <2026-11-17 09:05> |
| Predicted billed hours / cost | <5.3 h> / $<187> |
| Money left before launch | $<336> |
| Hard stop at | <09:05 + 10.6 h = 19:41> |
| Bucket | r2:mygpt/ckpt/main_350m/ |
| W&B run | <link> |
| ntfy topic | <mygpt-xxxxxxxx> (do not publish) |
| Preflight | READY, <n> WARN: <which> |

## First 15 minutes

| Check | Expected | Seen |
|---|---|---|
| Header line | params ~337M, world 8, accum 2, steps 38146 | |
| Loss step 1 | ~10.4 (ln 32768) | |
| Loss step 300 | < ~6 | |
| Node tokens/s after compile | 8 x <04.3 tok/s> x <05.3 scaling> = | |
| MFU | close to 04.3 | |
| First checkpoint in bucket | ckpt_00001000.pt ~2.9 GB | |

## Timeline

| Time | Step | Event | Action | Cost impact |
|---|---|---|---|---|
| 09:05 | - | node started | preflight READY | |
| 09:20 | 1,000 | first checkpoint uploaded | none | |
| | | | | |

## Incidents

One block per incident (spike, NaN, slowdown, node death). Paste the matching rows from
`python tools/watch_run.py runs/main_350m --target-tokens 20e9 --markdown`.

### <YYYY-MM-DD HH:MM> <short title>

- Symptom:
- Evidence (log lines, grad norm, nvidia-smi, dmesg):
- Decision and why:
- Commands run:
- Steps / tokens lost or repeated:
- Extra cost:

## Close-out (07.3)

| Item | Value |
|---|---|
| Final step / tokens | |
| Final val loss / val bpb / HellaSwag (in-training) | |
| Wall clock (node start to terminate) | |
| Billed cost (provider usage page) | |
| Predicted vs real | |
| Resumes / skipped batches / LR changes | |
| Final checkpoint in bucket (name, size) | |
| HF export (class, max logit diff) | |
| Node terminated at | |
