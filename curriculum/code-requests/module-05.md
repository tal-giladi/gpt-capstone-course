# Code requests from Module 5 (for labs/common owner)

Lessons 05.1–05.3 work with the code as it is today; each item says what the lesson does now and
what would be better. Nothing here blocks the lessons.

1. **HellaSwag download race under torchrun (`mygpt/evaluate.py: load_hellaswag`).** Every rank
   calls `urllib.request.urlretrieve` into the same `~/.cache/mygpt/hellaswag_val.jsonl` when the
   file is missing, so ranks can read a half-written file (JSONDecodeError) or clobber each other.
   Lessons 05.1/05.3 work around it by pre-downloading once
   (`python -c "from mygpt.evaluate import load_hellaswag; load_hellaswag()"`) before `torchrun`.
   Request: download to a temp file + `os.replace`, and in `train.py` let rank 0 load first and
   `barrier()` before the others.
   DONE 2026-11-13 by main maintainer: load_hellaswag downloads to a per-process temp file + os.replace; train.py loads HellaSwag once at startup on rank 0, barrier, then the other ranks (examples passed to hellaswag()). Lessons 05.1 (text, troubleshooting row, quiz q3) updated; the pre-download step stays as a connectivity check.
2. **`python -m mygpt.evaluate --ckpt RUN_DIR` fails** (`PermissionError`/`IsADirectoryError`: it
   calls `torch.load` on the directory). `labs/common/README.md` documents "file or run dir", and
   `export_hf` accepts a dir. Request: resolve a directory with `latest_checkpoint()`. Module 5
   lessons avoid the CLI (the final eval is in the training log); Module 8 should use a file path
   until fixed.
   DONE 2026-11-13 by main maintainer: `python -m mygpt.evaluate --ckpt RUN_DIR` resolves the newest ckpt_*.pt with latest_checkpoint() (exits with a message if none). Tested.
3. **`baseline_124m.yaml` header comment is stale**: "On 8xH100 expect roughly 1.2-1.5M tok/s ->
   ~2 h wall clock" is the 8×A100 figure. At 40% MFU an H100 SXM does ~460k tok/s per GPU on this
   config (6N + attention = 855 MFLOP/token), i.e. ~3.3–3.7M tok/s per node, ~45–55 min of pure
   training. Lesson 05.3 predicts from the learner's own 04.3 number. Request: update the comment.
   DONE 2026-11-13 by main maintainer: baseline_124m.yaml header now says ~3.3-3.7M tok/s per 8xH100 SXM node at ~40% MFU -> ~45-55 min of training (8xA100 ~2 h).
4. **`log.jsonl` is only pushed to `ckpt_remote` at the end of a run / clean stop.** After a
   SIGKILL the bucket log lags, and if a fresh box resumes with an empty out_dir, the new
   `log.jsonl` (starting at the resume step) overwrites the longer one at the end. Lesson 05.2
   teaches `sync.sh pull-ckpt RUN DIR` before relaunching. Request: (a) push `log.jsonl` together
   with every checkpoint upload; (b) in `rclone_pull_latest`, also pull `log.jsonl` when out_dir has
   none.
   DONE 2026-11-13 by main maintainer: log.jsonl is pushed with every checkpoint upload; rclone_pull_latest also pulls log.jsonl when out_dir has none. Tested (mocked rclone). Lesson 05.2 (checkpoint paragraph, drill C text and box, step 6, troubleshooting row, quiz q1) updated.
5. **No way to skip data on resume after a NaN that repeats at the same data position.** Lesson
   05.2 recovers by rolling back to an earlier checkpoint and lowering `train.lr`. Request (nice to
   have): `train.skip_batches=N` (advance the loader N global batches after resume, logged).
   DONE 2026-11-13 by main maintainer: `train.skip_batches=N` advances the train loader N global batches (N x accum micro-batches) after a resume, prints it and logs {step, skip_batches, skipped_tokens} to log.jsonl. Applied once: not re-applied when resuming from a checkpoint written with the same N; ignored on a fresh run. Tested. Lesson 05.2 rollback procedure unchanged (still valid).
6. **Rank seeding note (no change needed):** `torch.manual_seed(t.seed + rank)` is called before
   the model is built, so ranks initialise different weights; DDP broadcasts rank 0's parameters
   at construction, so this is correct. A comment in `train.py` would save readers the worry.
   DONE 2026-11-13 by main maintainer: Comment added in train.py at the seeding line.
