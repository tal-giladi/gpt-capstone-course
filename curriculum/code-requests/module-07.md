# Code requests from Module 7 (for labs/common owner)

Lessons 07.1–07.3 work with the code as it is today. Items are improvements; each says what the
lesson does now.

1. **`export_hf --push` creates a public repo on personal accounts.** `HfApi().create_repo(args.push,
   repo_type="model", exist_ok=True)` uses the Hub default visibility. Lesson 07.3 tells learners to
   create the repo with `hf repos create ... --private` first and then `hf upload`. Request: add
   `--private` to `export_hf` (passed to `create_repo(private=True)`), or make private the default
   until Module 11.
   DONE 2026-11-13 by main maintainer: `export_hf --push` now creates the repo private by default (`--private/--no-private`, passed to `create_repo(private=...)`), prints the repo's actual visibility, warns if a pre-existing repo is public, and prints `hf repos settings <id> --public` for Module 11. Lesson 07.3 uses `export_hf --push` for the optional private copy; 11.1 uses `--public` (the old `--private false` was not a valid flag).
2. **Nice to have: `train.wandb` alerts.** `train.py` does not call `run.alert()`. Lesson 07.1 uses
   W&B's built-in "run crashed / finished" emails plus `tools/watch_run.py` (ntfy) instead. A
   `run.alert()` on a non-finite loss and on the final step would add an off-node alert for NaN stops.
   DONE 2026-11-13 by main maintainer: `train.py` calls `run.alert()` (level ERROR) on a non-finite loss/grad stop, naming the last good checkpoint, and (INFO) after the final step, when W&B is on; new `train.wandb_alerts` (default true) turns them off; failures never stop training. Lesson 07.1 adds the W&B "Scriptable run alerts" setting.
3. **Nice to have: `mygpt.evaluate --eval-tokens -1` (whole val split).** Module 8 evaluates the
   whole common val set by passing the split's token count from `meta.json`; a `-1` shortcut would
   remove that step (see Module 8 requests).
   DONE 2026-11-13 by main maintainer: `mygpt.evaluate --eval-tokens -1` (or `all`) evaluates every full val window, grouped by lcm(64, batch) x seq_len like `hf_bpb.py`; the output JSON now includes `eval_tokens`, `batch`, `seq_len`.
