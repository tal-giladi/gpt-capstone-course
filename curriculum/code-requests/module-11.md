# Code requests from Module 11 (for labs/common owner)

1. **Nice to have — `mygpt/export_hf.py --push` should create the repo private.** `HfApi().create_repo(args.push, ...)`
   creates a public repo for a personal account. Lessons 07.3 and 11.1 warn about it and create repos with
   `hf repos create --private` first, but a `--private` flag (default on) would remove the trap:
   `create_repo(args.push, repo_type="model", private=args.private, exist_ok=True)`.
2. **Nice to have — `[demo]` extra.** Lesson 11.2's Space pins `torch==2.13.0` (newest on ZeroGPU's supported list
   on 2026-11-13) while the project uses torch 2.15; nothing to change in `pyproject.toml`, just noting that the
   Space's `requirements.txt` in `labs/module-11/space/` is deliberately separate from the project pins.
