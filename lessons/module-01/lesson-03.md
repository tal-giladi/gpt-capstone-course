---
id: "01.3"
module: 1
minutes: 15
practice_minutes: 45
prerequisites: ["01.1", "01.2"]
objectives:
  - Create your private project repo on GitHub from the course's lab code, with the Module 1 tools, a .gitignore and budget.md.
  - Install the pinned environment with uv and a CPU-only PyTorch, and run the test suite green.
  - Run the CPU smoke training, read its output, and prove that checkpoint resume continues where it stopped.
  - Verify that no secret is tracked in the repo before the first push.
volatility: implementation
sources:
  - title: "uv installation"
    url: https://docs.astral.sh/uv/getting-started/installation/
  - title: "uv: using uv with PyTorch (--torch-backend)"
    url: https://docs.astral.sh/uv/guides/integration/pytorch/
  - title: "uv releases (0.13.x, October 2026)"
    url: https://github.com/astral-sh/uv/releases
  - title: "GitHub CLI: gh repo create"
    url: https://cli.github.com/manual/gh_repo_create
  - title: "W&B quickstart (API key via WANDB_API_KEY)"
    url: https://docs.wandb.ai/quickstart/
last_verified: "2026-11-13"
---

# 01.3 · Your project repo and a CPU dry run

Every machine you rent in this course starts the same way: `git clone` your repo, install the pinned environment, run the tests. This lesson builds that repo from the course's lab code, installs it on your laptop, and runs a two-layer GPT on synthetic data for twenty steps, so the whole loop of config, training, validation, checkpoint and resume works before any of it costs money. If something is broken here, it is free to fix; on an 8×H100 node the same mistake costs $35 an hour.

## Goal

You have a private GitHub repo (called `mygpt` in this course) containing the `mygpt` package, scripts, configs and tests from `labs/common`, the Module 1 tools in `tools/`, a `.gitignore` that keeps secrets and data out, and your `budget.md`. On your laptop `pytest -q` passes, the CPU smoke run trains and its loss falls, a second run resumes from the checkpoint, and `check_accounts.py` confirms no secret is tracked.

## What you need

- Accounts and `~/.config/mygpt/secrets.env` from [01.2](lesson-02.md); `budget.md` from [01.1](lesson-01.md).
- The course lab files: the Module 1 zip contains `labs/common/` (the project code) and `labs/module-01/` (the tools).
- A laptop with Linux, macOS, or Windows with WSL2 (recommended on Windows, because every rented box runs Linux and the lessons use bash). About 3 GB free disk, 8 GB RAM, internet.
- git 2.40+ and, optionally, the GitHub CLI `gh`.
- About 60 minutes.
- **Expected cost: $0.**

## Choices and costs

| Decision | Options | Pick |
|---|---|---|
| Where the dry run runs | Your laptop (free) or a cloud box (from ~$0.01/h) | **Laptop.** The smoke config is a 0.1M-parameter model on synthetic data; it trains in under a minute on any CPU |
| Python tooling | uv, pip + venv, conda | **uv** (0.13.x as of 2026-11-13). One static binary, fast installs, installs Python itself, and picks the right PyTorch wheel with `--torch-backend`. The rented boxes use the same commands (`setup_box.sh` in 04.2) |
| PyTorch build on the laptop | CPU wheel (~200 MB) or the default CUDA wheel (several GB of `nvidia-*` packages) | **CPU** via `--torch-backend cpu`. The GPU boxes install their own CUDA 13.0 wheels later |
| Python version | 3.11–3.13 supported by the project (uv 0.13 defaults to a newer Python) | **3.12**, the same on every box, pinned explicitly |
| Repo visibility | Private or public | **Private** until Module 11, when you clean it up and publish it as part of your portfolio |

All $0.

## Do this

### 1. Install uv and check your tools (5 min)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # Linux, macOS, WSL
# Windows PowerShell (only if you don't use WSL):
# powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv --version        # expect uv 0.13.x or newer
git --version
```

### 2. Assemble the repo (10 min)

Unzip the course lab files somewhere outside your future repo (for example `~/Downloads/gpt-labs/`), then:

```bash
LABS=~/Downloads/gpt-labs/labs          # where you unzipped them
mkdir -p ~/code/mygpt && cd ~/code/mygpt
cp -r "$LABS"/common/. .                 # mygpt/, scripts/, configs/, tests/, pyproject.toml, README.md
mkdir -p tools
cp "$LABS"/module-01/check_accounts.py "$LABS"/module-01/secrets.env.example tools/
cp "$LABS"/module-01/project.gitignore .gitignore
cp ~/mygpt-start/budget.md .            # your budget from 01.1
rm -rf .venv *.egg-info .pytest_cache    # in case the zip carried build leftovers
ls
```

You should see `budget.md configs mygpt pyproject.toml README.md scripts tests tools` (plus the hidden `.gitignore`). From now on you add each module's lab helpers to `tools/` and commit them, so every rented box gets them with `git clone`.

### 3. Create the environment and install (5–10 min)

```bash
uv venv --python 3.12
source .venv/bin/activate                # Windows PowerShell: .venv\Scripts\activate
uv pip install -e ".[dev]" --torch-backend cpu
python -c "import torch, mygpt; print(torch.__version__, torch.cuda.is_available())"
```

The last line prints the PyTorch version followed by `False` (no CUDA on the laptop, as intended).

### 4. Run the tests (2–3 min)

```bash
pytest -q
```

The first run downloads the GPT-2 BPE files used by `tiktoken` (a few MB), so it needs internet.

### 5. The CPU smoke run (1 min)

```bash
python -m mygpt.train --config configs/smoke_cpu.yaml
```

Open `configs/smoke_cpu.yaml` while it runs: a 2-layer, 64-wide GPT with a 256-token vocabulary on synthetic Markov-chain data, 20 steps of 1,024 tokens, fp32, no compile, checkpoints every 10 steps into `runs/smoke_cpu/`. The same `train` section exists in the real configs; only the numbers change.

### 6. Prove resume works (1 min)

Ask for 40 steps instead of 20. The trainer finds the step-20 checkpoint in `runs/smoke_cpu/` and continues instead of starting over:

```bash
python -m mygpt.train --config configs/smoke_cpu.yaml train.train_tokens=40960
ls runs/smoke_cpu/
```

`key=value` overrides like this are how every later lesson changes a config without editing it.

### 7. Optional: see a run in W&B (3 min)

```bash
set -a; . ~/.config/mygpt/secrets.env; set +a
python -m mygpt.train --config configs/smoke_cpu.yaml train.wandb_project=mygpt-smoke \
  train.run_name=smoke-wandb train.out_dir=runs/smoke-wandb
```

Open wandb.ai: project `mygpt-smoke` shows a run `smoke-wandb` with the loss curve.

### 8. Check for secrets, commit and push (10 min)

```bash
set -a; . ~/.config/mygpt/secrets.env; set +a
git init -b main
git add .
git status --short | head -50                  # read it: no .env, no runs/, no .venv/
python tools/check_accounts.py                 # the git line must say OK
git commit -m "Project skeleton from course lab code, budget, Module 1 tools"
```

Create the private GitHub repo and push, with the GitHub CLI:

```bash
gh auth login                                   # once, in the browser
gh repo create mygpt --private --source . --push
```

or on github.com: **New repository** → name `mygpt`, **Private**, no README/license/.gitignore (your repo already has them) → then `git remote add origin git@github.com:<you>/mygpt.git && git push -u origin main`.

> [!WARNING]
> Run `git status` and `check_accounts.py` **before** the first push, every time you add files with `git add .`. Push protection on GitHub only catches known key formats on public repos; your private repo relies on you and the `.gitignore`.

### 9. Ledger and README

Add the repo URL and your bucket name placeholder (`r2:mygpt`, created in 02.2) to `README.md` under a heading **My project**, add the ledger line below to `budget.md`, commit and push.

## How you know it worked

- `pytest -q` ends with a line like `100 passed in 103.46s` (the count grows as the course adds tests; anything with `failed` or `error` is not done). On a 2024–2026 laptop it takes one to two minutes.
- The smoke run prints a parameter line and falling loss, ending in about half a minute. Measured on 2026-11-13 on a Windows laptop:

  ```text
  [train] params 0.1M | world 1 | micro 8x64 | accum 2 | steps 20 | device cpu | dtype fp32
  step      1/20 | loss 5.5691 | lr 5.00e-03 | gnorm 0.82 | 5.6k tok/s | mfu 0.0%
  step     10 | val_loss 4.8439 | val_bpb 6.9883
  step     20/20 | loss 4.2751 | lr 1.07e-03 | gnorm 0.69 | 6.4k tok/s | mfu 0.0%
  step     20 | val_loss 4.2578 | val_bpb 6.1428
  ```

  The first loss is about $\ln 256 \approx 5.55$: an untrained model guessing uniformly over 256 tokens. It falls toward about 4.3 in 20 steps. Your numbers may differ in the second decimal.
- The resume run prints `[train] resumed from runs/smoke_cpu/ckpt_00000020.pt at step 20`, continues at step 21 (not step 1) and ends near loss 3.4 at step 40. `runs/smoke_cpu/` holds the last two checkpoints and `log.jsonl`.
- `check_accounts.py` prints `OK    git  no secret values in tracked files`.
- On GitHub the repo is marked **Private** and shows your files, with no `runs/`, `.venv/` or `.env` files.

## When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `uv pip install` downloads `nvidia-cublas` and other `nvidia-*` packages | `--torch-backend cpu` missing | `uv pip uninstall torch` and reinstall with the flag |
| `No interpreter found for Python 3.12` | uv could not download Python (proxy, offline) | `uv python install 3.12`; check your network; on corporate networks set `HTTPS_PROXY` |
| `README.md` or `pyproject.toml` not found during install | You copied the `common` folder itself instead of its contents | Copy `common/.` into the repo root so `pyproject.toml` is at the top level |
| `pytest` fails in a tokenizer test with a network error | First `tiktoken` use downloads the GPT-2 files | Run again with internet access; behind a proxy set `HTTPS_PROXY` |
| Smoke loss does not fall, or is `nan` | You edited `smoke_cpu.yaml` (for example `lr`), or ran with an override typo | `git checkout configs/smoke_cpu.yaml`; check that overrides are `train.key=value` |
| The second run starts again at step 1 | Different `train.out_dir`, or you deleted `runs/smoke_cpu/` | Resume looks only in `out_dir`; use the same one |
| `git status` shows `.venv/`, `runs/` or `secrets.env` | `.gitignore` missing or not at the repo root | Copy `project.gitignore` to `.gitignore` at the root; `git rm -r --cached <path>` for anything already added |
| `check_accounts.py` reports a secret in a tracked file | A key was pasted into a file inside the repo | Do not push. Remove the file from the index, revoke and replace the key (01.2), then commit |
| `gh repo create` fails with `Name already exists` | You already have a `mygpt` repo | Pick another name and use it wherever the course says `<your-repo>` |

## Artifact

- Private GitHub repo `<you>/mygpt` with the project code, `tools/check_accounts.py`, `.gitignore`, `README.md` and `budget.md`, pushed.
- A green `pytest -q` and a smoke run that resumed, both on your laptop.
- `budget.md` ledger line:

  ```text
  | 2026-11-13 | 01.3 | Project repo, local install, tests and CPU smoke run | $0.00 | $0.00 |
  ```
