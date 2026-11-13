#!/usr/bin/env bash
# Prepare a fresh Linux GPU box for the project (lesson 04.2). Idempotent: safe to run twice.
#
#   git clone https://github.com/<you>/<repo>.git ~/mygpt && bash ~/mygpt/tools/setup_box.sh
#
# Expects in the environment (RunPod secrets, or `set -a; . ~/r2.env; set +a` before running):
#   RCLONE_CONFIG_R2_* and MYGPT_REMOTE (lesson 02.2). HF_TOKEN is optional here.
# Installs tmux, git, rclone and uv; creates .venv in the repo with the pinned dependencies;
# runs the CPU tests and tools/gpu_check.py.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"

echo "== system packages"
if command -v apt-get >/dev/null; then
  $SUDO apt-get update -qq
  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git tmux curl unzip htop >/dev/null
fi
command -v rclone >/dev/null || curl -fsSL https://rclone.org/install.sh | $SUDO bash
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
rclone version | head -1
uv --version

echo "== project in $REPO_DIR"
cd "$REPO_DIR"
[ -d .venv ] || uv venv --python 3.12
# shellcheck disable=SC1091
. .venv/bin/activate
uv pip install -e ".[dev]"
python -c "import torch; print('torch', torch.__version__, 'built for CUDA', torch.version.cuda)"

echo "== bucket"
if [ -n "${RCLONE_CONFIG_R2_TYPE:-}" ]; then
  rclone lsf "${MYGPT_REMOTE:-r2:mygpt}/" | head -20
else
  echo "WARNING: RCLONE_CONFIG_R2_* not set; bucket access not configured"
fi

echo "== tests (CPU, < 2 min)"
pytest -q

echo "== GPU health"
python tools/gpu_check.py
echo "Setup done. Next time: cd $REPO_DIR && . .venv/bin/activate"
