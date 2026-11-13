"""Check that your capstone accounts and secrets work, without ever printing a secret.

    set -a; . ~/.config/mygpt/secrets.env; set +a     # and r2.env once you have it (02.2)
    python tools/check_accounts.py

Checks (each one is skipped if its variable is not set yet):
  HF_TOKEN           Hugging Face whoami: user name and token type (warns on a write token)
  WANDB_API_KEY      Weights & Biases: the user the key belongs to
  ANTHROPIC_API_KEY  lists models (free call, no tokens billed)
  OPENAI_API_KEY     lists models (free call, no tokens billed)
  RCLONE_CONFIG_R2_* lists the bucket root with rclone (from lesson 02.2 on)
  secret files       ~/.config/mygpt/*.env readable only by you (Linux/macOS)
  git                no tracked file in the current repo contains any of the secret values

Standard library only. Exit code 1 if any check fails. Checked against the HF, W&B, Anthropic
and OpenAI APIs on 2026-11-13.
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import stat
import subprocess
import sys
import urllib.error
import urllib.request

SECRET_VARS = ["HF_TOKEN", "WANDB_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
               "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY", "RCLONE_CONFIG_R2_ACCESS_KEY_ID"]
results: list[tuple[str, str, str]] = []


def report(status: str, name: str, detail: str) -> None:
    results.append((status, name, detail))
    print(f"{status:5s} {name:18s} {detail}", flush=True)


def http(url: str, headers: dict, data: bytes | None = None, timeout: int = 20):
    req = urllib.request.Request(url, data=data, headers={"User-Agent": "mygpt-check/1.0", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def http_error(e: Exception) -> str:
    if isinstance(e, urllib.error.HTTPError):
        return f"HTTP {e.code} (wrong, expired or revoked key?)"
    return f"{type(e).__name__}: {e}"


def check_hf():
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        return report("SKIP", "Hugging Face", "HF_TOKEN not set")
    try:
        d = http("https://huggingface.co/api/whoami-v2", {"Authorization": f"Bearer {tok}"})
    except Exception as e:  # noqa: BLE001
        return report("FAIL", "Hugging Face", http_error(e))
    role = (d.get("auth", {}).get("accessToken", {}) or {}).get("role", "?")
    if role == "write":
        return report("WARN", "Hugging Face", f"user={d.get('name')} token=write: use a fine-grained read token on rented boxes")
    report("OK", "Hugging Face", f"user={d.get('name')} token={role}")


def check_wandb():
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        return report("SKIP", "Weights & Biases", "WANDB_API_KEY not set")
    auth = base64.b64encode(f"api:{key}".encode()).decode()
    try:
        d = http("https://api.wandb.ai/graphql", {"Authorization": f"Basic {auth}", "Content-Type": "application/json"},
                 json.dumps({"query": "{ viewer { username entity } }"}).encode())
    except Exception as e:  # noqa: BLE001
        return report("FAIL", "Weights & Biases", http_error(e))
    v = (d.get("data") or {}).get("viewer")
    if not v:
        return report("FAIL", "Weights & Biases", "key not accepted")
    report("OK", "Weights & Biases", f"user={v.get('username')} entity={v.get('entity')}")


def check_anthropic():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return report("SKIP", "Anthropic API", "ANTHROPIC_API_KEY not set (only needed in Module 10)")
    try:
        d = http("https://api.anthropic.com/v1/models", {"x-api-key": key, "anthropic-version": "2023-06-01"})
    except Exception as e:  # noqa: BLE001
        return report("FAIL", "Anthropic API", http_error(e))
    report("OK", "Anthropic API", f"{len(d.get('data', []))} models visible")


def check_openai():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return report("SKIP", "OpenAI API", "OPENAI_API_KEY not set (only needed in Module 10)")
    try:
        d = http("https://api.openai.com/v1/models", {"Authorization": f"Bearer {key}"})
    except Exception as e:  # noqa: BLE001
        return report("FAIL", "OpenAI API", http_error(e))
    report("OK", "OpenAI API", f"{len(d.get('data', []))} models visible")


def check_r2():
    if not os.environ.get("RCLONE_CONFIG_R2_TYPE"):
        return report("SKIP", "R2 bucket", "r2.env not loaded (created in lesson 02.2)")
    if not shutil.which("rclone"):
        return report("FAIL", "R2 bucket", "rclone not installed")
    remote = os.environ.get("MYGPT_REMOTE", "r2:mygpt")
    p = subprocess.run(["rclone", "lsf", remote, "--max-depth", "1"], capture_output=True, text=True, timeout=60)
    if p.returncode != 0:
        return report("FAIL", "R2 bucket", (p.stderr.strip().splitlines() or ["rclone error"])[-1][:120])
    report("OK", "R2 bucket", f"{remote}: {len(p.stdout.split())} top-level entries")


def check_secret_files():
    d = os.path.expanduser("~/.config/mygpt")
    if not os.path.isdir(d):
        return report("SKIP", "secret files", f"{d} does not exist yet")
    if os.name == "nt":
        return report("SKIP", "secret files", "permission check needs Linux/macOS/WSL")
    bad = [f for f in os.listdir(d) if f.endswith(".env")
           and os.stat(os.path.join(d, f)).st_mode & (stat.S_IRWXG | stat.S_IRWXO)]
    if bad:
        return report("FAIL", "secret files", f"readable by others: {', '.join(bad)} -> chmod 600")
    report("OK", "secret files", f"{d}/*.env are private")


def check_git():
    if not shutil.which("git"):
        return report("SKIP", "git", "git not installed")
    p = subprocess.run(["git", "ls-files", "-z"], capture_output=True, timeout=60)
    if p.returncode != 0:
        return report("SKIP", "git", "not inside a git repo")
    values = [os.environ[v] for v in SECRET_VARS if len(os.environ.get(v, "")) >= 12]
    leaks = []
    for path in filter(None, p.stdout.decode("utf-8", "replace").split("\0")):
        if path.endswith(".env") and not path.endswith(".env.example"):
            leaks.append(f"{path} (env file tracked)")
            continue
        try:
            if os.path.getsize(path) > 5_000_000:
                continue
            with open(path, "rb") as f:
                blob = f.read()
        except OSError:
            continue
        if any(v.encode() in blob for v in values):
            leaks.append(path)
    if leaks:
        return report("FAIL", "git", "secret found in tracked file(s): " + ", ".join(leaks[:5]))
    report("OK", "git", f"no secret values in tracked files ({len(values)} secrets checked)")


def main() -> int:
    for check in (check_hf, check_wandb, check_anthropic, check_openai, check_r2, check_secret_files, check_git):
        try:
            check()
        except Exception as e:  # noqa: BLE001
            report("FAIL", check.__name__.replace("check_", ""), f"{type(e).__name__}: {e}")
    failed = [r for r in results if r[0] == "FAIL"]
    print(f"\n{len(results) - len(failed)} ok/skipped/warned, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
