---
id: "04.2"
module: 4
minutes: 15
practice_minutes: 75
prerequisites: ["01.3", "02.2", "04.1"]
objectives:
  - Launch a 1×H100 SXM pod from a saved template with secrets injected as environment variables.
  - Verify driver, CUDA, GPU health, bucket access and network speed before spending GPU time on training.
  - Install the pinned project environment with uv and run a GPU smoke test.
  - Arm two independent automatic stops and prove that one of them actually stops the pod.
volatility: implementation
sources:
  - title: "RunPod: use SSH (basic vs exposed TCP)"
    url: https://docs.runpod.io/pods/configuration/use-ssh
  - title: "RunPod: manage pods (stop, terminate, scheduled stop)"
    url: https://docs.runpod.io/pods/manage-pods
  - title: "runpodctl stop pod"
    url: https://docs.runpod.io/runpodctl/reference/runpodctl-stop-pod
  - title: "RunPod secrets"
    url: https://docs.runpod.io/pods/templates/secrets
  - title: "RunPod pod pricing and billing"
    url: https://docs.runpod.io/pods/pricing
  - title: "CUDA Toolkit 13.0 release notes (minimum driver)"
    url: https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/
  - title: "PyTorch: CUDA 13.0 as the stable PyPI wheels"
    url: https://dev-discuss.pytorch.org/t/transitioning-pypi-cuda-wheels-to-cuda-13-0-as-the-stable-release-2-11/3325
last_verified: "2026-11-13"
---

# 04.2 · Your first H100 box

The first hour on a rented GPU decides whether the next hundred go well. Most expensive failures in projects like this are not exotic: a driver too old for the PyTorch wheel, a host whose GPU is half as fast as it should be, a network that needs 40 minutes to pull your shards while seven other GPUs wait, or a box that nobody stopped. This lesson launches one H100, checks each of those in minutes, and leaves you with a template and a setup script that make every later box a five-minute job.

## Goal

You launch a 1×H100 SXM RunPod pod from your own saved template, confirm it is healthy (driver >= 580, all checks in `gpu_check.py` pass, bf16 matmul near the expected speed), measure its download speed from R2, install the pinned environment with uv, run the GPU smoke test, and watch the auto-stop guard stop the pod by itself. You commit an environment record so later runs can be compared to this one.

## What you need

- RunPod account with SSH key, credit and secrets from [04.1](lesson-01.md).
- Your project repo on GitHub with the Module 4 tools copied into `tools/` (download the [Module 4 lab files](../../labs/module-04/) and commit them), and the bucket from [02.2](../module-02/lesson-02.md) with the raw data from [02.3](../module-02/lesson-03.md).
- About 90 minutes, of which ~60 on the GPU.
- **Expected cost: $3–4** (about 1.2 h of Community Cloud H100 SXM at $2.69/h; $5–6 on Secure Cloud at $4.39/h).

## Choices and costs

| Choice | Options | Pick for this lesson |
|---|---|---|
| GPU | H100 SXM ($2.69 Community / $4.39 Secure), H100 PCIe ($1.99 / $2.89), RTX 4090 ($0.34 / $0.89) | **H100 SXM**: same chip as the 8-GPU nodes, so 04.3's numbers transfer |
| Cloud tier | Community (cheaper, single hosts) or Secure (datacenter) | **Community** if an H100 SXM with CUDA 13 is listed, else Secure |
| Image | Official RunPod PyTorch templates (Ubuntu 22.04/24.04, CUDA 12.9/13.0/13.2 variants) or a custom image | Official PyTorch template with a **CUDA 13.x** tag; the project installs its own pinned torch into a venv anyway, the image only provides drivers, SSH and tools |
| Disk | Container disk (erased on stop, not billed while stopped) vs volume disk (kept, $0.20/GB-month while stopped) | **Container disk 80 GB, volume disk 0 GB.** Data and checkpoints live in R2, so nothing on the box needs to survive |
| SSH | Basic SSH via `ssh.runpod.io` (no scp/sftp) or SSH over exposed TCP (full SSH) | **Exposed TCP** (official PyTorch templates support it) |

Prices checked 2026-11-13 on the RunPod pricing page.

> [!NOTE]
> Re-check the H100 SXM price on the deploy page before you click Deploy and write the price you actually pay into `budget.md`.

> [!WARNING]
> From the moment the pod is running you pay $2.69–4.39 an hour whether or not anything runs on it. Step 2 arms a hard stop before you do anything else. Do not skip it, even "just for a quick look".

## Do this

### 1. Save a template and deploy (10 min)

1. RunPod → **Pods → Deploy**. Choose Community Cloud (or Secure), open the filters and set **CUDA version** to 13.0 or newer, then select **H100 SXM**, GPU count 1.
2. Template: the official **Runpod Pytorch** template, a tag with CUDA 13.x. Click **Edit Template**:
   - Container disk **80 GB**, volume disk **0 GB**.
   - Expose TCP port **22** (on by default in the official templates).
   - Environment variables (the `{{ … }}` values are your secrets from 04.1):

   ```text
   RCLONE_CONFIG_R2_TYPE=s3
   RCLONE_CONFIG_R2_PROVIDER=Cloudflare
   RCLONE_CONFIG_R2_ACCESS_KEY_ID={{ RUNPOD_SECRET_r2_access_key_id }}
   RCLONE_CONFIG_R2_SECRET_ACCESS_KEY={{ RUNPOD_SECRET_r2_secret_access_key }}
   RCLONE_CONFIG_R2_ENDPOINT={{ RUNPOD_SECRET_r2_endpoint }}
   RCLONE_CONFIG_R2_REGION=auto
   RCLONE_CONFIG_R2_NO_CHECK_BUCKET=true
   MYGPT_REMOTE=r2:mygpt
   HF_TOKEN={{ RUNPOD_SECRET_hf_token }}
   ```

   Save it as a new private template named `mygpt-gpu`. Every later GPU lesson deploys from it.
3. Pricing: **On-Demand**. Click **Deploy** and note the time.

### 2. Arm the hard stop first (1 min)

When the pod shows **Running**, open **Connect → Web Terminal** (or SSH, next step) and run:

```bash
nohup bash -c 'sleep 3h; runpodctl pod stop $RUNPOD_POD_ID' >/tmp/hardstop.log 2>&1 &
echo "hard stop armed for $(date -u -d '+3 hours' +%H:%M) UTC"
```

This is the pattern from RunPod's own docs. Inside a pod `RUNPOD_POD_ID` and a pod-scoped API key are already set, so `runpodctl` can stop its own pod. Whatever happens next, this pod cannot run for more than 3 hours.

### 3. Connect over SSH (2 min)

**Connect → SSH over exposed TCP** shows a command like this; run it on your laptop:

```bash
ssh root@<POD_IP> -p <PORT> -i ~/.ssh/id_ed25519
```

### 4. Check driver and GPU before installing anything (2 min)

```bash
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,clocks.max.sm --format=csv
env | grep -c RCLONE_CONFIG_R2_     # expect 7
```

You want `NVIDIA H100 80GB HBM3` (the SXM part), driver **580 or newer**, 81559 MiB, a 700 W power limit. Current PyTorch wheels on PyPI are built for CUDA 13.0, which needs driver >= 580; an older driver gives you `torch.cuda.is_available() == False` later.

### 5. Install the project and run the health check (10 min)

```bash
git clone https://github.com/<you>/<your-repo>.git ~/mygpt
bash ~/mygpt/tools/setup_box.sh 2>&1 | tee /tmp/setup.log
```

[`setup_box.sh`](../../labs/module-04/setup_box.sh) installs tmux, rclone and uv, creates `~/mygpt/.venv` with the dependencies pinned in `pyproject.toml`, lists your bucket, runs the CPU test suite and then [`gpu_check.py`](../../labs/module-04/gpu_check.py). It is idempotent: rerun it after fixing anything.

### 6. Measure the network to your bucket (3 min)

```bash
cd ~/mygpt && . .venv/bin/activate
time rclone copy r2:mygpt/raw/fineweb-edu/sample-10BT/000_00000.parquet /tmp/net/ \
  --multi-thread-streams 8 --progress
rm -rf /tmp/net
```

Divide 2,153 MB by the seconds. Then estimate how long an 8×H100 node would sit idle while pulling the ~45 GB main-run shards: 45,000 MB ÷ (your MB/s) seconds. At 200 MB/s that is under 4 minutes ($2.20 of an 8× node); at 20 MB/s it is 38 minutes ($22).

### 7. GPU smoke test (5 min)

```bash
tmux new -s work
python -m mygpt.train --config configs/smoke_cpu.yaml train.out_dir=runs/smoke-gpu
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 1   # in a second tmux window: Ctrl-b c
```

### 8. Record the environment (2 min)

```bash
mkdir -p env
{ date -u; nvidia-smi; python tools/gpu_check.py --quick; uv pip freeze; } > env/h100-runpod-$(date -u +%F).txt 2>&1
git add env/ && git commit -m "Env record: 1xH100 SXM RunPod" && git push
```

Pushing from the pod needs GitHub credentials. If you have none there, `scp -P <PORT> root@<POD_IP>:mygpt/env/*.txt env/` to your laptop and commit from there. Never store a GitHub token on a shared template.

### 9. Arm the idle guard and watch it work (10 min)

The idle guard stops the pod after N minutes with the GPU idle. In real runs you use 20–30 minutes; here you use 5 so you can see it fire:

```bash
tmux new -d -s guard "bash scripts/autoshutdown.sh --idle-minutes 5 --stop-cmd 'runpodctl pod stop $RUNPOD_POD_ID' 2>&1 | tee /tmp/guard.log"
```

Now leave the GPU idle. Within about 6 minutes your SSH session drops and the RunPod console shows the pod as **Exited/Stopped**.

> [!CAUTION]
> Terminating a pod deletes its container disk for good. Everything you need from this box (the env record) must be in git or the bucket before step 10.

### 10. Terminate (1 min)

In the console: pod → **Terminate**. A stopped pod with a 0 GB volume disk costs nothing, but a terminated pod cannot surprise you later. Check **Billing → usage** shows the spend you expect.

## How you know it worked

- `gpu_check.py` ends with `HEALTHY`, and the GPU line reads like `GPU 0: NVIDIA H100 80GB HBM3, 79 GiB, 79 GiB free, bf16 matmul 6xx–7xx TFLOP/s (6x–7x% of 989 dense peak)`. Below ~55% of peak the script flags the host.
- `pytest -q` in the setup log reports all tests passed.
- The smoke run prints decreasing loss and finishes in about a minute; `nvidia-smi` in the second window showed GPU utilization above 0 while it ran (proof it ran on the GPU, not the CPU).
- You have a number for R2 → pod download speed in MB/s, typically several hundred on a datacenter host.
- `/tmp/guard.log` was the last thing written before the pod stopped by itself, and the console shows the pod stopped about 5–6 minutes after you went idle.
- `env/h100-runpod-<date>.txt` is in git.

## When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `torch.cuda.is_available()` is False, or `gpu_check` reports driver < 580 | Host driver too old for CUDA 13 wheels | Terminate; redeploy with the CUDA version filter set to 13.0+ |
| bf16 matmul far below 600 TFLOP/s, or `power.limit` well under 700 W | Throttled, power-capped or mislabeled host (often an H100 PCIe/NVL sold as SXM) | Terminate within the first minutes and take another host; note the host in `budget.md` |
| GPU shows memory already used before you started | Another process or a broken previous tenant | Terminate and redeploy; you should never share an on-demand GPU |
| `Permission denied (publickey)` | Key added after the pod was created, or pasted without `ssh-ed25519` | Fix the key in Settings, then deploy a new pod; keys are injected at creation |
| `scp` or `sftp` fails | You used the basic `ssh.runpod.io` command | Use the "SSH over exposed TCP" command with `-p <PORT>` |
| `rclone` says `AccessDenied` and `echo $RCLONE_CONFIG_R2_SECRET_ACCESS_KEY` shows `{{ RUNPOD_SECRET_… }}` | Secret name in the template does not match the secret | Fix the name in the template (they are case-sensitive) and redeploy |
| `runpodctl: unknown command "pod"` | Older runpodctl in the image | Use `runpodctl stop pod $RUNPOD_POD_ID` (older syntax) in both guards |
| The pod did not stop after 6 idle minutes | Guard not running, or something keeps the GPU busy (a forgotten `watch nvidia-smi` does not, a stuck Python process does) | `tmux attach -t guard`, read the log; kill stray processes; never leave without the hard stop from step 2 |
| `No space left on device` | Container disk too small for venv + shards | 80 GB is enough for 1 GPU; the 8-GPU template in Module 5 needs ~150 GB |

## Artifact

- Private RunPod template `mygpt-gpu` (CUDA 13.x PyTorch image, 80 GB container disk, 0 GB volume, R2/HF secrets as env vars).
- `tools/setup_box.sh`, `tools/gpu_check.py`, `tools/budget_calc.py` and `env/h100-runpod-<date>.txt` committed in your repo; your measured R2 download speed noted in `budget.md`.
- `budget.md` line:

  ```text
  | 2026-11-13 | 04.2 | RunPod 1x H100 SXM Community, ~1.2 h at $2.69/h, setup + health check | ~$3.25 | $____ |
  ```
