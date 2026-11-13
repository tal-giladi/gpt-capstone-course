---
id: "02.3"
module: 2
minutes: 15
practice_minutes: 90
prerequisites: ["01.3", "02.1", "02.2"]
objectives:
  - Choose between downloading at home and on a rented CPU box from measured bandwidth and price.
  - Download FineWeb-Edu sample/10BT and 30 files of sample/100BT with a token and Xet high-performance mode.
  - Verify every file against the Hub's sha256 at a pinned commit and upload it to R2 with a checked copy.
  - Finish the data card and delete the rented box so it stops billing.
volatility: implementation
sources:
  - title: "FineWeb-Edu dataset card"
    url: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
  - title: "Hugging Face Hub rate limits"
    url: https://huggingface.co/docs/hub/rate-limits
  - title: "huggingface_hub v1.0 announcement (hf_xet replaces hf_transfer)"
    url: https://huggingface.co/blog/huggingface-hub-v1
  - title: "huggingface_hub migration guide (HF_HUB_ENABLE_HF_TRANSFER ignored)"
    url: https://huggingface.co/docs/huggingface_hub/main/en/concepts/migration
  - title: "Hetzner Cloud regular-performance (CPX) prices"
    url: https://www.hetzner.com/cloud/regular-performance/
  - title: "RunPod pod pricing and billing"
    url: https://docs.runpod.io/pods/pricing
  - title: "DigitalOcean Droplet pricing (CPU-Optimized)"
    url: https://www.digitalocean.com/pricing/droplets
  - title: "DigitalOcean Droplet billing (powered-off Droplets are billed)"
    url: https://docs.digitalocean.com/products/droplets/details/pricing/
  - title: "Hugging Face Storage Buckets (server-side copy from dataset repos)"
    url: https://huggingface.co/docs/hub/storage-buckets
  - title: "rclone S3 backend (multipart upload tuning)"
    url: https://rclone.org/s3/
last_verified: "2026-11-13"
---

# 02.3 · Downloading FineWeb-Edu into your bucket

This is the first lesson where a rented machine bills you by the hour, so the habits start here: decide where the work runs from measured numbers, run long jobs inside `tmux`, verify before you trust, and delete the machine the moment you are done. You move about 93 GB from Hugging Face into your R2 bucket and leave behind a manifest and a data card that pin exactly what you trained on.

## Goal

Your bucket holds `raw/fineweb-edu/sample-10BT/` (14 files, 28.52 GB) and `raw/fineweb-edu/sample-100BT/` (the first 30 files, 64.56 GB), every file checked against the Hub's sha256 at one pinned dataset commit. `manifests/` holds the two JSON manifests, `DATA_CARD.md` is complete in your repo and in the bucket, and the CPU box you used no longer exists.

## What you need

- R2 bucket and `r2.env` from [02.2](lesson-02.md); `DATA_CARD.md` draft from [02.1](lesson-01.md).
- Your project repo on GitHub from [01.3](../module-01/lesson-03.md), with the Module 2 tools in `tools/`.
- Your HF read token (`HF_TOKEN`) from [01.2](../module-01/lesson-02.md).
- A Hetzner Cloud account (recommended below) or a RunPod account with credit, and an SSH key pair on your laptop (`ssh-keygen -t ed25519` if you have none).
- About 2 hours, mostly waiting.
- **Expected cost: under $1** (about 1.5 h of a 16-vCPU box), $0 if you do it at home.

## Choices and costs

The job is bandwidth and disk, not CPU: download ~93 GB, hash it, upload ~93 GB. You need ~100 GB of free disk and a fast, uncapped link.

| Where | Price (checked 2026-11-13) | Disk / traffic | Trade-off |
|---|---|---|---|
| **Hetzner Cloud CPX62, Helsinki** (16 vCPU, 32 GB) | €0.1779/h excl. VAT (~$0.21/h), billed hourly, capped monthly | 640 GB NVMe, 20 TB traffic included | **Recommended.** Cheap, big local disk, fast data-centre links. The same box type tokenizes in Module 3. New accounts can need ID verification; some types show "unavailable" in some locations |
| Hetzner Cloud CPX42 / CPX52 | €0.0953 / €0.1370 per hour | 320 / 480 GB, 20 TB | Also enough for this lesson; fewer cores for Module 3 |
| RunPod CPU pod | Shown in the console when you deploy (per-vCPU pricing, billed per second) | Container disk $0.10/GB-month while running; no ingress/egress fees | Fine if you already have RunPod credit; give it >= 120 GB container disk |
| DigitalOcean CPU-Optimized droplet, 16 vCPU / 32 GiB | $0.50/h | 200 GiB SSD, 7,000 GiB outbound transfer | Simple signup; disk just big enough for this lesson (Module 3 needs more). A powered-off droplet still bills: destroy it |
| At home | $0 | Your disk; your ISP's upload | Only if 02.2 measured >= 50 MB/s upload and you have 100 GB free. Home upload is usually the bottleneck |
| Hugging Face Storage Bucket server-side copy | Storage $18/TB-month above your plan's private quota | No download at all | Skipped: our pipeline reads from R2 with rclone (see 02.2) |

> [!NOTE]
> Hetzner raised CPX/CCX prices twice in 2026 (April and June) and the US locations have much smaller traffic allowances. Check the live price for your location on the Hetzner page before you create the server, and put the number you see in `budget.md`.

**Recommendation:** Hetzner CPX62 in Helsinki (`hel1`), Ubuntu 24.04. Download, verify and upload in one session, then delete the server.

How the files get from Hugging Face to the box:

| Method | Cost | When |
|---|---|---|
| **`scripts/download_fineweb.py`** (huggingface_hub with the Xet client) | $0 | Default: resumes, skips finished files, writes the folder layout the manifest step expects |
| `hf download HuggingFaceFW/fineweb-edu --repo-type dataset --include "sample/10BT/*" --local-dir data/raw` | $0 | Same files via the CLI; they land in `data/raw/sample/10BT/`, so point `--dir` of the manifest there |
| `hf buckets cp hf://datasets/HuggingFaceFW/fineweb-edu/sample/10BT hf://buckets/<you>/<bucket>/mygpt/raw/fineweb-edu/sample-10BT` | $0 transfer; storage on your HF plan | Only if your bucket is an HF Storage Bucket (02.2): server-side copy, no upload from the box. You still download once to build the sha256 manifest |

> [!WARNING]
> A Hetzner server bills until it is **deleted**, also when it is powered off. A forgotten CPX62 costs about $150 a month. The last step of this lesson deletes it; set a phone timer for 3 hours as a backstop.

## Do this

### 1. Create the server (5 min)

In the Hetzner Cloud console: **Security → SSH keys → Add** your `~/.ssh/id_ed25519.pub`. Then **Add server**: location Helsinki, image Ubuntu 24.04, type Shared vCPU → AMD → **CPX62**, your SSH key, name `mygpt-cpu`. No volumes, no backups. Or with the `hcloud` CLI:

```bash
hcloud server create --name mygpt-cpu --type cpx62 --image ubuntu-24.04 \
  --location hel1 --ssh-key my-laptop
hcloud server ip mygpt-cpu
```

Write the start time in your notes.

### 2. Prepare the box (10 min)

From your laptop, copy the R2 env file and log in:

```bash
scp ~/.config/mygpt/r2.env root@<IP>:/root/r2.env
ssh root@<IP>
```

On the box:

```bash
apt-get update && apt-get install -y git tmux
curl https://rclone.org/install.sh | bash
curl -LsSf https://astral.sh/uv/install.sh | sh && . ~/.local/bin/env
tmux new -s data                         # everything below runs inside tmux
git clone https://github.com/<you>/<your-repo>.git mygpt && cd mygpt
uv venv --python 3.12 && . .venv/bin/activate
uv pip install -e . --torch-backend cpu  # CPU-only torch: no 3 GB of CUDA wheels on a CPU box
chmod 600 /root/r2.env && set -a && . /root/r2.env && set +a
read -s -p "HF token: " HF_TOKEN && export HF_TOKEN && echo
export HF_XET_HIGH_PERFORMANCE=1
rclone lsf r2:mygpt/ && hf auth whoami
```

`HF_XET_HIGH_PERFORMANCE=1` lets the Xet download client use more connections and disk throughput. Older guides tell you to `pip install hf_transfer` and set `HF_HUB_ENABLE_HF_TRANSFER=1`; since `huggingface_hub` 1.0 that flag is ignored.

> [!TIP]
> If your SSH connection drops, log back in and run `tmux attach -t data`. The downloads keep running inside tmux.

### 3. Download (10–30 min)

```bash
python scripts/download_fineweb.py --subset sample-10BT  --out data/raw/sample-10BT
python scripts/download_fineweb.py --subset sample-100BT --max-files 30 --out data/raw/sample-100BT
du -sh data/raw/*
```

### 4. Verify against the Hub at a pinned commit (5 min)

Record the dataset commit once and check both folders against it. The script compares size and sha256 of every file with the Hub's records and writes a manifest.

```bash
REV=$(python -c "from huggingface_hub import HfApi; print(HfApi().dataset_info('HuggingFaceFW/fineweb-edu').sha)")
echo $REV
python tools/fineweb_manifest.py --dir data/raw/sample-10BT --prefix sample/10BT \
  --revision $REV --workers 16 --out manifests/sample-10BT.json
python tools/fineweb_manifest.py --dir data/raw/sample-100BT --prefix sample/100BT \
  --max-files 30 --revision $REV --workers 16 --out manifests/sample-100BT-30.json
```

On 2026-11-13 the commit was `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` (last data change July 2025). If yours differs, the dataset was updated; that is fine, your manifest records which one you have.

### 5. Upload to R2 and check (10–20 min)

```bash
RCL="--transfers 8 --s3-upload-concurrency 8 --s3-chunk-size 64M --stats 30s --stats-one-line"
rclone copy data/raw/sample-10BT  r2:mygpt/raw/fineweb-edu/sample-10BT  $RCL
rclone copy data/raw/sample-100BT r2:mygpt/raw/fineweb-edu/sample-100BT $RCL
rclone check data/raw/sample-10BT  r2:mygpt/raw/fineweb-edu/sample-10BT  --one-way
rclone check data/raw/sample-100BT r2:mygpt/raw/fineweb-edu/sample-100BT --one-way
rclone copy manifests r2:mygpt/manifests
rclone size r2:mygpt/raw/fineweb-edu/
```

Memory used by multipart uploads is roughly transfers × concurrency × chunk size = 8 × 8 × 64 MB = 4 GB, well inside 32 GB.

### 6. Finish the data card

On your laptop, fill in the remaining fields of `DATA_CARD.md` from the two manifests (`revision`, `n_files`, `total_bytes`, `est_gpt2_tokens`), the download date, machine and location. Then:

```bash
scp root@<IP>:mygpt/manifests/*.json manifests/
git add DATA_CARD.md manifests/ && git commit -m "Data card: FineWeb-Edu 10BT + 100BT[0:30], pinned and verified"
git push
rclone copy DATA_CARD.md r2:mygpt/
```

### 7. Delete the box

> [!CAUTION]
> Deleting the server destroys its disk. Do it only after `rclone check` reported 0 differences for both folders and the manifests are in the bucket and in git.

```bash
hcloud server delete mygpt-cpu      # or console: server → Delete
hcloud server list                  # must not list mygpt-cpu
```

On another provider: RunPod CPU pod → console → **Terminate**; DigitalOcean → `doctl compute droplet delete mygpt-cpu` (or Destroy in the console). Powering off does not stop billing on any of them.

## How you know it worked

- Both manifest runs end with `OK`, for example:

  ```text
  HuggingFaceFW/fineweb-edu@87f09149ef47 sample/10BT: 14 files, 28.52 GB, ~10.00B GPT-2 tokens (estimate)
  OK
  HuggingFaceFW/fineweb-edu@87f09149ef47 sample/100BT: 30 files, 64.56 GB, ~22.54B GPT-2 tokens (estimate)
  OK
  ```

- Both `rclone check` runs print `0 differences found`.
- `rclone size r2:mygpt/raw/fineweb-edu/` reports `Total objects: 44` and `Total size: 86.69 GiB (93080627715 Byte)`; rclone prints GiB, which is why the number is smaller than 93 GB.
- `hcloud server list` (or the console) shows no server, and the Hetzner usage page shows roughly €0.18 per hour you had it.

## When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `429 Too Many Requests` during download | No token, so you share the anonymous per-IP limit | `export HF_TOKEN=...` and rerun; finished files are skipped |
| Download crawls at a few MB/s | Xet high-performance mode off, or a congested route | Set `HF_XET_HIGH_PERFORMANCE=1`; if still slow after 10 minutes, try a box in another location (Falkenstein, Nuremberg) |
| `No space left on device` | Disk smaller than 100 GB (wrong server type, or a RunPod pod with the default 20 GB container disk) | Use CPX42 or larger; on RunPod set container disk >= 120 GB at deploy time |
| Manifest says `size ... !=` or `sha256 mismatch` for a file | Interrupted download left a partial file | Delete that file and rerun the download command; rerun the manifest |
| Manifest says `missing` for files you did download | `download_fineweb.py` wrote into a subfolder (for example `sample/10BT/`) | Point `--dir` at the folder that actually contains the `.parquet` files |
| `rclone check` reports differences | Upload interrupted | Rerun the same `rclone copy`; it only sends what differs; check again |
| `uv pip install` pulls `nvidia-*` packages | `--torch-backend cpu` missing | Reinstall with the flag; CUDA wheels are useless on this box and waste minutes |
| You powered the server off instead of deleting it | Habit from laptops | It is still billing: delete it |

## Artifact

- In the bucket: `raw/fineweb-edu/sample-10BT/` (14 files), `raw/fineweb-edu/sample-100BT/` (30 files), `manifests/*.json`, `DATA_CARD.md`.
- In git: complete `DATA_CARD.md` and `manifests/`.
- `budget.md` line (use your real hours and the price you saw):

  ```text
  | 2026-11-13 | 02.3 | Hetzner CPX62 hel1, 1.5 h at EUR 0.1779/h, FineWeb-Edu download to R2 | ~$0.35 | $____ |
  ```
