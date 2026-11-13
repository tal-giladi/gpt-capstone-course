---
id: "02.2"
module: 2
minutes: 15
practice_minutes: 35
prerequisites: ["01.2", "02.1"]
objectives:
  - Compare R2, B2, S3, Hugging Face storage and GPU-provider volumes on monthly cost and egress for this project's ~180 GB.
  - Create a private R2 bucket and a bucket-scoped API token.
  - Configure rclone through environment variables and prove a round trip with a checksum check.
  - Fix the bucket layout that every later module reads from and writes to.
volatility: implementation
sources:
  - title: "Cloudflare R2 pricing"
    url: https://developers.cloudflare.com/r2/pricing/
  - title: "Backblaze B2 pricing"
    url: https://www.backblaze.com/cloud-storage/pricing
  - title: "Amazon S3 pricing"
    url: https://aws.amazon.com/s3/pricing/
  - title: "Hugging Face Hub storage limits"
    url: https://huggingface.co/docs/hub/storage-limits
  - title: "Hugging Face Storage Buckets"
    url: https://huggingface.co/docs/hub/storage-buckets
  - title: "Hugging Face storage pricing"
    url: https://huggingface.co/storage
  - title: "RunPod pod pricing (storage)"
    url: https://docs.runpod.io/pods/pricing
  - title: "rclone S3 backend (Cloudflare R2 section)"
    url: https://rclone.org/s3/
last_verified: "2026-11-13"
---

# 02.2 · Object storage for datasets and checkpoints

GPU boxes in this course are rented by the hour and thrown away, often at a different provider each time. Your data and checkpoints therefore live in one place that every box can reach cheaply: an S3-compatible bucket. This lesson picks the bucket, creates it with a key that can touch only that bucket, and wires up rclone so the same two commands move data on your laptop, a CPU box and an 8×H100 node.

## Goal

You have a private Cloudflare R2 bucket named `mygpt`, an API token scoped to that bucket only, an `r2.env` file outside your repo that defines an rclone remote called `r2`, and a verified round trip (upload, checksum compare, download, delete). You also have the bucket layout written down, which later modules rely on.

## What you need

- A Cloudflare account (free) and a payment card. R2 needs a payment method on file even when you stay inside the free tier.
- rclone 1.74 or newer on your laptop (installed below).
- Your secrets handling from [01.2](../module-01/lesson-02.md) (password manager, nothing secret in git).
- About 50 minutes.
- **Expected cost: $0 today; about $2.50/month once the bucket holds ~180 GB.**

## Choices and costs

### What you will store and move

- **Stored:** ~93 GB raw parquet (02.3), ~65 GB tokenized shards (Module 3), ~20 GB of checkpoints you keep. Call it **180 GB for ~3 months**.
- **Moved out (egress):** tokenized shards pulled to every GPU box. With a dev box, a baseline node, ablation boxes, a main-run node and a couple of restarts, expect ~6 pulls of up to 65 GB, so **~400 GB of egress**, possibly from several providers.

### Prices (checked 2026-11-13)

| Option | Storage | Egress to the internet | Monthly cost for this project | Notes |
|---|---|---|---|---|
| **Cloudflare R2** (Standard) | $0.015/GB-month, first 10 GB free | **Free** | ~$2.55 | S3 API, works with rclone/boto3. Class A ops $4.50/M, Class B $0.36/M (irrelevant at our file counts) |
| Backblaze B2 | $6.95/TB-month (~$0.007/GB), first 10 GB free | Free up to 3× your stored data per month, then $0.01/GB | ~$1.25 | S3 API. Cheapest storage; the 3× cap (~540 GB/month here) is enough but not unlimited |
| AWS S3 Standard (us-east-1) | ~$0.023/GB-month | ~$0.09/GB after the first 100 GB/month | ~$4 storage + ~$26 egress | Only sensible if your GPUs are also in AWS, same region |
| Hugging Face Storage Bucket / private dataset repo | Free accounts: 100 GB private total. PRO ($9/month): 1 TB private included, then $18/TB-month | Included up to 8× stored | $9 (PRO) | Storage Buckets are S3-like, non-versioned and have an S3-compatible API; can copy FineWeb-Edu server-side without downloading. Not used here because our scripts are tested with rclone on R2 |
| RunPod network volume | $0.07/GB-month (<1 TB) | Free inside RunPod | ~$12.60 | Lives in one RunPod data centre; only pods there can mount it. A cache next to the GPUs, not your source of truth |
| Nebius object storage | $0.0147/GiB-month | $0.015/GiB (Standard tier) | ~$2.70 + ~$6 | Only interesting if you train on Nebius |

> [!NOTE]
> Prices drift. Before you create the bucket, open the R2 and B2 pricing pages linked in the sources, check the numbers, and update the storage line in your `budget.md`.

### Recommendation: Cloudflare R2

Use **R2** for this project. Egress is free no matter where the GPUs are, which matters because you will move between RunPod, Lambda or Vast.ai depending on price and availability in Module 4. The storage premium over B2 is about $1.30 a month, which is less than ten minutes of a single idle H100. B2 is a fine fallback (the same rclone commands work with a `b2` or S3-type remote) if you cannot get R2.

> [!CAUTION]
> An R2 API secret can read, overwrite and delete everything it is scoped to. Scope it to the one bucket, store it in your password manager and in a `chmod 600` file outside any git repo, and never paste it into a notebook, a Docker image, a W&B config or a chat. If it leaks, delete the token in the Cloudflare dashboard immediately and create a new one.

## Do this

### 1. Create the bucket

1. In the Cloudflare dashboard open **R2 Object Storage** and, if asked, add a payment method and activate R2 (the free tier is included).
2. **Create bucket** → name `mygpt` → location **Automatic** → default storage class **Standard**. Leave public access **off** (no `r2.dev` URL, no custom domain).
3. Note your **Account ID** (shown on the R2 overview page). Your S3 endpoint is `https://<ACCOUNT_ID>.r2.cloudflarestorage.com`.

### 2. Create a token that can touch only this bucket

1. R2 overview → **Manage API tokens** → **Create Account API token** (or User API token).
2. Permissions: **Object Read & Write**. Bucket scope: **Apply to specific buckets only** → `mygpt`. TTL: until your project ends (for example 6 months).
3. Create it, then copy the **Access Key ID** and **Secret Access Key** straight into your password manager. Cloudflare shows the secret once.

### 3. Install rclone

```bash
# Linux (and the rented boxes later):
sudo -v ; curl https://rclone.org/install.sh | sudo bash
# macOS:
brew install rclone
# Windows (PowerShell):
winget install Rclone.Rclone

rclone version   # expect v1.74 or newer
```

### 4. Define the `r2` remote through environment variables

No `rclone config` wizard: the remote lives in one env file you can copy to any box. Start from [`labs/module-02/r2.env.example`](../../labs/module-02/r2.env.example) (you copied it to `tools/` in 02.1).

```bash
mkdir -p ~/.config/mygpt
cp tools/r2.env.example ~/.config/mygpt/r2.env
chmod 600 ~/.config/mygpt/r2.env
nano ~/.config/mygpt/r2.env        # paste key id, secret, account id
set -a; . ~/.config/mygpt/r2.env; set +a
```

The file sets `RCLONE_CONFIG_R2_TYPE=s3`, `..._PROVIDER=Cloudflare`, the keys, `..._ENDPOINT`, `..._REGION=auto` and `..._NO_CHECK_BUCKET=true`. The last one matters: a bucket-scoped token is not allowed to list or create buckets, and without it rclone tries and fails with `AccessDenied`.

> [!TIP]
> On Windows use WSL or Git Bash for these commands, or set the same variables with `$env:RCLONE_CONFIG_R2_TYPE="s3"` and so on in PowerShell.

### 5. Prove the round trip

```bash
echo "hello from $(hostname) at $(date -u +%FT%TZ)" > /tmp/hello.txt
rclone copy /tmp/hello.txt r2:mygpt/tmp/
rclone lsf r2:mygpt/tmp/
rclone cat r2:mygpt/tmp/hello.txt
rclone check /tmp/ r2:mygpt/tmp/ --include hello.txt
rclone deletefile r2:mygpt/tmp/hello.txt
```

### 6. Measure your upload speed from home (decides 02.3)

```bash
head -c 1G </dev/urandom > /tmp/1g.bin
time rclone copy /tmp/1g.bin r2:mygpt/tmp/ --s3-upload-concurrency 8 --s3-chunk-size 64M --progress
rclone deletefile r2:mygpt/tmp/1g.bin; rm /tmp/1g.bin
```

Divide 1,000 MB by the seconds it took. Below ~50 MB/s (400 Mbit/s), uploading 93 GB from home takes more than half an hour of saturated upload plus the same download; in 02.3 you will do the transfer from a cheap cloud box instead.

### 7. Write down the bucket layout

Every later module uses these prefixes. Add this block to your project `README.md`:

```text
r2:mygpt/
  DATA_CARD.md                    copy of the data card (02.3)
  manifests/                      sha256 manifests of raw data (02.3)
  raw/fineweb-edu/sample-10BT/    14 parquet files, baseline (02.3)
  raw/fineweb-edu/sample-100BT/   first 30 parquet files, main run (02.3)
  tokenizer/                      your BPE tokenizer (03.1)
  tok/                            uint16 token shards, one folder per tokenizer+subset (03.2)
  ckpt/<run_name>/                training checkpoints (Module 5 onwards)
  hf/                             exported Hugging Face models (Module 8 onwards)
  tmp/                            tests and scratch, safe to delete
```

> [!WARNING]
> Use `rclone copy`, not `rclone sync`, unless you mean it. `sync` makes the destination identical to the source, which **deletes** files in the bucket that are missing locally. A `sync` from an empty folder to `r2:mygpt/` wipes your data and checkpoints, and R2 has no recycle bin. When you do need `sync`, run it with `--dry-run` first.

## How you know it worked

- `rclone lsf r2:mygpt/tmp/` prints `hello.txt`, and `rclone cat` prints your line back.
- `rclone check` ends with `0 differences found` and `1 matching files`.
- The 1 GB upload finishes and reports a rate; you wrote that number down.
- In the Cloudflare dashboard the bucket `mygpt` shows public access disabled, and the token page shows one token scoped to `mygpt` with Object Read & Write.
- `git status` in your project repo does **not** show any file containing the secret.

## When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `AccessDenied` on `rclone lsd r2:` or on the first `copy` | Bucket-scoped tokens cannot list or create buckets | Use `rclone lsf r2:mygpt` instead of `lsd r2:`; make sure `RCLONE_CONFIG_R2_NO_CHECK_BUCKET=true` is set |
| `SignatureDoesNotMatch` | Wrong secret (often a trailing space or the key ID pasted twice), or the box clock is far off | Re-paste the secret; check `date -u` |
| `NoSuchBucket` or objects appear under `mygpt/mygpt/` | You copied the endpoint URL that ends in `/mygpt` | The endpoint is only `https://<ACCOUNT_ID>.r2.cloudflarestorage.com`, without the bucket |
| `didn't find section in config file` | Env vars not exported in this shell | Run `set -a; . ~/.config/mygpt/r2.env; set +a` again; check with `env | grep RCLONE_CONFIG_R2_TYPE` |
| Upload is very slow and uses one connection | Default multipart settings | Pass `--s3-upload-concurrency 8 --s3-chunk-size 64M` and `--transfers 8` for many files |
| You committed `r2.env` by accident | `tools/` was added with `git add .` | Delete the token in Cloudflare **first**, create a new one, then remove the file and add `*.env` to `.gitignore`. A secret pushed once is compromised even after you rewrite history |

## Artifact

- Bucket `r2:mygpt` (private) and a bucket-scoped token in your password manager.
- `~/.config/mygpt/r2.env` on your laptop (never in git), and the bucket layout block in your project `README.md`. Commit the README:

  ```bash
  git add README.md && git commit -m "Storage: R2 bucket layout"
  ```

- `budget.md` line (estimate; R2 bills monthly in arrears):

  ```text
  | 2026-11-13 | 02.2 | Cloudflare R2, ~180 GB x 3 months at $0.015/GB-mo, egress free | ~$7.65 | $0.00 so far |
  ```
