---
id: "04.1"
module: 4
minutes: 20
practice_minutes: 40
prerequisites: ["01.1", "01.2", "02.2"]
objectives:
  - Compare GPU providers on live price per GPU-hour, 8×H100 availability, storage, egress and billing rules.
  - Explain SXM vs PCIe/NVL and on-demand vs community vs interruptible, and when each fits this project.
  - Set up a RunPod account with SSH key, prepaid credit and secrets, ready to launch a pod in 04.2.
  - Record today's prices in budget.md and name a primary and a fallback provider for the 8-GPU runs.
volatility: implementation
sources:
  - title: "RunPod GPU pricing"
    url: https://www.runpod.io/pricing
  - title: "RunPod pod pricing and billing docs"
    url: https://docs.runpod.io/pods/pricing
  - title: "RunPod manage pods (stop vs terminate)"
    url: https://docs.runpod.io/pods/manage-pods
  - title: "RunPod secrets"
    url: https://docs.runpod.io/pods/templates/secrets
  - title: "Lambda pricing"
    url: https://lambda.com/pricing
  - title: "Lambda: creating and managing instances"
    url: https://docs.lambda.com/public-cloud/on-demand/creating-managing-instances/
  - title: "Vast.ai pricing"
    url: https://vast.ai/pricing
  - title: "Nebius prices"
    url: https://nebius.com/prices
  - title: "Modal pricing"
    url: https://modal.com/pricing
  - title: "Verda (formerly DataCrunch) GPU pricing"
    url: https://verda.com/pricing
  - title: "DataCrunch is changing its name to Verda"
    url: https://verda.com/blog/datacrunch-is-changing-its-name-to-verda
  - title: "TensorDock"
    url: https://www.tensordock.com/
  - title: "Paperspace pricing"
    url: https://www.paperspace.com/pricing
  - title: "Hyperbolic GPU marketplace"
    url: https://hyperbolic.ai/marketplace
  - title: "NVIDIA H100 specifications"
    url: https://www.nvidia.com/en-us/data-center/h100/
last_verified: "2026-11-13"
---

# 04.1 · Where to rent GPUs and what they cost

GPU prices for the same H100 differ by more than 2× between providers this week, and the cheapest offer is not always the cheapest run. This lesson compares the providers you can actually use with a credit card today, explains the few hardware and billing distinctions that change your bill, and ends with an account that is ready to launch a pod, with your secrets stored in the provider instead of in files.

## Goal

You have today's prices for H100 SXM (1× and 8×) and cheaper GPUs from at least three providers in `budget.md`, a decision on your primary provider (RunPod in this course) and a fallback for the 8-GPU runs, and a RunPod account with an SSH key, $25–50 of prepaid credit and your R2 and Hugging Face secrets stored as RunPod secrets.

## What you need

- `budget.md` and the spending rules from [01.1](../module-01/lesson-01.md); your secrets in a password manager ([01.2](../module-01/lesson-02.md)); your R2 keys from [02.2](../module-02/lesson-02.md).
- An SSH key pair on your laptop (`ssh-keygen -t ed25519 -C mygpt` if you have none).
- A payment card. About 1 hour.
- **Expected cost: $0 spent.** You prepay $25–50 of RunPod credit, which the next two lessons use.

## Choices and costs

### Three hardware facts that change the bill

- **SXM vs PCIe vs NVL.** An H100 **SXM** runs at up to 700 W, has 3.35 TB/s HBM3 and 900 GB/s NVLink to the other seven GPUs on an HGX board: 989 dense BF16 TFLOP/s. **PCIe** and **NVL** cards run at 350–400 W with lower clocks (NVL: 835 dense TFLOP/s, 94 GB) and, on most hosts, talk to each other over PCIe or a 2-card NVLink bridge only. For one-GPU work a PCIe card is fine and cheaper; for 8-GPU data-parallel training you want an **SXM** node, because gradient all-reduce over PCIe eats the scaling.
- **Price per hour is not price per token.** A B200 costs ~1.8–2× an H100 per hour and has ~2.3× the dense BF16 peak (2,250 TFLOP/s). Whether that wins for a 124M–350M model depends on how well small matmuls fill it. You find out in 04.3 by measuring dollars per billion tokens, not by reading spec sheets.
- **Dense, not sparse.** Vendors headline BF16 numbers "with sparsity" (H100 SXM: 1,979). Training uses dense math, so halve them. MFU in this course is always against the dense peak.

### Live prices (checked 2026-11-13)

Per GPU-hour, on-demand unless noted.

| Provider | H100 SXM, 1 GPU | H100 SXM, 8-GPU node | Cheaper GPUs | Storage and egress | Billing notes |
|---|---|---|---|---|---|
| **RunPod Secure Cloud** | $4.39 | $4.39 per GPU in an 8-GPU pod ($35.12/h); multi-node Instant Clusters $4.79 | H100 PCIe $2.89, A100 SXM $2.19, H200 $5.29, B200 $7.99, RTX 4090 $0.89 | Volume disk $0.10/GB-mo running, **$0.20 stopped**; network volume $0.07/GB-mo; **no ingress/egress fees** | Per second; pods stop at $0 balance; default spend limit $80/h |
| **RunPod Community Cloud** | $2.69 | 8× offers appear at times ($21.52/h) | H100 PCIe $1.99, A100 SXM $1.39, RTX 4090 $0.34 | As above | Individual hosts, less redundancy; good for 1-GPU work |
| Lambda | $4.29, **$4.89 from 2026-11-16** | $3.99, **$4.59 from 2026-11-16** ($36.72/h per node) | B200 8× $7.69, A100 80GB 8× $3.19, GH200 $2.59 (new prices) | Persistent filesystems extra | `shutdown` inside the VM does **not** stop billing; terminate in the dashboard or API |
| Vast.ai (marketplace) | from $1.87, median $2.40 | 8× offers $2.67–3.00 per GPU | H200 median $4.61, RTX 4090 median $0.49 | Disk billed while the instance exists, also stopped; **bandwidth billed per TB by each host** (often ~$1–5/TB, sometimes far more) | Per second; interruptible ("bid") 50%+ cheaper; host quality varies, filter for verified datacenter hosts |
| Hyperbolic | $3.19 (marketplace page; raised from $2.89 on 2026-08-04 per trackers) | marketplace offers | | | Supplier marketplace; prices move with supply |
| Nebius | $4.50; preemptible from $0.79 (dynamic) | HGX 8-GPU VMs at the same rate | H200 $5.40, B200 $8.50, L40S from $1.55 | Shared filesystem $0.08/GiB-mo; network egress free | Self-serve cloud; quotas for new accounts |
| Modal | ~$3.95 ($0.001097/s) | up to 8 GPUs per container | B200 ~$6.25, H200 ~$4.54, A100 80GB ~$2.50 | | Serverless functions, $30/month free credits; awkward for a multi-hour SSH workflow |
| Verda (formerly DataCrunch) | $5.189; spot $2.595 | not listed on the public page | A100 SXM 80GB $2.374 (spot $1.187), L40S $1.998 | | EU (Finland/Iceland) datacenters; spot halves the price |
| TensorDock (marketplace) | from $2.25 | not advertised on the public page | A100 SXM from $1.80, RTX 4090 from $0.35 | | Prepaid; servers are **deleted** when the balance reaches $0; host-set prices |
| Paperspace (DigitalOcean) | $5.95 on-demand ("promo") | GPU price × count | A100 80GB only listed with a 3-year commitment | | Notebook-style platform; not competitive on-demand |
| Prime Intellect | aggregator: shows offers from many clouds | varies | varies | per underlying provider | Handy for finding 8× availability; read the underlying provider's terms |
| CoreWeave, AWS | ~$6.16 (CoreWeave 8× node), ~$6.88 (AWS p5.48xlarge) per trackers | contract / large accounts | | | Not worth it at this scale |

Not usable for this project: **Google Colab** (Pro $9.99/month for 100 compute units, per third-party listings) and **Kaggle** notebooks (free weekly GPU quota, T4/P100-class). Notebook sessions with time limits, no root SSH box, no H100 SXM and no 8-GPU node; fine for reading code, not for these runs.

Vast.ai numbers are a snapshot of the public offer list on 2026-11-13; CoreWeave and AWS figures come from third-party price trackers, not the providers' pages.

> [!NOTE]
> GPU prices changed several times in 2026 and Lambda's rise takes effect three days after this check. Before you rent, open the provider's live page, compare with this table, and update the price lines in `budget.md`. The course still plans at **$4.00/GPU-h**; lesson 04.3 replaces that with your real numbers.

### On-demand, community and interruptible

| Kind | What you get | Use it for |
|---|---|---|
| On-demand, datacenter (RunPod Secure, Lambda, Nebius) | The GPU is yours until you stop it | 8-GPU baseline and main run |
| Community / marketplace (RunPod Community, Vast.ai) | Same GPUs in smaller hosting sites; cheaper; host can be flaky or slow to start | 1-GPU dev box, benchmarks, ablations |
| Interruptible / spot / preemptible (Vast.ai bids, Nebius preemptible, RunPod spot where offered) | 50–80% cheaper, but can be taken away at any moment | Only jobs that checkpoint often and resume automatically: short ablations, evals. Not your first main run |

### Recommendation for this project

- **Primary: RunPod.** One account and one balance for everything: Community Cloud H100 SXM at $2.69 for the dev and benchmark box (04.2, 04.3, ablations), Secure Cloud 8×H100 SXM for the baseline and main run. Per-second billing, no egress fees, a CLI (`runpodctl`) that can stop the pod from inside it, and secrets that become environment variables.
- **Fallback for 8×H100: Vast.ai** verified datacenter hosts (~$21–24/h per node today, but check the per-TB bandwidth price and reliability score of the exact host) or **Lambda** ($36.72/h per node from 16 November). Module 5 makes the final call with your benchmark numbers.
- Skip hyperscalers and anything needing a sales call.

> [!WARNING]
> Billing does not stop when your job ends. A finished-but-running 8×H100 node costs **$21–37 every hour**, about $500–900 a day. A stopped RunPod pod still bills its volume disk ($0.20/GB-month), a stopped Vast.ai instance still bills its disk, and a Lambda VM keeps billing until it is terminated. Lesson 04.2 sets up automatic stopping; never rely on remembering.

## Do this

### 1. RunPod account, key and credit (15 min)

1. Sign up at runpod.io and enable two-factor authentication.
2. **Settings → SSH Public Keys**: paste the whole line of `~/.ssh/id_ed25519.pub`, including the leading `ssh-ed25519`.
3. **Billing**: add $25–50 of credit. Leave automatic top-up **off**; a pod stops when the balance hits $0, which is a useful last-resort cap during these lessons.

### 2. Store your secrets in RunPod (10 min)

**Secrets → Create Secret**, one per value. RunPod substitutes them into a pod's environment at boot and never shows the value again.

| Secret name | Value |
|---|---|
| `r2_access_key_id` | R2 Access Key ID |
| `r2_secret_access_key` | R2 Secret Access Key |
| `r2_endpoint` | `https://<ACCOUNT_ID>.r2.cloudflarestorage.com` |
| `hf_token` | Hugging Face read token |

Other providers have no equivalent secret store that RunPod's template syntax would work with: on Vast.ai put the same variables in the template's environment field (visible to anyone with access to your account) or, better, `scp` your `r2.env` and `secrets.env` to the box and `set -a; . ~/r2.env; set +a`; on Lambda, Nebius or Verda always use the `scp` route. Never bake keys into a Docker image.

In 04.2 you map them to environment variables like `RCLONE_CONFIG_R2_SECRET_ACCESS_KEY={{ RUNPOD_SECRET_r2_secret_access_key }}`.

### 3. Check live prices and 8×H100 availability (15 min)

1. RunPod → **Pods → Deploy**: filter H100 SXM. Note the Community and Secure prices, and whether an **8-GPU** H100 SXM pod is offered right now in Secure Cloud (change the GPU count to 8).
2. Vast.ai: list current 8×H100 SXM offers from verified hosts, sorted by price, without an account:

   ```bash
   pip install vastai
   vastai search offers 'gpu_name=H100_SXM num_gpus=8 verified=true rentable=true' -o 'dph'
   ```

   For the cheapest three, divide `$/hr` by 8, and note `inet_down`/`inet_up` (Mbps), reliability, and the bandwidth price shown on the offer card.
3. Open Lambda's pricing page and one more provider from the table.

### 4. Write it down

Add a dated provider table to `budget.md`, for example:

```markdown
## GPU prices seen 2026-11-13
| Provider | 1x H100 SXM | 8x H100 SXM (per GPU) | 8x available now? | Notes |
|---|---|---|---|---|
| RunPod Community | 2.69 | 2.69 (rare) | no | dev/bench box |
| RunPod Secure | 4.39 | 4.39 | yes | primary for 8x |
| Vast.ai verified | 1.87-2.40 | 2.67-3.00 | yes | fallback, check $/TB |
| Lambda | 4.29 -> 4.89 | 3.99 -> 4.59 | ? | price rise 2026-11-16 |
Primary: RunPod. Fallback for 8x: Vast.ai (verified) / Lambda.
```

## How you know it worked

- RunPod Settings shows your SSH key starting with `ssh-ed25519`, Billing shows your credit, and auto top-up is off.
- RunPod Secrets lists four secrets with the names above (values hidden).
- `vastai search offers ...` returned a list of 8×H100 SXM offers with prices.
- `budget.md` has a dated price table with at least three providers and a named primary and fallback, and it is committed.

## When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| SSH key "not recognized" later in 04.2 | Pasted without the `ssh-ed25519` prefix, or pasted the private key | Paste the full single line from the `.pub` file |
| No 8×H100 SXM in Secure Cloud right now | Capacity comes and goes by the hour | Normal. Note it; check again before Module 5; your fallback exists for exactly this |
| `vastai search offers` returns nothing | Query typo (`H100 SXM` with a space) | Use `gpu_name=H100_SXM`; try `num_gpus>=8` |
| Card declined or account verification requested | Fraud checks on GPU clouds are strict | Use a card in your own name; complete the verification; don't create duplicate accounts |
| You are tempted by a $0.80/h preemptible H100 for the main run | It looks like a 4× saving | It can be reclaimed at any minute; without tested resume (Module 5) you lose hours. Use it only for ablations later |

## Artifact

- RunPod account ready: SSH key, prepaid credit, four secrets.
- Dated provider price table and primary/fallback decision in `budget.md`, committed:

  ```bash
  git add budget.md && git commit -m "GPU providers and live prices 2026-11-13"
  ```

- `budget.md` ledger line (credit is a prepayment, not spend):

  ```text
  | 2026-11-13 | 04.1 | RunPod prepaid credit $50 (not yet spent) | $0.00 | $0.00 |
  ```
