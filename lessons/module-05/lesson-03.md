---
id: "05.3"
module: 5
minutes: 20
practice_minutes: 120
prerequisites: ["05.2"]
objectives:
  - Predict the baseline's wall clock and cost from your 04.3 measurements and check the prediction in the first ten minutes.
  - Run the 10B-token GPT-2 124M baseline on an 8×H100 SXM node and recognise a healthy loss, val and HellaSwag curve.
  - Compare your result with published reference numbers, including how much of a HellaSwag difference is noise.
  - Secure the final checkpoint and log in the bucket, record the results, and terminate the node.
volatility: implementation
sources:
  - title: "Karpathy, llm.c discussion #481: GPT-2 124M in 90 minutes on 8×A100 (HellaSwag 29.9 vs OpenAI 29.4)"
    url: https://github.com/karpathy/llm.c/discussions/481
  - title: "Karpathy, build-nanogpt play.ipynb (OpenAI GPT-2 124M reference: val loss 3.2924 on FineWeb-Edu, HellaSwag 0.294463)"
    url: https://raw.githubusercontent.com/karpathy/build-nanogpt/master/play.ipynb
  - title: "Karpathy, build-nanogpt repository"
    url: https://github.com/karpathy/build-nanogpt
  - title: "Keller Jordan et al., modded-nanogpt (NanoGPT speedrun record table)"
    url: https://github.com/KellerJordan/modded-nanogpt
  - title: "Brown et al., Language Models are Few-Shot Learners (GPT-3 Small HellaSwag 33.7)"
    url: https://papers.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html
  - title: "RunPod GPU pricing"
    url: https://www.runpod.io/pricing
  - title: "RunPod: manage pods (stop, terminate)"
    url: https://docs.runpod.io/pods/manage-pods
  - title: "Lambda pricing"
    url: https://lambda.com/pricing
  - title: "Vast.ai pricing"
    url: https://vast.ai/pricing
  - title: "Nebius prices"
    url: https://nebius.com/prices
last_verified: "2026-11-13"
---

# 05.3 · Running the 10B-token baseline

This is the first time you put real money on one run: an 8×H100 node for a bit more than an hour, about $45–55. Everything has been rehearsed, so the work today is mostly watching the right numbers at the right moments: in the first ten minutes (is it as fast as you predicted?), during the run (does the curve look like a healthy GPT-2?), and at the end (is the checkpoint safely in the bucket before you terminate?). You finish with a GPT-2 124M you trained yourself and the reference numbers every later model in this course is compared against.

## Goal

You have trained GPT-2 124M on 10B FineWeb-Edu tokens on one 8×H100 SXM node. The final checkpoint `ckpt_00019073.pt` and the full `log.jsonl` are in `r2:mygpt/ckpt/baseline_124m/`, the run is on W&B, and `results.md` in your repo holds the final val loss, val bpb and HellaSwag next to the published GPT-2 numbers, the measured speed and scaling efficiency, and the real cost. The node is terminated.

## What you need

- The `mygpt-8gpu` template, `launch/baseline_8xh100.sh` and today's 8×H100 price from [05.1](lesson-01.md); your recovery procedure from [05.2](lesson-02.md).
- Your per-GPU tokens/s for `baseline_124m` from [04.3](../module-04/lesson-03.md) and the R2 download speed from [04.2](../module-04/lesson-02.md).
- At least $70 of RunPod credit (the run plus margin; a pod stops at $0 balance, which would also stop your run).
- About 2 hours of your time, ~1.5 of them with the node running. You can leave during the run, but be back for the last 15 minutes.
- **Expected cost: $45–55** on RunPod Secure 8×H100 SXM at $35.12/h (about 1.3–1.5 billed hours).

## Choices and costs

### Predict the run before you pay for it

With $r$ = your per-GPU tokens/s from 04.3 and $s$ = scaling efficiency on 8 GPUs (plan with 0.9):

$$t_{\text{train}} = \frac{10^{10}}{8 \cdot r \cdot s}\ \text{s}$$

Example with 04.3's illustrative $r = 480{,}000$: $10^{10} / (8 \times 480{,}000 \times 0.9) = 2{,}894$ s ≈ 48 minutes of pure training. Then add what the config does besides training:

| Item | Estimate on 8×H100 | Why |
|---|---|---|
| Pure training, 19,073 steps | ~48 min | Formula above (ESTIMATE, use your $r$) |
| `torch.compile` warm-up | 1–2 min | First steps compile the model |
| Val loss, 76 evals × 10.5M tokens | ~3–4 min | Forward-only, ~8% of training tokens |
| HellaSwag, 20 evals × 10,042 examples | ~5–10 min | Uncompiled forward passes, split over 8 GPUs |
| 20 checkpoint saves | ~1–2 min | All GPUs wait while rank 0 writes ~1.5 GB |
| Setup: deploy, `setup_box.sh` with tests, pull 20 GB of shards | ~15 min | At 300 MB/s the shards take ~70 s; slower networks take longer |
| Finish: verify bucket, write results, terminate | ~5–10 min | |
| **Billed node time** | **~1.3–1.5 h** | **~$45–55 at $35.12/h** |

The course plan budgeted $50 for this line. If your $r$ from 04.3 was much lower than the example, recompute before you rent.

### What the published runs did

| Run | Hardware | Data | Time | Result |
|---|---|---|---|---|
| OpenAI GPT-2 124M (2019) | not published | WebText, token count not published | | HellaSwag 29.4% (llm.c post); 0.2945 in Karpathy's evaluation; val loss 3.2924 on Karpathy's FineWeb-Edu val split |
| GPT-3 Small 125M (2020) | not published | 300B tokens | | HellaSwag 33.7% |
| llm.c reproduction (May 2024) | 8×A100 80GB SXM | 10B FineWeb tokens | ~90 min, ~$20 | HellaSwag 29.9%, train loss ~3.29 |
| build-nanogpt (2024) | one cloud GPU node | 10B FineWeb-Edu tokens (the data you use) | "~1hr and ~$10" per its README | Reference curves and the OpenAI baselines in its `play.ipynb` |
| modded-nanogpt speedrun, record #1 vs record #92 | 8×H100 | FineWeb, to val loss 3.28 | 45 min (2024-05-28) → 0.665 min (2026-08-30) | Same loss target, ~70× faster through optimizer, architecture and systems changes |

Two things follow. First, your run is a solved problem: if it does not land near these numbers, something is wrong with your data or setup, not with the recipe. Second, the speedrun shows how much better a 124M model can be trained than the 2019 recipe; Module 6 tests a few of those ideas (Muon, rotary embeddings, QK-norm, untied head) on your own data.

### Where to run it

| 8×H100 SXM node (checked 2026-11-13) | Price/h | Predicted cost (1.3–1.5 h) | What changes in the steps |
|---|---|---|---|
| **RunPod Secure Cloud** | $35.12 | $46–53 | Nothing: `mygpt-8gpu` template |
| Lambda | $31.92 ($36.72 from 2026-11-16) | $42–55 | `scp` `r2.env` and `secrets.env`, load them; Lambda terminate command in both stops (04.2); terminate in the dashboard |
| Vast.ai verified datacenter host | ~$21–24 | $28–36 | Check the host's per-TB bandwidth price (~20 GB of shards in); load the env files; `vastai destroy instance` in both stops (04.2), destroy at the end |
| Nebius | ~$36 | $47–54 | Env files over `scp`; delete the VM at the end |

> [!NOTE]
> Prices checked 2026-11-13 (RunPod Secure 8×H100 SXM $35.12/h; Lambda $31.92/h, rising to $36.72/h on 2026-11-16). The speedrun record changes often; numbers above are from its README on 2026-11-13. Re-check the price on the deploy page and write it into `budget.md` before you launch.

> [!WARNING]
> From the moment the node is running it costs **$35 an hour**, used or not. Arm the hard stop before anything else, run the idle guard, and do not leave a finished node running "until tomorrow": that is $840.

## Do this

### 1. Deploy and arm both stops (5 min)

RunPod → **Secure Cloud** → **H100 SXM**, GPU count **8**, template `mygpt-8gpu`, On-Demand → **Deploy**. Note the time. In the web terminal or SSH:

```bash
nohup bash -c 'sleep 3h; runpodctl pod stop $RUNPOD_POD_ID' >/tmp/hardstop.log 2>&1 &
```

Three hours is about twice the prediction. If your own prediction is longer than 1.5 h, use twice your number.

### 2. Set up and check the node (10 min)

```bash
git clone https://github.com/<you>/<your-repo>.git ~/mygpt && bash ~/mygpt/tools/setup_box.sh 2>&1 | tee /tmp/setup.log
cd ~/mygpt && . .venv/bin/activate
nvidia-smi topo -m | head -10        # every GPU pair: NV18 (all-to-all NVLink)
tmux new -d -s guard "bash scripts/autoshutdown.sh --idle-minutes 20 --stop-cmd 'runpodctl pod stop $RUNPOD_POD_ID' 2>&1 | tee /tmp/guard.log"
```

`gpu_check.py` (run by `setup_box.sh`) must end with `HEALTHY` and list 8 GPUs. If one GPU is slow or missing, terminate now and redeploy: $3 lost now is cheaper than a slow run.

### 3. Pull the shards and HellaSwag (2–5 min)

```bash
bash scripts/sync.sh pull-tok gpt2-10BT data/tok/gpt2-10BT
ls data/tok/gpt2-10BT | wc -l        # ~100 train shards + val + meta.json
du -sh data/tok/gpt2-10BT            # ~20 GB
python -c "from mygpt.evaluate import load_hellaswag; print(len(load_hellaswag()))"   # 10042
```

Your train split holds ~9.95B tokens, a little less than the config's 10B. The loader wraps around for the last ~0.5% of steps, which does not matter for a single-epoch baseline.

### 4. Launch (1 min)

```bash
tmux new -s train
bash launch/baseline_8xh100.sh
```

Detach with <kbd>Ctrl</kbd>+<kbd>b</kbd> <kbd>d</kbd>. Open the W&B run `baseline_124m` in project `my-gpt` on your laptop.

### 5. The first ten minutes: is it the run you predicted? (10 min)

Check these in order and stop the run (<kbd>Ctrl</kbd>+<kbd>c</kbd> in the train window) if one fails:

1. Header: `[train] params 124.5M | world 8 | micro 64x1024 | accum 1 | steps 19073 | ... | dtype bf16`.
2. Loss: ~10.9–11.0 at step 1, below ~7 by step 200, falling smoothly.
3. Speed after compile (step 30 onward): compute scaling efficiency against 04.3:

   ```bash
   python tools/run_summary.py runs/baseline_124m --gpus 8 --single-gpu-tps <your 04.3 tok/s> \
     --target-tokens 10e9 --price-per-gpu-hour 4.39
   ```

   Expect `scaling_efficiency` 0.85–1.0 and MFU close to your 1-GPU benchmark. Below ~0.8 on an SXM node means a problem (a slow GPU, another process, PCIe topology); stop and investigate rather than paying for a slow hour.
4. At step 1000 (about 3 minutes in): `rclone lsf r2:mygpt/ckpt/baseline_124m/` shows `ckpt_00001000.pt`. A long run without a visible first upload is a run you cannot recover.

### 6. During the run: what healthy looks like

| Steps (tokens) | Train / val loss | HellaSwag acc_norm | Notes |
|---|---|---|---|
| 0–715 (warmup) | 11 → ~6 | ~0.25 (chance) | LR ramps to 6e-4; grad norm high and falling |
| ~1,000 (0.5B) | ~4.0–4.3 | ~0.26 | Fast drop is over |
| ~5,000 (2.6B) | ~3.4–3.5 | ~0.27–0.28 | Slow, steady improvement |
| ~10,000 (5.2B) | ~3.2–3.3 | ~0.28–0.29 | |
| 19,073 (10B) | val ~3.05–3.15 | ~0.29–0.31 | LR at 6e-5; curve flattening |

These are ESTIMATES from the published reproductions above (FineWeb-Edu runs end lower than FineWeb runs because the text is more predictable); your val split is your own, so expect differences of a few hundredths. What matters is the shape: smooth descent, grad norm settling below ~1 after warmup, val loss tracking train loss closely (single epoch, no overfitting), HellaSwag slowly leaving 0.25.

Small loss spikes that recover within a few hundred steps are normal. A spike that does not recover, or a NaN stop, means the 05.2 rollback procedure. If the node dies, follow 05.2 drill C on a new node with `launch/baseline_8xh100.sh`.

### 7. The end of the run (10 min)

The last step logs `val_loss`, `val_bpb` and `hellaswag`, saves `ckpt_00019073.pt`, waits for the upload and copies `log.jsonl` to the bucket. Then:

```bash
ls -l runs/baseline_124m/ckpt_00019073.pt
rclone lsf r2:mygpt/ckpt/baseline_124m/ --format "sp" | sort      # same size for ckpt_00019073.pt, plus log.jsonl
python tools/run_summary.py runs/baseline_124m --gpus 8 --single-gpu-tps <your 04.3 tok/s> --markdown > results-baseline.md
```

If the final checkpoint or the log is missing in the bucket: `bash scripts/sync.sh push-ckpt baseline_124m runs/baseline_124m` and list again. Copy `results-baseline.md` and the training output to your laptop:

```bash
scp -P <PORT> root@<POD_IP>:mygpt/{results-baseline.md,runs/baseline_124m.out} .
```

### 8. Terminate the node (1 min)

> [!CAUTION]
> Terminating deletes the node's disk. Do it only after the checkpoint and `log.jsonl` are listed in the bucket with the right size and `results-baseline.md` is on your laptop.

RunPod console → the pod → **Terminate**. Then **Billing → usage**: the charge should match your prediction. Write the real amount into `budget.md`.

### 9. Write `results.md` (15 min, laptop)

Create `results.md` in your repo with the `run_summary.py` table and this comparison, filled in with your numbers:

```markdown
### Baseline: GPT-2 124M, 10B FineWeb-Edu tokens (GPT-2 tokenizer), 8xH100 SXM, <date>

| Model | Val loss (own FineWeb-Edu val) | Val bpb | HellaSwag acc_norm (in-training eval) |
|---|---|---|---|
| My baseline, step 19,073 | 3.0x | 0.9x | 0.29x |
| OpenAI GPT-2 124M | (3.29 on Karpathy's split; measured on mine in Module 8) | | 0.2945 (Karpathy), 29.4% (llm.c post) |
| GPT-3 Small 125M (300B tokens) | | | 0.337 |

Checkpoint: r2:mygpt/ckpt/baseline_124m/ckpt_00019073.pt  W&B: <run link>
Speed: <node tok/s>, MFU <x>, scaling efficiency <s> vs 1 GPU. Wall clock <h>, cost $<real>.
```

Val bpb converts the loss to bits per byte of text, $\text{bpb} = \frac{\text{loss}}{\ln 2} \cdot \frac{\text{tokens}}{\text{bytes}}$: with a loss of 3.07 nats and ~4.57 bytes per GPT-2 token that is $3.07 / 0.693 / 4.57 \approx 0.97$. It is the number your own-tokenizer models in Module 6 and 7 are compared on.

## How you know it worked

- The run reached step 19,073, its last line reports `val_loss` around 3.0–3.2 and `hellaswag` around 0.29–0.31.
- HellaSwag has 10,042 examples, so its standard error is about $\sqrt{0.3 \times 0.7 / 10{,}042} \approx 0.0046$: anything within about ±0.01 of 0.2945 is "matches GPT-2". Above that, you beat it; well below 0.28, something is off.
- `scaling_efficiency` was 0.85 or better and the billed time is within ~20% of your prediction.
- `r2:mygpt/ckpt/baseline_124m/` contains `ckpt_00019073.pt` (~1.5 GB) and `log.jsonl`; the node is terminated and billing shows it.

## When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| No 8×H100 SXM pod available in Secure Cloud | Capacity varies through the day | Try again in an hour, or use the fallback from 05.1 (Lambda; a verified Vast.ai host). Export the R2 and W&B variables from your env file there, and pass a provider stop command to `autoshutdown.sh` |
| Scaling efficiency below 0.8 | One GPU throttled or busy, or not an all-NVLink node | `nvidia-smi` (all 8 at ~100% and ~700 W?), `nvidia-smi topo -m` (NV18?). Stop, terminate, redeploy |
| Run hangs: no new step lines, GPUs at 100% or 0% | A GPU failed (look for `Xid` in `dmesg`) or NCCL is stuck | Kill (`pkill -KILL -f mygpt.train`), terminate the node, resume on a new one (05.2 drill C) |
| Loss stalls around 7 or higher after 1,000 steps | Shards do not match the model (wrong tokenizer or `vocab_size`), or corrupted shards | Check `data/tok/gpt2-10BT/meta.json` (tokenizer gpt2, vocab 50257); rerun the Module 3 shard check |
| HellaSwag still 0.25 at step 5,000 while loss is normal | Evaluation uses the wrong tokenizer | The shards' `meta.json` must name `gpt2`; check the eval line in the output |
| Val loss clearly above train loss and the gap growing | Train data repeating (only some shards present locally), or a val shard from a different tokenization run | count the `train_*.bin` files (all ~100 shards present?); val and train must come from the same Module 3 run |
| `No space left on device` | Container disk smaller than 150 GB | Template disk size; free space by removing `data/raw` if you pulled it by mistake |
| W&B shows nothing but training runs | Missing `WANDB_API_KEY` or network trouble to wandb.ai | Training continues and `log.jsonl` has everything; fix the key for the next run |
| Pod stopped by the hard stop before the end | Run slower than predicted, or late start | Rent again, pull with `sync.sh pull-ckpt`, relaunch: it resumes at the last checkpoint |

## Artifact

- In the bucket: `ckpt/baseline_124m/ckpt_00019073.pt` and `ckpt/baseline_124m/log.jsonl` (plus the intermediate checkpoints).
- In your repo: `results.md` with the summary table, the reference comparison and the W&B link; the console log copied in step 7, moved to `reports/baseline_124m.out`.
- `budget.md`: the measured speed and scaling efficiency (they replace 04.3's 0.9 for the main-run estimate), and the ledger line:

  ```text
  | 2026-11-13 | 05.3 | RunPod Secure 8x H100 SXM, ~1.4 h at $35.12/h, GPT-2 124M baseline 10B tokens | ~$49 | $____ |
  ```
