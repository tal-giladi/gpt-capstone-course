---
id: "07.3"
module: 7
minutes: 20
practice_minutes: 60
prerequisites: ["07.2"]
objectives:
  - Verify that the final checkpoint and log are complete in the bucket before anything is deleted.
  - Export the final checkpoint to a standard Hugging Face class with a passing logit check and a working generation, and store it under hf/my-gpt-350m-base/ (and optionally in a private Hub repo).
  - Shut down every billable resource of the run and prune the bucket without losing anything Modules 8–11 need.
  - Reconcile budget.md with the provider's real charges and state the money left for the rest of the course.
volatility: implementation
sources:
  - title: "Hugging Face Transformers: Qwen3 model documentation"
    url: https://huggingface.co/docs/transformers/model_doc/qwen3
  - title: "Hugging Face Hub CLI guide (hf auth login, hf repos create --private, hf upload)"
    url: https://huggingface.co/docs/huggingface_hub/guides/cli
  - title: "Hugging Face Hub: storage limits (free accounts 100 GB private storage)"
    url: https://huggingface.co/docs/hub/storage-limits
  - title: "RunPod docs: Pod pricing (stopped volume disks and network volumes keep billing)"
    url: https://docs.runpod.io/pods/pricing
  - title: "RunPod docs: manage pods (stop, terminate)"
    url: https://docs.runpod.io/pods/manage-pods
  - title: "rclone filtering (--filter, --dry-run)"
    url: https://rclone.org/filtering/
  - title: "Cloudflare R2 pricing"
    url: https://developers.cloudflare.com/r2/pricing/
last_verified: "2026-11-13"
---

# 07.3 · Finishing the run and shutting everything down

The last half hour of the main run decides whether you have a model or only a bill. The final checkpoint has to be safely in the bucket, converted into a format the rest of the world can load, and checked to behave the same as the model you trained, before the node with the only fast copy is gone. Then the money side: every pod, volume and stray checkpoint that keeps billing gets shut down or deleted on purpose, and `budget.md` gets the real number from the provider's usage page rather than the estimate. You end with your base model in `hf/my-gpt-350m-base/`, which is where Modules 8 to 11 pick it up.

## Goal

`r2:mygpt/hf/my-gpt-350m-base/` holds your model as a standard Hugging Face model (`Qwen3ForCausalLM`, `LlamaForCausalLM` or `GPT2LMHeadModel`, depending on your recipe), with tokenizer, chat template and `mygpt_export.json` showing a logit difference below 1e-3 against the native checkpoint. The final native checkpoint, the pre-decay checkpoint and the full `log.jsonl` are in `r2:mygpt/ckpt/main_350m/`; the 8×H100 node is terminated, no other pod or volume is billing, and `budget.md` has the real cost of the run and the money left.

## What you need

- The finished main run on its node ([07.2](lesson-02.md)), and `reports/main-run-log.md`.
- `HF_TOKEN` with write access in your password manager ([01.2](../module-01/lesson-02.md)), only if you also want a private Hub copy.
- About 30 minutes on the node (still at the full hourly price) and 30 minutes on your laptop.
- **Expected cost: ~$15–20** of node time for the checks, export and uploads (~0.5 h at $35.12/h, already inside 07.1's prediction); bucket storage afterwards ~$1.5–2 per month.

## Choices and costs

### What to keep, and where

| Item | Size | Keep where | Monthly cost (R2 $0.015/GB-mo, checked 2026-11-13) | Why |
|---|---|---|---|---|
| Final native checkpoint `ckpt_00038146.pt` | ~2.9 GB | Bucket | $0.04 | Re-export, more training, exact reproduction |
| Pre-decay checkpoint `ckpt_00030000.pt` | ~2.9 GB | Bucket | $0.04 | WSD lets you continue from the stable phase later (more tokens, a new decay) without starting over |
| `log.jsonl`, run log, configs | < 10 MB | Bucket + git | ~$0 | The evidence for every number in Module 8 and 11 |
| HF export `hf/my-gpt-350m-base/` (fp32 safetensors) | ~1.35 GB | Bucket (+ optional private Hub repo) | $0.02 | What Modules 8–11 load |
| The other ~36 checkpoints | ~105 GB | Delete | would be ~$1.60 | Not needed once the run is finished and checked |
| Rolled-back checkpoints (if any) | 2.9 GB each | Delete | | Diverged states |
| Baseline `ckpt/baseline_124m/ckpt_00019073.pt` + log | ~1.5 GB | Keep | $0.02 | Module 8 compares against it |

### Export target

The export class is fixed by your recipe (06.4's export rule): RoPE + RMSNorm + SwiGLU + no bias + QK-norm → `Qwen3ForCausalLM`; without QK-norm → `LlamaForCausalLM`; all-GPT-2 settings → `GPT2LMHeadModel`. `export_hf` writes fp32 weights by default (~1.35 GB for 337M parameters). Keep fp32 as the master copy; Module 11 can publish a bf16 copy (~0.67 GB) for the demo.

### Optional: a private copy on the Hugging Face Hub

| Option | Cost | Trade-off |
|---|---|---|
| Bucket only (`hf/my-gpt-350m-base/`) | ~$0.02/month | Enough for Modules 8–11; one copy |
| **Bucket + private Hub repo** | $0 (free accounts include 100 GB of private storage) | A second copy in another company; Module 11 makes the final model public |
| Public Hub repo now | $0 | Not yet: Module 8 has not measured it and Module 11 writes the model card first |

> [!WARNING]
> `python -m mygpt.export_hf ... --push you/repo` creates the repo with the Hub's default visibility (public for a personal account). For a private copy, create the repo first with `hf repos create ... --private`, then upload into it.

> [!CAUTION]
> Terminating the node deletes its disk, and `rclone delete` in the bucket cannot be undone. Terminate only after the final checkpoint, `log.jsonl` and the HF export are listed in the bucket with the right sizes; delete bucket files only with a `--dry-run` first and a filter that names what to keep.

## Do this

### 1. Confirm the end of the run (5 min, node)

```bash
cd ~/mygpt && tail -4 runs/main_350m.out
ls -l runs/main_350m/ckpt_00038146.pt
rclone lsf r2:mygpt/ckpt/main_350m/ --format "sp" | sort -t';' -k2 | tail -4
```

The output ends with the last step's `val_loss`, `val_bpb` and `hellaswag acc_norm` lines; the training process has exited (no `mygpt.train` in `ps aux`). The bucket lists `ckpt_00038146.pt` with exactly the local size and `log.jsonl`. If either is missing: `bash scripts/sync.sh push-ckpt main_350m runs/main_350m` and list again. (If you shortened the run in 07.2, your last step is lower, e.g. `ckpt_00030517.pt` for 16B tokens.)

### 2. Export to Hugging Face format (5 min, node)

```bash
python -m mygpt.export_hf --ckpt runs/main_350m --out hf/my-gpt-350m-base 2>&1 | tee reports/export-main.txt
ls -la hf/my-gpt-350m-base
```

The tokenizer is found automatically next to the shards (`data/tok/bpe32k-100BT-30/tokenizer.json`). The printed JSON must show your architecture, `"max_abs_logit_diff_fp32"` below `1e-3`, `"params"` ~337M, `"step": 38146` and `"tokens"` ~20B. The folder holds `config.json`, `generation_config.json`, `model.safetensors` (~1.35 GB), the tokenizer files with the chat template, and `mygpt_export.json`.

### 3. Check it loads and talks (5 min, node)

```bash
python scripts/chat.py --model hf/my-gpt-350m-base --prompt "The three states of matter are"
python - <<'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("hf/my-gpt-350m-base")
t = AutoTokenizer.from_pretrained("hf/my-gpt-350m-base")
ids = t("Photosynthesis is the process by which plants")["input_ids"]
print(type(m).__name__, sum(p.numel() for p in m.parameters()) / 1e6, "M params")
print("roundtrip ok:", t.decode(ids) == "Photosynthesis is the process by which plants", "| eos:", t.eos_token)
EOF
```

The completion should be fluent, on-topic English for a sentence or two (a base model, so no chat behaviour yet; it may ramble or repeat after that). Gibberish or one repeated token after a passing logit check almost always means the wrong tokenizer was attached: re-export with `--tokenizer data/tok/bpe32k-100BT-30`.

### 4. Upload the export (5 min, node)

```bash
bash scripts/sync.sh push-hf my-gpt-350m-base hf/my-gpt-350m-base
rclone lsf r2:mygpt/hf/my-gpt-350m-base/ --format "sp"
git add reports/export-main.txt && git commit -m "Main run: HF export" && git push
```

Optional private Hub copy (needs `HF_TOKEN` exported in this shell):

```bash
hf auth login --token $HF_TOKEN
hf repos create <you>/my-gpt-350m-base --private
hf upload <you>/my-gpt-350m-base hf/my-gpt-350m-base . --private
```

Open `https://huggingface.co/<you>/my-gpt-350m-base` while logged in: it shows the "Private" badge and the files.

### 5. Copy the evidence off and terminate (5 min)

On your laptop:

```bash
scp -P <PORT> root@<POD_IP>:mygpt/runs/{main_350m.out,watch.out} reports/
git add reports/ && git commit -m "Main run: console and watcher logs" && git push
```

Then kill the guards and terminate: RunPod console → the pod → **Terminate** (not Stop). On Lambda: **Terminate** in the dashboard; on Vast.ai: **Destroy** the instance. The pod disappears from the list.

### 6. Hunt for anything else that bills (10 min, laptop)

| Where | Look for | Action |
|---|---|---|
| RunPod → Pods | Any pod in Running or Stopped (stopped pods bill their volume disk at $0.20/GB-mo) | Terminate what you do not need this week |
| RunPod → Storage | Network volumes ($0.07/GB-mo, billed whether used or not) | Delete those you are not using for Modules 8–10 |
| Lambda → Instances, File systems | Running instances, persistent file systems | Terminate / delete |
| Vast.ai → Instances | Stopped instances still bill storage | Destroy |
| Bucket | Old checkpoints of this run and rolled-back states | Step 7 |

### 7. Prune the bucket (10 min, laptop)

Keep the final and the pre-decay checkpoint, delete the other checkpoints of this run. Always dry-run first:

```bash
rclone delete r2:mygpt/ckpt/main_350m/ --dry-run \
  --filter "- /ckpt_00038146.pt" --filter "- /ckpt_00030000.pt" --filter "+ ckpt_*.pt" --filter "- *"
```

The dry run lists only checkpoints you mean to delete (the intermediate ones and anything under `rolled_back/`) and never `log.jsonl`, the final or the pre-decay checkpoint. Then run the same command without `--dry-run`, and check what is left:

```bash
rclone lsf r2:mygpt/ckpt/main_350m/ -R --format "sp"
rclone size r2:mygpt
```

### 8. Reconcile the budget (10 min, laptop)

RunPod → **Billing** → usage (Lambda: Billing → Usage; Vast.ai: Billing → Invoices): add up every charge of the main run, including dead nodes and incident time. Fill in the open ledger line from 07.1, then the close-out table of `reports/main-run-log.md`:

$$\text{deviation} = \frac{\text{actual} - \text{predicted}}{\text{predicted}}$$

Example: predicted $187, actual $201 → +7.5%. Above +15%, write one line on where it went (an incident, a slower node, a late terminate), because Module 11's cost ledger reports it. Update the money-left line: cap $500 minus all actuals, and the monthly storage from `rclone size` × $0.015.

## How you know it worked

- `reports/export-main.txt` shows a standard class, `max_abs_logit_diff_fp32` below 1e-3 and the final step; `chat.py` produced fluent English.
- `rclone lsf r2:mygpt/hf/my-gpt-350m-base/` lists `model.safetensors` (~1.35 GB), `config.json`, `tokenizer.json`, `tokenizer_config.json`, `mygpt_export.json`.
- `rclone lsf r2:mygpt/ckpt/main_350m/ -R` lists exactly `ckpt_00030000.pt`, `ckpt_00038146.pt` and `log.jsonl`.
- The provider consoles show no running or stopped pod and no volume you did not decide to keep; the RunPod balance stops falling.
- `budget.md` has the real cost of the run, the deviation from the prediction and the money left.

## When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `ValueError: ablation-only combination` from `export_hf` | The model switches do not match an exportable bundle (should have been caught in 06.4 step 3) | Do not terminate. Keep the native checkpoint in the bucket and file the mismatch; exporting would need code changes, so evaluate natively with `mygpt.evaluate` for now |
| `RuntimeError: exported ... logits differ` | Mapping or dtype mismatch in the export | Do not terminate; note the difference value and the class; re-run with `--dtype fp32`; report it with `reports/export-main.txt` |
| `tokenizer ... not found on this box; pass --tokenizer DIR` | Exporting on a box without the shards | `bash scripts/sync.sh pull-tokenizer tokenizers/bpe32k` and pass `--tokenizer tokenizers/bpe32k` |
| Completion is gibberish but the logit check passed | Wrong tokenizer attached (e.g. GPT-2's) | Re-export with `--tokenizer data/tok/bpe32k-100BT-30` |
| `hf upload` returns 403 | Token without write scope, or the free private-storage quota is used up | Create a token with write access; delete unused private repos or keep the bucket copy only |
| `rclone delete --dry-run` lists `log.jsonl` or the final checkpoint | Filter order wrong (the first matching rule wins) | Keep the two `- /ckpt_...` keep-rules first, then `+ ckpt_*.pt`, then `- *` |
| The bill is much higher than the watcher's projection | Pod left running after the run, or an old stopped pod with a volume | Usage page per pod and hour; terminate it; note the cost as an incident |

## Artifact

- In the bucket: `hf/my-gpt-350m-base/` (the base model for Modules 8–11), `ckpt/main_350m/ckpt_00038146.pt`, `ckpt/main_350m/ckpt_00030000.pt`, `ckpt/main_350m/log.jsonl`. Optional: private Hub repo `<you>/my-gpt-350m-base`.
- In git: `reports/export-main.txt`, `reports/main-run-log.md` (closed), `reports/main_350m.out`, `reports/watch.out`.
- `budget.md`: the completed main-run line and the money left, for example:

  ```text
  | 2026-11-17 | 07.1–07.3 | RunPod Secure 8x H100 SXM, main run 337M x 20B, 5.7 h at $35.12/h (incl. 1 rollback) | ~$187 | $201 |
  ```
