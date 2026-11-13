"""Direct Preference Optimization (DPO) of your SFT model with TRL.

  python scripts/dpo.py --model hf/my-gpt-350m-sft --out hf/my-gpt-350m-dpo
  python scripts/dpo.py --model hf/my-gpt-350m-sft --dataset trl-lib/ultrafeedback_binarized \
      --beta 0.1 --lr 5e-7 --epochs 1 --out hf/my-gpt-350m-dpo

The dataset needs `chosen` / `rejected` columns of conversational message lists;
trl-lib/ultrafeedback_binarized is already in that shape. Datasets that add a string `prompt`
column next to message lists (HuggingFaceH4/ultrafeedback_binarized, argilla/*, Tulu 3 preference
mixtures) are normalized automatically: if `chosen` already starts with the user turn the string
prompt is dropped, otherwise it becomes a one-message user turn. Other columns are dropped.
The reference model is a frozen copy of --model (TRL creates it, in the same fp32 dtype).
Weights are loaded and kept in fp32 (master weights) and trained under bf16 autocast on a bf16
GPU, so the tiny DPO updates are not rounded away. --resume continues from the newest
OUT/checkpoint-*; --save-final-only deletes the checkpoints after the final save.
Any other DPOConfig field: --set key=value (repeatable). Install: pip install -e .[post]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import load_hf_dataset, parse_set, remove_checkpoints, resume_checkpoint  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="SFT model dir")
    p.add_argument("--dataset", default="trl-lib/ultrafeedback_binarized", help="HF dataset id or local .jsonl")
    p.add_argument("--dataset-config", default=None)
    p.add_argument("--dataset-revision", default=None, metavar="SHA", help="pin the hub dataset to this commit/tag")
    p.add_argument("--split", default="train")
    p.add_argument("--eval-split", default="test", help="'' to skip eval")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--beta", type=float, default=0.1, help="KL strength: higher = stay closer to the SFT model")
    p.add_argument("--lr", type=float, default=5e-7)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=8, help="per device")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--report-to", default="none", help="none | wandb")
    p.add_argument("--resume", action="store_true", help="continue from the newest OUT/checkpoint-* (fresh start if none)")
    p.add_argument("--save-final-only", action="store_true", help="delete OUT/checkpoint-* after saving the final model")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="extra DPOConfig field")
    return p.parse_args(argv)


def _is_messages(value) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(m, dict) and "role" in m for m in value)


def normalize_preferences(ds):
    """Keep only prompt/chosen/rejected, and make a string `prompt` next to conversational
    chosen/rejected consistent (TRL rejects that mix):
      - chosen starts with a non-assistant turn (it already contains the prompt) -> drop `prompt`
        (TRL then takes the shared prefix of chosen/rejected as the prompt);
      - chosen holds only the assistant answer -> prompt becomes [{"role": "user", "content": prompt}].
    String chosen/rejected (non-conversational) datasets pass through with the extra columns removed."""
    cols = set(ds.column_names)
    if not {"chosen", "rejected"} <= cols:
        raise SystemExit(f"DPO dataset needs 'chosen' and 'rejected' columns, has {sorted(cols)}")
    ds = ds.remove_columns([c for c in ds.column_names if c not in ("prompt", "chosen", "rejected")])
    if "prompt" not in ds.column_names or len(ds) == 0:
        return ds
    first = ds[0]
    if not (_is_messages(first["chosen"]) and isinstance(first["prompt"], str)):
        return ds
    if first["chosen"][0]["role"] != "assistant":
        print("dataset: string 'prompt' next to full conversations -> dropped (prompt = shared prefix)", flush=True)
        return ds.remove_columns(["prompt"])
    print("dataset: string 'prompt' next to answer-only messages -> converted to a user turn", flush=True)
    return ds.map(lambda r: {"prompt": [{"role": "user", "content": r["prompt"]}]})


def main(argv=None):
    args = parse_args(argv)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    # fp32 master weights always; bf16 only as autocast (bf16=True below). At DPO learning rates
    # (5e-7..1e-6) bf16-stored weights would keep only ~1-2% of each update.
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)

    train_ds = load_hf_dataset(args.dataset, args.dataset_config, args.split, args.dataset_revision)
    eval_ds = None
    if args.eval_split and not args.dataset.endswith((".jsonl", ".json")):
        try:
            eval_ds = load_hf_dataset(args.dataset, args.dataset_config, args.eval_split, args.dataset_revision)
            eval_ds = eval_ds.select(range(min(len(eval_ds), 1000)))
        except ValueError:
            eval_ds = None
    if args.max_samples:
        train_ds = train_ds.shuffle(seed=0).select(range(min(args.max_samples, len(train_ds))))
    train_ds = normalize_preferences(train_ds)
    if eval_ds is not None:
        eval_ds = normalize_preferences(eval_ds)

    kw = dict(
        output_dir=args.out,
        beta=args.beta,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.1,  # float < 1 = fraction of total steps
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        bf16=bf16,  # autocast; the weights stay fp32
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=500,
        report_to=args.report_to,
    )
    cfg = DPOConfig(**{**kw, **parse_set(args.set)})  # --set wins over the defaults above
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=train_ds, eval_dataset=eval_ds, processing_class=tok)
    trainer.train(resume_from_checkpoint=resume_checkpoint(args.out, args.resume))
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    if args.save_final_only and trainer.is_world_process_zero():
        removed = remove_checkpoints(args.out)
        print(f"--save-final-only: removed {len(removed)} checkpoint folder(s)")
    print(f"saved DPO model to {args.out}")
    return trainer


if __name__ == "__main__":
    main()
