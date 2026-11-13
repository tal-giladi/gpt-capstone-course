"""Direct Preference Optimization (DPO) of your SFT model with TRL.

  python scripts/dpo.py --model hf/my-gpt-350m-sft --out hf/my-gpt-350m-dpo
  python scripts/dpo.py --model hf/my-gpt-350m-sft --dataset trl-lib/ultrafeedback_binarized \
      --beta 0.1 --lr 5e-7 --epochs 1 --out hf/my-gpt-350m-dpo

The dataset needs `chosen` / `rejected` columns (conversational message lists, optionally a
`prompt` column); trl-lib/ultrafeedback_binarized is already in that shape. The reference
model is a frozen copy of --model (TRL creates it). Any other DPOConfig field:
--set key=value (repeatable). Install: pip install -e .[post]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import parse_set  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="SFT model dir")
    p.add_argument("--dataset", default="trl-lib/ultrafeedback_binarized")
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
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="extra DPOConfig field")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16 if bf16 else torch.float32)

    if args.dataset.endswith((".jsonl", ".json")):
        train_ds, eval_ds = load_dataset("json", data_files=args.dataset, split="train"), None
    else:
        train_ds = load_dataset(args.dataset, split=args.split)
        eval_ds = None
        if args.eval_split:
            try:
                eval_ds = load_dataset(args.dataset, split=args.eval_split)
                eval_ds = eval_ds.select(range(min(len(eval_ds), 1000)))
            except ValueError:
                eval_ds = None
    if args.max_samples:
        train_ds = train_ds.shuffle(seed=0).select(range(min(args.max_samples, len(train_ds))))

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
        bf16=bf16,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=500,
        report_to=args.report_to,
    )
    cfg = DPOConfig(**{**kw, **parse_set(args.set)})  # --set wins over the defaults above
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=train_ds, eval_dataset=eval_ds, processing_class=tok)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"saved DPO model to {args.out}")


if __name__ == "__main__":
    main()
