"""Supervised fine-tuning (SFT) of your exported HF model with TRL.

  python scripts/sft.py --model hf/my-gpt-350m --out hf/my-gpt-350m-sft
  python scripts/sft.py --model hf/my-gpt-350m --dataset HuggingFaceTB/smoltalk --dataset-config all \
      --max-samples 200000 --epochs 1 --out hf/my-gpt-350m-sft
  accelerate launch --num_processes 8 scripts/sft.py ...      # multi-GPU (DDP via accelerate)

What it does: loads a chat dataset with a `messages` column, formats it with the chat template
written by mygpt.export_hf (plain-text <|user|>/<|assistant|> tags, EOT ends each answer),
trains with loss only on assistant tokens (--assistant-only-loss, default on), and saves the
model + tokenizer to --out. Any other SFTConfig field: --set key=value (repeatable).
Install: pip install -e .[post]
"""
from __future__ import annotations

import argparse

import yaml


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF model dir (from mygpt.export_hf) or hub id")
    p.add_argument("--dataset", default="HuggingFaceTB/smol-smoltalk", help="HF dataset id or local .jsonl")
    p.add_argument("--dataset-config", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--eval-split", default="test", help="'' to skip eval")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=16, help="per device")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-length", type=int, default=1024, help="must be <= the model's context")
    p.add_argument("--assistant-only-loss", type=lambda s: s.lower() != "false", default=True)
    p.add_argument("--packing", action="store_true", help="pack short conversations (faster, slightly different loss)")
    p.add_argument("--chat-template", default=None, help="file with a Jinja chat template to override the tokenizer's")
    p.add_argument("--lora", action="store_true", help="train LoRA adapters (PEFT) instead of all weights")
    p.add_argument("--report-to", default="none", help="none | wandb")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="extra SFTConfig field")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # Heavy imports only after argparse, so --help is instant.
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(args.model)
    if args.chat_template:
        tok.chat_template = open(args.chat_template, encoding="utf-8").read()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16 if bf16 else torch.float32)

    if args.dataset.endswith((".jsonl", ".json")):
        ds = load_dataset("json", data_files=args.dataset, split="train")
        train_ds, eval_ds = ds, None
    else:
        train_ds = load_dataset(args.dataset, args.dataset_config, split=args.split)
        eval_ds = None
        if args.eval_split:
            try:
                eval_ds = load_dataset(args.dataset, args.dataset_config, split=args.eval_split)
                eval_ds = eval_ds.select(range(min(len(eval_ds), 2000)))
            except ValueError:
                eval_ds = None
    if args.max_samples:
        train_ds = train_ds.shuffle(seed=0).select(range(min(args.max_samples, len(train_ds))))
    keep = {"messages"}
    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep])
    if eval_ds is not None:
        eval_ds = eval_ds.remove_columns([c for c in eval_ds.column_names if c not in keep])

    extra = {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in args.set)}
    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        assistant_only_loss=args.assistant_only_loss,
        packing=args.packing,
        bf16=bf16,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=500,
        report_to=args.report_to,
        **extra,
    )
    peft_config = None
    if args.lora:
        from peft import LoraConfig
        peft_config = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds, eval_dataset=eval_ds,
                         processing_class=tok, peft_config=peft_config)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"saved SFT model to {args.out}")


if __name__ == "__main__":
    main()
