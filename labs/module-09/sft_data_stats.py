"""Measure a chat or preference dataset through YOUR tokenizer and chat template before you rent a GPU.

  python tools/sft_data_stats.py --model hf/my-gpt-350m-base --dataset HuggingFaceTB/smol-smoltalk
  python tools/sft_data_stats.py --model hf/my-gpt-350m-sft --dataset trl-lib/ultrafeedback_binarized --kind dpo
  python tools/sft_data_stats.py --model hf/my-gpt-350m-base --dataset my_chats.jsonl --n 0      # whole local file

It streams --n rows (no full download), renders each with the tokenizer's chat template (the
same plain-text <|user|>/<|assistant|> template that sft.py trains with) and reports:

  sft: tokens per conversation, share over --max-length (truncated), assistant (trained) share,
       conversations whose assistant tokens are cut away completely (TRL drops them);
  dpo: prompt / chosen / rejected lengths and the share of pairs longer than --max-length.

Then it extrapolates to the whole split and turns tokens into compute, GPU-hours and dollars:
  SFT: 6 * params * tokens per epoch;  DPO: (6 + 2) * params * tokens (policy fwd+bwd, reference fwd).
The MFU default (0.20) is an ESTIMATE for an un-compiled Hugging Face Trainer on one H100 with
padded batches; replace it with what you measure in the first minutes of the real run.
Needs: transformers, datasets (pip install -e ".[post]"). CPU only, a few minutes, $0.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics

PEAK = {"h100": 989e12, "a100": 312e12, "h200": 989e12, "l40s": 362e12, "rtx4090": 165e12}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF model dir (tokenizer + chat template + config are read from it)")
    p.add_argument("--dataset", required=True, help="hub id or local .jsonl")
    p.add_argument("--config", default=None, help="dataset config name, e.g. 'all' for smoltalk")
    p.add_argument("--split", default="train")
    p.add_argument("--kind", choices=["sft", "dpo"], default="sft")
    p.add_argument("--n", type=int, default=3000, help="rows to sample (0 = all, only sensible for local files)")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--epochs", type=float, default=None, help="default: 2 for sft, 1 for dpo (the script defaults)")
    p.add_argument("--rows", type=int, default=0, help="rows used for training (0 = whole split, or --max-samples)")
    p.add_argument("--params", type=float, default=0, help="model parameters (default: counted from the model dir)")
    p.add_argument("--gpu", choices=sorted(PEAK), default="h100")
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--mfu", type=float, default=0.20, help="ESTIMATE; replace with your measured value")
    p.add_argument("--price-per-gpu-hour", type=float, default=2.69)
    p.add_argument("--json", default=None, help="also write the report as JSON here")
    return p.parse_args(argv)


def count_params(model_dir: str) -> float:
    info = os.path.join(model_dir, "mygpt_export.json")
    if os.path.exists(info):
        with open(info, encoding="utf-8") as f:
            return float(json.load(f)["params"])
    from transformers import AutoConfig, AutoModelForCausalLM
    import torch
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model_dir))
    seen, total = set(), 0
    for prm in model.parameters():  # tied weights counted once
        if id(prm) not in seen:
            seen.add(id(prm))
            total += prm.numel()
    return float(total)


def rows(args):
    from datasets import load_dataset
    if args.dataset.endswith((".jsonl", ".json")):
        ds = load_dataset("json", data_files=args.dataset, split="train", streaming=True)
        total = None
    else:
        ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True)
        total = None
        try:
            from datasets import load_dataset_builder
            info = load_dataset_builder(args.dataset, args.config).info
            total = info.splits[args.split].num_examples if info.splits else None
        except Exception:  # noqa: BLE001 - size is optional, the sample is what matters
            total = None
    it = iter(ds)
    return (itertools.islice(it, args.n) if args.n else it), total


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def sft_stats(tok, sample, max_len):
    total, kept, trained, over, dropped = [], [], [], 0, 0
    for r in sample:
        enc = tok.apply_chat_template(r["messages"], return_dict=True, return_assistant_tokens_mask=True)
        ids, mask = enc["input_ids"], enc["assistant_masks"]
        total.append(len(ids))
        kept.append(min(len(ids), max_len))
        t = sum(mask[:max_len])
        trained.append(t)
        over += len(ids) > max_len
        dropped += t == 0
    n = len(total)
    return {
        "rows_sampled": n,
        "tokens_mean": statistics.mean(total), "tokens_median": statistics.median(total),
        "tokens_p90": pct(total, 0.9), "tokens_max": max(total),
        "share_truncated": over / n, "share_dropped_no_assistant_tokens": dropped / n,
        "kept_tokens_mean": statistics.mean(kept), "trained_tokens_mean": statistics.mean(trained),
        "trained_share_of_kept": sum(trained) / sum(kept),
    }, statistics.mean(kept)


def dpo_stats(tok, sample, max_len):
    pl, cl, rl, over = [], [], [], 0
    for r in sample:
        chosen, rejected = r["chosen"], r["rejected"]
        prompt = r.get("prompt") or chosen[:-1]
        p_ids = tok.apply_chat_template(prompt, add_generation_prompt=True, return_dict=True)["input_ids"]
        c_ids = tok.apply_chat_template(prompt + chosen[-1:], return_dict=True)["input_ids"]
        r_ids = tok.apply_chat_template(prompt + rejected[-1:], return_dict=True)["input_ids"]
        pl.append(len(p_ids))
        cl.append(len(c_ids) - len(p_ids))
        rl.append(len(r_ids) - len(p_ids))
        over += max(len(c_ids), len(r_ids)) > max_len
    n = len(pl)
    # tokens processed per pair: prompt+chosen and prompt+rejected, each capped at max_len
    per_pair = statistics.mean(min(p + c, max_len) + min(p + r, max_len) for p, c, r in zip(pl, cl, rl))
    return {
        "rows_sampled": n,
        "prompt_tokens_mean": statistics.mean(pl), "prompt_tokens_p90": pct(pl, 0.9),
        "chosen_tokens_mean": statistics.mean(cl), "rejected_tokens_mean": statistics.mean(rl),
        "chosen_minus_rejected_mean": statistics.mean(c - r for c, r in zip(cl, rl)),
        "share_chosen_longer": sum(c > r for c, r in zip(cl, rl)) / n,
        "share_over_max_length": over / n,
        "tokens_per_pair_after_cap": per_pair,
    }, per_pair


def main(argv=None):
    args = parse_args(argv)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if not tok.chat_template:
        raise SystemExit(f"{args.model} has no chat template; export it with mygpt.export_hf")
    sample, split_rows = rows(args)
    sample = list(sample)
    if not sample:
        raise SystemExit("no rows read")
    if args.kind == "sft":
        report, per_row = sft_stats(tok, sample, args.max_length)
        flops_per_token_factor, epochs = 6.0, args.epochs if args.epochs is not None else 2.0
    else:
        report, per_row = dpo_stats(tok, sample, args.max_length)
        flops_per_token_factor, epochs = 8.0, args.epochs if args.epochs is not None else 1.0
    n_rows = args.rows or split_rows or len(sample)
    params = args.params or count_params(args.model)
    tokens_per_epoch = per_row * n_rows
    flops = flops_per_token_factor * params * tokens_per_epoch * epochs
    gpu_seconds = flops / (PEAK[args.gpu] * args.mfu)
    gpu_hours = gpu_seconds / 3600
    report.update({
        "dataset": args.dataset, "kind": args.kind, "max_length": args.max_length,
        "rows_in_split": split_rows, "rows_trained": n_rows, "epochs": epochs, "params": params,
        "tokens_per_epoch": tokens_per_epoch, "train_flops": flops,
        "assumed_mfu": args.mfu, "gpu": args.gpu, "num_gpus": args.num_gpus,
        "gpu_hours": gpu_hours, "wall_hours": gpu_hours / args.num_gpus,
        "dollars": gpu_hours * args.price_per_gpu_hour, "price_per_gpu_hour": args.price_per_gpu_hour,
    })
    for k, v in report.items():
        if isinstance(v, float):
            v = f"{v:.3g}" if abs(v) >= 1e4 or abs(v) < 1e-2 else f"{v:.3f}"
        print(f"{k:36s} {v}")
    print(f"\nESTIMATE: {epochs:g} epoch(s) ~ {gpu_hours:.2f} GPU-h on {args.num_gpus}x {args.gpu.upper()} "
          f"at MFU {args.mfu:.0%} -> ~${report['dollars']:.2f} at ${args.price_per_gpu_hour}/GPU-h")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    main()
