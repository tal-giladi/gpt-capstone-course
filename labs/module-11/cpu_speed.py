"""Measure generation speed of an HF-format model on CPU, the way a CPU host would run it (lesson 11.2).

    python tools/cpu_speed.py --model release/my-gpt-350m --threads 2            # like a 2-vCPU Space / VPS
    python tools/cpu_speed.py --model release/my-gpt-350m --threads 2 --dtype bf16

Prints tokens per second for a fixed 200-token prompt and 128 new tokens (greedy, EOS ignored so
every run generates the same number of tokens), and the time a 160-token answer would take.
"""
from __future__ import annotations

import argparse
import time


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--prompt-tokens", type=int, default=200)
    p.add_argument("--new-tokens", type=int, default=128)
    a = p.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(a.threads)
    dtype = torch.float32 if a.dtype == "fp32" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).eval()
    n_params = sum(t.numel() for t in model.parameters())
    vocab = model.config.vocab_size
    ids = torch.randint(1, vocab, (1, a.prompt_tokens), generator=torch.Generator().manual_seed(0))
    kw = dict(do_sample=False, eos_token_id=None, pad_token_id=0)
    with torch.no_grad():
        model.generate(ids, max_new_tokens=8, **kw)  # warm-up
        t = time.perf_counter()
        model.generate(ids, max_new_tokens=a.new_tokens, min_new_tokens=a.new_tokens, **kw)
        dt = time.perf_counter() - t
    tps = a.new_tokens / dt
    print(f"{n_params / 1e6:.0f}M params, {a.threads} threads, {a.dtype}: {tps:.1f} tokens/s "
          f"-> a 160-token answer takes ~{160 / tps:.0f} s")


if __name__ == "__main__":
    main()
