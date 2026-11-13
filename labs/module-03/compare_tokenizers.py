"""Compare your BPE tokenizer with GPT-2's on held-out FineWeb-Edu documents (lesson 03.1).

    python tools/compare_tokenizers.py --tokenizer tokenizers/bpe32k \
        --input data/raw/sample-100BT/000_00000.parquet --docs 20000

Use documents the tokenizer was NOT trained on (03.1 trains on the 30th file, 002_00009, and
evaluates on the first documents of 000_00000, which later become the validation split).

Prints, for both tokenizers: UTF-8 bytes per token, tokens per document, the projected token
count of the main-run subset, and the per-token training-compute share of the output layer for
the main_350m model (d=1024, 24 layers, context 1024). Then checks the special tokens: the
end-of-text id, any other special tokens, and whether a literal "<|endoftext|>" typed inside
ordinary text is turned into the real end-of-text id (it must not be). Exit code 1 if a check
fails.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

EXPECTED_SPECIALS = ["<|endoftext|>"]
SAMPLES = [
    "The mitochondria is the powerhouse of the cell.",
    "In 1969, 600 million people watched the Apollo 11 landing.",
    "def area(r):\n    return 3.14159 * r ** 2",
    "Photosynthesis converts CO2 and H2O into glucose (C6H12O6).",
    "Le café est très bon à Montréal.",
]


def load_docs(path: str, n: int) -> list[str]:
    import pyarrow.parquet as pq
    docs: list[str] = []
    for batch in pq.ParquetFile(path).iter_batches(columns=["text"], batch_size=1024):
        docs.extend(t or "" for t in batch.column(0).to_pylist())
        if len(docs) >= n:
            break
    return docs[:n]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer", required=True, help="dir with tokenizer.json (your BPE)")
    p.add_argument("--input", required=True, help="one parquet file with held-out documents")
    p.add_argument("--docs", type=int, default=20000)
    p.add_argument("--gpt2-tokens-subset", type=float, default=22.54e9,
                   help="GPT-2 token estimate of the main-run subset (from your 02.3 manifest)")
    p.add_argument("--d-model", type=int, default=1024)
    p.add_argument("--n-layer", type=int, default=24)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--block-params", type=float, default=303_562_752,
                   help="non-embedding parameters (main_350m.yaml: 24 x (4*1024^2 + 3*1024*2752))")
    p.add_argument("--out", default=None, help="write the numbers as JSON here")
    args = p.parse_args(argv)

    from mygpt.tokenizer import get_tokenizer

    docs = load_docs(args.input, args.docs)
    nbytes = sum(len(d.encode("utf-8")) for d in docs)
    toks = {"gpt2": get_tokenizer("gpt2"), "own": get_tokenizer(args.tokenizer)}
    d, L = args.d_model, args.n_layer
    body_flops = 6 * args.block_params + 6 * L * args.seq_len * d   # per token: weights + attention
    res: dict = {"input": os.path.abspath(args.input), "docs": len(docs), "bytes": nbytes}
    print(f"{len(docs):,} held-out documents, {nbytes / 1e6:.1f} MB of UTF-8 text\n")
    print(f"{'tokenizer':10s} {'vocab':>7s} {'tokens':>12s} {'bytes/tok':>10s} {'tok/doc':>8s} "
          f"{'subset tokens':>14s} {'head share':>10s}")
    for name, tok in toks.items():
        n = sum(len(ids) for ids in tok.encode_batch(docs))
        bpt = nbytes / n
        res[name] = {"vocab": tok.vocab_size, "tokens": n, "bytes_per_token": round(bpt, 4)}
        print(f"{name:10s} {tok.vocab_size:>7,} {n:>12,} {bpt:>10.3f} {n / len(docs):>8.0f}", end="")
        head_flops = 6 * tok.vocab_size * d
        res[name]["head_share"] = head_flops / (body_flops + head_flops)
        if name == "gpt2":
            print(f" {args.gpt2_tokens_subset / 1e9:>13.2f}B {100 * res[name]['head_share']:>9.1f}%")
        else:
            proj = args.gpt2_tokens_subset * n / res["gpt2"]["tokens"]
            res[name]["subset_tokens_est"] = proj
            print(f" {proj / 1e9:>13.2f}B {100 * res[name]['head_share']:>9.1f}%")
    diff = res["own"]["tokens"] / res["gpt2"]["tokens"] - 1
    res["own_vs_gpt2_tokens"] = round(diff, 4)
    print(f"\nown tokenizer uses {100 * diff:+.1f}% tokens vs GPT-2 on the same text")

    print("\nexamples (own | gpt2 token counts):")
    for s in SAMPLES:
        own_ids = toks["own"].encode(s)
        print(f"  {len(own_ids):3d} | {len(toks['gpt2'].encode(s)):3d}  {s[:60]!r}")
        assert toks["own"].decode(own_ids) == s, f"round-trip failed for {s!r}"

    ok = True
    print("\nspecial tokens:")
    from tokenizers import Tokenizer
    path = os.path.join(args.tokenizer, "tokenizer.json") if os.path.isdir(args.tokenizer) else args.tokenizer
    raw = Tokenizer.from_file(path)
    specials = {t["content"]: t["id"] for t in json.load(open(path, encoding="utf-8")).get("added_tokens", [])}
    res["specials"] = specials
    for s in EXPECTED_SPECIALS:
        if s not in specials:
            print(f"  MISSING {s}  (train_tokenizer.py always adds it: wrong tokenizer.json?)")
            ok = False
        else:
            print(f"  {specials[s]:>6d}  {s}")
    extra = sorted(set(specials) - set(EXPECTED_SPECIALS))
    if extra:
        print(f"  other special tokens: {', '.join(f'{x} ({specials[x]})' for x in extra)}")
    if raw.get_vocab_size() != toks["own"].vocab_size:
        print(f"  vocab mismatch: {raw.get_vocab_size()} vs {toks['own'].vocab_size}")
        ok = False
    eot = toks["own"].eot_id
    literal = toks["own"].encode("Use <|endoftext|> to mark the end.")
    res["literal_eot_becomes_special"] = eot in literal
    if eot in literal:
        print("  FAIL   literal '<|endoftext|>' in ordinary text is encoded as the real EOT id.\n"
              "         Web pages that contain this string would inject fake document breaks.\n"
              "         Fix: the data tokenizer must set encode_special_tokens=True (see 03.1).")
        ok = False
    else:
        print("  OK     literal '<|endoftext|>' in ordinary text stays ordinary text")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"\nwrote {args.out}")
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
