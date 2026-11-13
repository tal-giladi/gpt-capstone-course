"""Tokenize parquet documents into uint16 .bin shards (+ meta.json) for mygpt.train.

  python scripts/tokenize_shards.py --input data/raw/fineweb_edu_10BT --tokenizer gpt2 \
      --out data/fineweb_edu_10B_gpt2 --workers 32
  python scripts/tokenize_shards.py --input data/raw/fineweb_edu_100BT --tokenizer tokenizers/own32k \
      --out data/fineweb_edu_100B_own32k --max-tokens 21000000000 --workers 64

The first --val-tokens tokens (whole documents) become val_*.bin, everything after goes to
train_*.bin, so the val split never changes when you add more training data. Each document
is written as [EOT] + tokens. --exclude takes the JSON list written by decontam.py --drop.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

_TOK = None


def _init(tokenizer: str):
    global _TOK
    from mygpt.tokenizer import get_tokenizer
    _TOK = get_tokenizer(tokenizer)


def _encode(texts: list[str]):
    out = []
    for ids in _TOK.encode_batch(texts):
        arr = np.empty(len(ids) + 1, dtype=np.uint16)
        arr[0] = _TOK.eot_id
        arr[1:] = ids
        out.append(arr)
    return out, [len(t.encode("utf-8")) for t in texts]


class ShardWriter:
    def __init__(self, out_dir: str, shard_tokens: int, vocab_size: int):
        from mygpt.data import write_shard
        self._write = write_shard
        self.out_dir, self.size, self.vocab = out_dir, shard_tokens, vocab_size
        self.buf = np.empty(shard_tokens, dtype=np.uint16)
        self.fill = 0
        self.split = "val"
        self.index = {"val": 0, "train": 0}
        self.tokens = {"val": 0, "train": 0}
        self.bytes = {"val": 0, "train": 0}
        self.docs = {"val": 0, "train": 0}

    def flush(self):
        if self.fill:
            path = os.path.join(self.out_dir, f"{self.split}_{self.index[self.split]:06d}.bin")
            self._write(path, self.buf[:self.fill], self.vocab)
            self.index[self.split] += 1
            self.fill = 0

    def set_split(self, split: str):
        if split != self.split:
            self.flush()
            self.split = split

    def add(self, doc: np.ndarray, nbytes: int):
        self.tokens[self.split] += doc.size
        self.bytes[self.split] += nbytes
        self.docs[self.split] += 1
        pos = 0
        while pos < doc.size:
            take = min(self.size - self.fill, doc.size - pos)
            self.buf[self.fill:self.fill + take] = doc[pos:pos + take]
            self.fill += take
            pos += take
            if self.fill == self.size:
                self.flush()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="parquet file or directory")
    p.add_argument("--tokenizer", required=True, help="gpt2 | DIR with tokenizer.json")
    p.add_argument("--out", required=True)
    p.add_argument("--shard-tokens", type=int, default=100_000_000)
    p.add_argument("--val-tokens", type=int, default=50_000_000)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--max-tokens", type=int, default=0, help="stop after this many tokens in total (0 = all)")
    p.add_argument("--exclude", default=None, help="JSON list of [parquet_file_name, row] to skip (decontam --drop)")
    p.add_argument("--docs-per-task", type=int, default=256)
    args = p.parse_args(argv)

    from _common import iter_parquet_texts
    from mygpt.data import write_meta
    from mygpt.tokenizer import get_tokenizer

    tok = get_tokenizer(args.tokenizer)
    assert tok.vocab_size <= 2 ** 16, "vocab too large for uint16 shards"
    os.makedirs(args.out, exist_ok=True)
    exclude = set()
    if args.exclude:
        with open(args.exclude, encoding="utf-8") as f:
            exclude = {(os.path.basename(a), int(b)) for a, b in json.load(f)}

    def batches():
        batch, skipped = [], 0
        for file, row, text in iter_parquet_texts(args.input):
            if (os.path.basename(file), row) in exclude:
                skipped += 1
                continue
            batch.append(text)
            if len(batch) == args.docs_per_task:
                yield batch
                batch = []
        if batch:
            yield batch
        if skipped:
            print(f"skipped {skipped} excluded documents")

    w = ShardWriter(args.out, args.shard_tokens, tok.vocab_size)
    t0, last_print = time.time(), 0
    tok_name = "gpt2" if args.tokenizer == "gpt2" else os.path.abspath(args.tokenizer)
    with Pool(args.workers, initializer=_init, initargs=(tok_name,)) as pool:
        for docs, nbytes in pool.imap(_encode, batches(), chunksize=4):
            for doc, nb in zip(docs, nbytes):
                if w.split == "val" and w.tokens["val"] >= args.val_tokens:
                    w.set_split("train")
                w.add(doc, nb)
            total = w.tokens["val"] + w.tokens["train"]
            if total - last_print > 50_000_000:
                last_print = total
                print(f"{total / 1e9:.2f}B tokens | {total / (time.time() - t0) / 1e6:.1f}M tok/s", flush=True)
            if args.max_tokens and total >= args.max_tokens:
                pool.terminate()
                break
    w.flush()
    if tok_name != "gpt2":  # ship the tokenizer with the shards (sync.sh push-tok copies both)
        import shutil
        src = os.path.join(tok_name, "tokenizer.json") if os.path.isdir(tok_name) else tok_name
        shutil.copy(src, os.path.join(args.out, "tokenizer.json"))
    meta = {"tokenizer": "gpt2" if tok_name == "gpt2" else "tokenizer.json",
            "tokenizer_source": tok_name,"vocab_size": tok.vocab_size, "eot_id": tok.eot_id,
            "train_tokens": w.tokens["train"], "val_tokens": w.tokens["val"],
            "train_bytes": w.bytes["train"], "val_bytes": w.bytes["val"],
            "train_docs": w.docs["train"], "val_docs": w.docs["val"],
            "total_tokens": w.tokens["train"] + w.tokens["val"],
            "total_bytes": w.bytes["train"] + w.bytes["val"],
            "shard_tokens": args.shard_tokens, "source": os.path.abspath(args.input)}
    write_meta(args.out, meta)
    print(json.dumps(meta, indent=2))
    print(f"done in {time.time() - t0:.0f}s -> {args.out}")
    return meta


if __name__ == "__main__":
    main()
