"""Token shards on disk and a deterministic, resumable distributed loader.

Shard file: 256 x int32 header [magic=20261113, version=1, n_tokens, vocab_size, 0...]
followed by n_tokens x uint16. Each document starts with the tokenizer's EOT id.
DIR/meta.json describes the tokenizer and token/byte counts (for bits-per-byte).
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

MAGIC, VERSION, HEADER_INTS = 20261113, 1, 256
HEADER_BYTES = HEADER_INTS * 4


def write_shard(path: str, tokens, vocab_size: int) -> None:
    tokens = np.asarray(tokens)
    assert vocab_size <= 2 ** 16 and (tokens.size == 0 or int(tokens.max()) < 2 ** 16)
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[:4] = [MAGIC, VERSION, tokens.size, vocab_size]
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.astype(np.uint16).tobytes())
    os.replace(tmp, path)


def read_header(path: str) -> dict:
    header = np.fromfile(path, dtype=np.int32, count=HEADER_INTS)
    if header[0] != MAGIC:
        raise ValueError(f"{path}: bad magic {header[0]} (not a mygpt shard)")
    if header[1] != VERSION:
        raise ValueError(f"{path}: unsupported shard version {header[1]}")
    return {"n_tokens": int(header[2]), "vocab_size": int(header[3])}


def read_shard(path: str) -> np.ndarray:
    """Memory-mapped uint16 view of the shard's tokens (no copy)."""
    n = read_header(path)["n_tokens"]
    return np.memmap(path, dtype=np.uint16, mode="r", offset=HEADER_BYTES, shape=(n,))


def shard_paths(data_dir: str, split: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(data_dir, f"{split}_*.bin")))
    if not paths:
        raise FileNotFoundError(f"no {split}_*.bin shards in {data_dir}")
    return paths


def read_meta(data_dir: str) -> dict:
    p = os.path.join(data_dir, "meta.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def write_meta(data_dir: str, meta: dict) -> None:
    with open(os.path.join(data_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def make_synthetic_shards(data_dir: str, vocab_size: int = 256, train_tokens: int = 400_000,
                          val_tokens: int = 40_000, seed: int = 0) -> None:
    """Learnable toy data for smoke tests: a sparse Markov chain (each token has 4 likely
    successors), documents of random length starting with EOT=0. Entropy ~ln(4) << ln(V)."""
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    succ = rng.integers(1, vocab_size, size=(vocab_size, 4))

    def stream(n):
        out = np.empty(n, dtype=np.uint16)
        i = 0
        while i < n:
            doc_len = int(rng.integers(32, 256))
            out[i] = 0
            tok = int(rng.integers(1, vocab_size))
            for j in range(i + 1, min(i + doc_len, n)):
                out[j] = tok
                tok = int(succ[tok, rng.integers(0, 4)])
            i += doc_len
        return out

    write_shard(os.path.join(data_dir, "val_000000.bin"), stream(val_tokens), vocab_size)
    write_shard(os.path.join(data_dir, "train_000000.bin"), stream(train_tokens), vocab_size)
    write_meta(data_dir, {"tokenizer": "synthetic", "vocab_size": vocab_size, "eot_id": 0,
                          "train_tokens": train_tokens, "val_tokens": val_tokens,
                          "train_bytes": train_tokens, "val_bytes": val_tokens,
                          "total_tokens": train_tokens + val_tokens,
                          "total_bytes": train_tokens + val_tokens})


class ShardDataset:
    """All shards of one split, memory-mapped lazily."""

    def __init__(self, data_dir: str, split: str):
        self.paths = shard_paths(data_dir, split)
        self.lengths = [read_header(p)["n_tokens"] for p in self.paths]
        self.vocab_size = read_header(self.paths[0])["vocab_size"]
        self._cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def total_tokens(self) -> int:
        return sum(self.lengths)

    def shard(self, i: int) -> np.ndarray:
        if i not in self._cache:
            self._cache.clear()  # keep one shard mapped at a time
            self._cache[i] = read_shard(self.paths[i])
        return self._cache[i]


class DistributedTokenLoader:
    """Walks the shards in order. Each step, rank r takes the r-th (B*T+1)-token window of a
    global B*T*world chunk. State = (shard, position) shared by all ranks, so resuming with a
    different world size still continues exactly where the token stream stopped."""

    def __init__(self, data_dir: str, split: str, B: int, T: int, rank: int = 0, world_size: int = 1,
                 device="cpu"):
        import torch  # lazy: the data-prep scripts use this module without paying for torch
        self.ds = ShardDataset(data_dir, split)
        self.B, self.T, self.rank, self.world = B, T, rank, world_size
        self.device = torch.device(device)
        self.chunk = B * T * world_size
        if max(self.ds.lengths) < self.chunk + 1:
            raise ValueError(f"{split} shards are smaller than one global batch ({self.chunk + 1} tokens)")
        self.reset()

    def reset(self):
        self.shard_idx, self.pos = 0, 0

    def state_dict(self) -> dict:
        return {"shard_idx": self.shard_idx, "pos": self.pos}

    def load_state_dict(self, s: dict):
        self.shard_idx, self.pos = int(s["shard_idx"]), int(s["pos"])

    def next_batch(self):
        import torch
        if self.pos + self.chunk + 1 > self.ds.lengths[self.shard_idx]:
            self.shard_idx = (self.shard_idx + 1) % len(self.ds)
            self.pos = 0
            while self.ds.lengths[self.shard_idx] < self.chunk + 1:  # skip tiny tail shards
                self.shard_idx = (self.shard_idx + 1) % len(self.ds)
        start = self.pos + self.rank * self.B * self.T
        buf = torch.from_numpy(self.ds.shard(self.shard_idx)[start:start + self.B * self.T + 1].astype(np.int64))
        self.pos += self.chunk
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)
        if self.device.type == "cuda":
            return x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        return x.to(self.device), y.to(self.device)
