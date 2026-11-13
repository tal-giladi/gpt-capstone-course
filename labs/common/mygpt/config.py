"""Config dataclasses and YAML loading with `section.key=value` overrides."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from typing import Any

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 50304          # GPT-2's 50257 padded to a multiple of 128 for speed
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
    dropout: float = 0.0
    pos_emb: str = "learned"         # learned | rope
    rope_theta: float = 10000.0
    norm: str = "layernorm"          # layernorm | rmsnorm
    norm_eps: float = 1e-5
    mlp: str = "gelu"                # gelu | swiglu
    qk_norm: bool = False
    bias: bool = True
    tie_embeddings: bool = True

    def __post_init__(self):
        assert self.pos_emb in ("learned", "rope"), self.pos_emb
        assert self.norm in ("layernorm", "rmsnorm"), self.norm
        assert self.mlp in ("gelu", "swiglu"), self.mlp
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        if self.pos_emb == "rope":
            assert (self.n_embd // self.n_head) % 2 == 0, "rope needs an even head_dim"

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def mlp_hidden(self) -> int:
        if self.mlp == "gelu":
            return 4 * self.n_embd
        h = int(8 * self.n_embd / 3)
        return 64 * ((h + 63) // 64)  # round up to a multiple of 64


@dataclass
class TrainConfig:
    data_dir: str = "data/shards"     # dir with train_*.bin / val_*.bin / meta.json, or "synthetic"
    train_tokens: int = 10_000_000_000
    total_batch_tokens: int = 524_288  # tokens per optimizer step, across all GPUs
    micro_batch_size: int = 64        # sequences per GPU per forward
    seq_len: int = 1024
    lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_tokens: int = 375_000_000
    schedule: str = "cosine"          # cosine | wsd
    decay_frac: float = 0.2           # wsd: fraction of steps spent decaying
    optimizer: str = "adamw"          # adamw | muon
    muon_lr: float = 0.02
    weight_decay: float = 0.1         # AdamW groups only (>=2D tensors: embeddings, lm_head, ...)
    muon_weight_decay: float = 0.0    # Muon groups only; applied as p *= 1 - muon_lr * wd
    grad_clip: float = 1.0
    dtype: str = "bf16"               # bf16 | fp32
    compile: bool = False
    seed: int = 1337
    out_dir: str = "runs/default"
    run_name: str = "default"
    eval_every: int = 250
    eval_tokens: int = 10_485_760
    hellaswag_every: int = 0          # 0 = off
    ckpt_every: int = 1000
    keep_last: int = 3
    ckpt_remote: str = ""             # rclone remote path, e.g. r2:my-bucket/ckpt/run1 ("" = off)
    wandb_project: str = ""           # tracker project name; "" = no live tracker (log.jsonl only)
    tracker: str = "wandb"            # wandb | trackio | none
    wandb_alerts: bool = True         # with wandb on: run.alert() on a NaN stop and when the run finishes
    log_every: int = 10
    skip_batches: int = 0             # on resume, advance the loader this many global batches (optimizer steps)

    def __post_init__(self):
        assert self.schedule in ("cosine", "wsd"), self.schedule
        assert self.optimizer in ("adamw", "muon"), self.optimizer
        assert self.dtype in ("bf16", "fp32"), self.dtype


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return {"model": asdict(self.model), "train": asdict(self.train)}


def _coerce(value: Any, typ: Any) -> Any:
    """YAML reads `10e9` as a string and `1e-3` fine; coerce to the dataclass field type."""
    typ = {"int": int, "float": float, "bool": bool, "str": str}.get(typ, typ)
    if typ is bool:
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if typ is int:
        return int(float(value)) if isinstance(value, str) else int(value)
    if typ is float:
        return float(value)
    if typ is str:
        return "" if value is None else str(value)
    return value


def _build(cls, raw: dict):
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(raw) - set(fields)
    if unknown:
        raise KeyError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**{k: _coerce(v, fields[k].type) for k, v in raw.items()})


def config_from_dict(d: dict) -> Config:
    return Config(model=_build(ModelConfig, d.get("model") or {}),
                  train=_build(TrainConfig, d.get("train") or {}))


def load_config(path: str | None, overrides: list[str] | None = None) -> Config:
    """Load YAML then apply overrides like ['train.lr=6e-4', 'model.qk_norm=true']."""
    raw: dict = {}
    if path:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    raw.setdefault("model", {})
    raw.setdefault("train", {})
    for ov in overrides or []:
        key, sep, val = ov.partition("=")
        if not sep or "." not in key:
            raise ValueError(f"override must look like section.key=value, got {ov!r}")
        section, name = key.split(".", 1)
        if section not in ("model", "train"):
            raise ValueError(f"unknown section {section!r} in {ov!r}")
        raw[section][name] = yaml.safe_load(val)
    return config_from_dict(raw)
