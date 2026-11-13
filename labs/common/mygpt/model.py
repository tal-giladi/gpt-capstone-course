"""GPT with switchable parts: learned/RoPE positions, LayerNorm/RMSNorm, GELU/SwiGLU, QK-norm,
bias on/off, tied/untied embeddings.

Module names follow Hugging Face GPT-2 (`transformer.h.N.attn.c_attn`, ...) so the GPT-2
export is a near 1:1 rename; RoPE uses HF's non-interleaved `rotate_half` convention so the
Llama/Qwen3 export produces identical logits.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    """Same math as HF LlamaRMSNorm: normalize in fp32, cast back, then scale."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def make_norm(cfg: ModelConfig, dim: int) -> nn.Module:
    if cfg.norm == "rmsnorm":
        return RMSNorm(dim, cfg.norm_eps)
    return nn.LayerNorm(dim, eps=cfg.norm_eps, bias=cfg.bias)


def rope_cache(head_dim: int, max_len: int, theta: float, device=None):
    """cos/sin of shape (max_len, head_dim); halves duplicated (HF convention)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    t = torch.arange(max_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    # x: (B, H, T, D); cos/sin: (T, D)
    return (x * cos + rotate_half(x) * sin).to(x.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.n_head, self.head_dim = cfg.n_head, cfg.head_dim
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.c_proj.RESIDUAL_PROJ = True
        if cfg.qk_norm:  # per-head RMSNorm over head_dim, before RoPE (Qwen3 layout)
            self.q_norm = RMSNorm(self.head_dim, cfg.norm_eps)
            self.k_norm = RMSNorm(self.head_dim, cfg.norm_eps)
        self.dropout = cfg.dropout

    def forward(self, x, cos=None, sin=None):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if self.cfg.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = q.transpose(1, 2), k.transpose(1, 2)  # (B, H, T, D)
        if cos is not None:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return F.dropout(self.c_proj(y), self.dropout, self.training)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.kind = cfg.mlp
        h = cfg.mlp_hidden
        if cfg.mlp == "swiglu":
            self.gate = nn.Linear(cfg.n_embd, h, bias=cfg.bias)
            self.up = nn.Linear(cfg.n_embd, h, bias=cfg.bias)
        else:
            self.c_fc = nn.Linear(cfg.n_embd, h, bias=cfg.bias)
        self.c_proj = nn.Linear(h, cfg.n_embd, bias=cfg.bias)
        self.c_proj.RESIDUAL_PROJ = True
        self.dropout = cfg.dropout

    def forward(self, x):
        if self.kind == "swiglu":
            h = F.silu(self.gate(x)) * self.up(x)
        else:
            h = F.gelu(self.c_fc(x), approximate="tanh")  # = HF "gelu_new"
        return F.dropout(self.c_proj(h), self.dropout, self.training)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln_1 = make_norm(cfg, cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = make_norm(cfg, cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x, cos=None, sin=None):
        x = x + self.attn(self.ln_1(x), cos, sin)
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=make_norm(cfg, cfg.n_embd),
        ))
        if cfg.pos_emb == "learned":
            self.transformer["wpe"] = nn.Embedding(cfg.block_size, cfg.n_embd)
        else:
            cos, sin = rope_cache(cfg.head_dim, cfg.block_size, cfg.rope_theta)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.transformer.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # GPT-2 init: N(0, 0.02); residual projections scaled by 1/sqrt(2 * n_layer).
        if isinstance(m, nn.Linear):
            std = 0.02 / math.sqrt(2 * self.cfg.n_layer) if getattr(m, "RESIDUAL_PROJ", False) else 0.02
            nn.init.normal_(m.weight, 0.0, std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wte.weight.numel()
            if "wpe" in self.transformer:
                n -= self.transformer.wpe.weight.numel()
        return n

    def flops_per_token(self, seq_len: int) -> float:
        """Training FLOPs/token (fwd+bwd): 6N plus attention 12*L*d*T (PaLM appendix B)."""
        n = self.num_params() - (self.transformer.wpe.weight.numel() if "wpe" in self.transformer else 0)
        return 6 * n + 12 * self.cfg.n_layer * self.cfg.n_embd * seq_len

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence {T} > block_size {self.cfg.block_size}"
        x = self.transformer.wte(idx)
        cos = sin = None
        if "wpe" in self.transformer:
            x = x + self.transformer.wpe(torch.arange(T, device=idx.device))
        else:
            cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        x = self.drop(x)
        for block in self.transformer.h:
            x = block(x, cos, sin)
        logits = self.lm_head(self.transformer.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size:])
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)
        return idx
