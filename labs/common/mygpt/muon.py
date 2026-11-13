"""Muon: momentum + Newton-Schulz orthogonalization for 2D hidden weight matrices.

Use Muon only for the matrices inside transformer blocks; embeddings, lm_head, norms and
biases go to AdamW (see `train.build_optimizer`).

DDP: gradients are already all-reduced by DDP, so every rank holds identical momentum. To
avoid every rank paying for every Newton-Schulz, parameter i is orthogonalized by rank
i % world_size and the result is broadcast. Momentum buffers stay identical on all ranks,
so rank 0's optimizer state is a complete checkpoint.
"""
from __future__ import annotations

import torch
import torch.distributed as dist


@torch.no_grad()
def newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximately map G to the nearest semi-orthogonal matrix (quintic iteration,
    coefficients from Keller Jordan's Muon). Runs in bf16 on GPU, fp32 on CPU."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.bfloat16 if G.is_cuda else torch.float32)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, nesterov: bool = True,
                 weight_decay: float = 0.0, ns_steps: int = 5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                assert p.ndim == 2, "Muon is for 2D weight matrices only"

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if distributed else 0
        world = dist.get_world_size() if distributed else 1
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                owner = idx % world
                idx += 1
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.lerp_(p.grad, 1 - group["momentum"])
                g = p.grad.lerp(buf, group["momentum"]) if group["nesterov"] else buf
                if rank == owner:
                    update = newton_schulz(g, group["ns_steps"])
                else:
                    update = torch.empty_like(p)
                if world > 1:
                    dist.broadcast(update, src=owner)
                # Scale so the update RMS is roughly independent of the matrix aspect ratio.
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"] * scale)
        return loss
