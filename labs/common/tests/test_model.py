import itertools

import pytest
import torch

from mygpt.config import ModelConfig, load_config
from mygpt.model import GPT, apply_rope, rope_cache
from mygpt.muon import Muon, newton_schulz

COMBOS = list(itertools.product(["learned", "rope"], ["layernorm", "rmsnorm"], ["gelu", "swiglu"],
                                [False, True], [False, True], [False, True]))


def tiny(**kw) -> ModelConfig:
    base = dict(vocab_size=97, n_layer=2, n_head=2, n_embd=32, block_size=16)
    return ModelConfig(**{**base, **kw})


@pytest.mark.parametrize("pos,norm,mlp,qk,bias,tie", COMBOS)
def test_forward_backward_every_combination(pos, norm, mlp, qk, bias, tie):
    torch.manual_seed(0)
    cfg = tiny(pos_emb=pos, norm=norm, mlp=mlp, qk_norm=qk, bias=bias, tie_embeddings=tie)
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (3, 16))
    y = torch.randint(0, cfg.vocab_size, (3, 16))
    logits, loss = model(x, y)
    assert logits.shape == (3, 16, cfg.vocab_size)
    assert torch.isfinite(loss) and abs(loss.item() - torch.log(torch.tensor(97.0)).item()) < 0.5
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert (model.lm_head.weight is model.transformer.wte.weight) == tie
    has_bias = any(n.endswith(".bias") for n, _ in model.named_parameters())
    assert has_bias == bias


def test_swiglu_hidden_and_causality():
    assert tiny(mlp="swiglu", n_embd=1024, n_head=16).mlp_hidden == 2752
    assert tiny(mlp="swiglu", n_embd=768, n_head=12).mlp_hidden == 2048
    torch.manual_seed(0)
    model = GPT(tiny(pos_emb="rope")).eval()
    x = torch.randint(0, 97, (1, 16))
    y = x.clone()
    y[0, 10:] = (y[0, 10:] + 1) % 97  # change the future only
    a, _ = model(x)
    b, _ = model(y)
    assert torch.allclose(a[0, :10], b[0, :10], atol=1e-6)


def test_rope_is_relative():
    cos, sin = rope_cache(8, 32, 10000.0)
    q, k = torch.randn(1, 1, 1, 8), torch.randn(1, 1, 1, 8)
    s1 = (apply_rope(q, cos[3:4], sin[3:4]) * apply_rope(k, cos[1:2], sin[1:2])).sum()
    s2 = (apply_rope(q, cos[12:13], sin[12:13]) * apply_rope(k, cos[10:11], sin[10:11])).sum()
    assert torch.allclose(s1, s2, atol=1e-5)


def test_config_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model:\n  n_layer: 4\ntrain:\n  train_tokens: 10e9\n  lr: 1e-3\n", encoding="utf-8")
    cfg = load_config(str(p), ["train.lr=6e-4", "model.qk_norm=true", "train.compile=false"])
    assert cfg.model.n_layer == 4 and cfg.model.qk_norm is True
    assert cfg.train.lr == 6e-4 and cfg.train.train_tokens == 10_000_000_000 and cfg.train.compile is False
    with pytest.raises(KeyError):
        load_config(str(p), ["train.not_a_field=1"])


def test_muon_step_runs_and_orthogonalizes():
    torch.manual_seed(0)
    G = torch.randn(64, 32)
    O = newton_schulz(G)
    s = torch.linalg.svdvals(O)
    assert s.max() < 1.3 and s.min() > 0.5  # NS5 lands singular values near 1 (not exactly)
    w = torch.nn.Parameter(torch.randn(32, 16))
    opt = Muon([w], lr=0.02, weight_decay=0.01)
    before = w.detach().clone()
    for _ in range(3):
        (w ** 2).sum().backward()
        opt.step()
        opt.zero_grad()
    assert not torch.equal(before, w) and torch.isfinite(w).all()
    assert "momentum_buffer" in opt.state[w]
