import pytest
import torch

from mygpt.config import ModelConfig
from mygpt.export_hf import convert, export, export_kind, max_logit_diff
from mygpt.model import GPT

GPT2 = dict(pos_emb="learned", norm="layernorm", mlp="gelu", qk_norm=False, bias=True)
LLAMA = dict(pos_emb="rope", norm="rmsnorm", mlp="swiglu", qk_norm=False, bias=False, rope_theta=10000.0)
QWEN3 = {**LLAMA, "qk_norm": True, "rope_theta": 500000.0}


def trained_like(cfg: ModelConfig) -> GPT:
    """Random init, then perturb every parameter so norms/biases are not trivially 1/0."""
    torch.manual_seed(0)
    model = GPT(cfg)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return model


@pytest.mark.parametrize("arch,kind", [(GPT2, "gpt2"), (LLAMA, "llama"), (QWEN3, "qwen3")])
@pytest.mark.parametrize("tie", [True, False])
def test_export_logits_match(arch, kind, tie, tiny_tokenizer, tmp_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = ModelConfig(vocab_size=448, n_layer=2, n_head=4, n_embd=64, block_size=32, tie_embeddings=tie, **arch)
    assert export_kind(cfg) == kind
    model = trained_like(cfg)
    info = export(model, cfg, str(tmp_path), tokenizer=tiny_tokenizer)
    assert info["max_abs_logit_diff_fp32"] < 1e-3
    expected = {"gpt2": "GPT2LMHeadModel", "llama": "LlamaForCausalLM", "qwen3": "Qwen3ForCausalLM"}[kind]
    hf = AutoModelForCausalLM.from_pretrained(str(tmp_path), dtype=torch.float32).eval()
    assert type(hf).__name__ == expected
    assert max_logit_diff(model, hf, cfg, seq=32, seed=1) < 1e-3   # reloaded from disk, full context
    tok = AutoTokenizer.from_pretrained(str(tmp_path))
    text = "the loss goes down"
    assert tok.decode(tok(text)["input_ids"]) == text
    chat = tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True)
    assert chat == "<|user|>\nhi\n<|assistant|>\n"


@pytest.mark.parametrize("bad", [dict(GPT2, qk_norm=True), dict(LLAMA, bias=True), dict(LLAMA, norm="layernorm"),
                                 dict(GPT2, pos_emb="rope")])
def test_ablation_only_combination_refused(bad):
    cfg = ModelConfig(vocab_size=64, n_layer=1, n_head=2, n_embd=16, block_size=8, **bad)
    with pytest.raises(ValueError, match="ablation-only combination"):
        convert(GPT(cfg), cfg)
