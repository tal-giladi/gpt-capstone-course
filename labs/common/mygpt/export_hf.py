"""Export a mygpt checkpoint to Hugging Face format.

  python -m mygpt.export_hf --ckpt runs/main/ckpt_00038000.pt --out hf/my-gpt-350m [--tokenizer DIR]

Export rule (anything else is refused as an "ablation-only combination"):
  learned + layernorm + gelu + bias + no qk_norm  -> GPT2LMHeadModel
  rope + rmsnorm + swiglu + no bias               -> LlamaForCausalLM
  ... same with qk_norm                           -> Qwen3ForCausalLM

The exported model is checked against the native one on a random batch (fp32 logits, max abs
diff < 1e-3) and the tokenizer is written next to it so AutoTokenizer.from_pretrained works.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from .config import ModelConfig

EOT = "<|endoftext|>"

# Plain-text chat template: role tags are ordinary text, so no new tokens / no embedding
# resize. {% generation %} marks assistant tokens for TRL's assistant_only_loss.
CHAT_TEMPLATE = (
    "{%- for message in messages -%}"
    "{%- if message['role'] == 'assistant' -%}"
    "{{ '<|assistant|>\n' }}{% generation %}{{ message['content'] + eos_token }}{% endgeneration %}{{ '\n' }}"
    "{%- else -%}"
    "{{ '<|' + message['role'] + '|>\n' + message['content'] + '\n' }}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{ '<|assistant|>\n' }}{%- endif -%}"
)


def export_kind(m: ModelConfig) -> str:
    if m.pos_emb == "learned" and m.norm == "layernorm" and m.mlp == "gelu" and m.bias and not m.qk_norm:
        return "gpt2"
    if m.pos_emb == "rope" and m.norm == "rmsnorm" and m.mlp == "swiglu" and not m.bias:
        return "qwen3" if m.qk_norm else "llama"
    raise ValueError(
        f"ablation-only combination (pos_emb={m.pos_emb}, norm={m.norm}, mlp={m.mlp}, bias={m.bias}, "
        f"qk_norm={m.qk_norm}): no standard HF architecture matches. Exportable: all-GPT-2 settings, or "
        "rope+rmsnorm+swiglu+bias=false (+qk_norm for Qwen3).")


def _gpt2(model, m: ModelConfig, eot_id: int):
    from transformers import GPT2Config, GPT2LMHeadModel
    hcfg = GPT2Config(vocab_size=m.vocab_size, n_positions=m.block_size, n_embd=m.n_embd, n_layer=m.n_layer,
                      n_head=m.n_head, n_inner=m.mlp_hidden, activation_function="gelu_new",
                      resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, layer_norm_epsilon=m.norm_eps,
                      tie_word_embeddings=m.tie_embeddings, bos_token_id=eot_id, eos_token_id=eot_id)
    hf = GPT2LMHeadModel(hcfg)
    # HF GPT-2 uses Conv1D (weight stored as in x out) where we use nn.Linear (out x in): transpose.
    conv1d = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")
    sd = {k: (v.t().contiguous() if k.endswith(conv1d) else v) for k, v in model.state_dict().items()}
    return hf, sd


def _llama_like(model, m: ModelConfig, eot_id: int, kind: str):
    common = dict(vocab_size=m.vocab_size, hidden_size=m.n_embd, intermediate_size=m.mlp_hidden,
                  num_hidden_layers=m.n_layer, num_attention_heads=m.n_head, num_key_value_heads=m.n_head,
                  head_dim=m.head_dim, max_position_embeddings=m.block_size, rms_norm_eps=m.norm_eps,
                  hidden_act="silu", attention_bias=False, tie_word_embeddings=m.tie_embeddings,
                  bos_token_id=eot_id, eos_token_id=eot_id, rope_theta=m.rope_theta)
    if kind == "qwen3":
        from transformers import Qwen3Config as C, Qwen3ForCausalLM as M
        common.update(use_sliding_window=False)
    else:
        from transformers import LlamaConfig as C, LlamaForCausalLM as M
        common.update(mlp_bias=False)
    hcfg = C(**common)
    # transformers >= 5 keeps RoPE settings in `rope_parameters`; make sure our theta is used.
    if getattr(hcfg, "rope_parameters", None) is not None:
        hcfg.rope_parameters = {**hcfg.rope_parameters, "rope_type": "default", "rope_theta": m.rope_theta}
    hf = M(hcfg)
    sd, C_ = {}, m.n_embd
    src = model.state_dict()
    sd["model.embed_tokens.weight"] = src["transformer.wte.weight"]
    sd["model.norm.weight"] = src["transformer.ln_f.weight"]
    sd["lm_head.weight"] = src["lm_head.weight"]
    for i in range(m.n_layer):
        a, b = f"transformer.h.{i}.", f"model.layers.{i}."
        q, k, v = src[a + "attn.c_attn.weight"].split(C_, dim=0)
        sd.update({
            b + "input_layernorm.weight": src[a + "ln_1.weight"],
            b + "post_attention_layernorm.weight": src[a + "ln_2.weight"],
            b + "self_attn.q_proj.weight": q, b + "self_attn.k_proj.weight": k, b + "self_attn.v_proj.weight": v,
            b + "self_attn.o_proj.weight": src[a + "attn.c_proj.weight"],
            b + "mlp.gate_proj.weight": src[a + "mlp.gate.weight"],
            b + "mlp.up_proj.weight": src[a + "mlp.up.weight"],
            b + "mlp.down_proj.weight": src[a + "mlp.c_proj.weight"],
        })
        if m.qk_norm:
            sd[b + "self_attn.q_norm.weight"] = src[a + "attn.q_norm.weight"]
            sd[b + "self_attn.k_norm.weight"] = src[a + "attn.k_norm.weight"]
    return hf, sd


def convert(model, m: ModelConfig, eot_id: int = 0):
    """Native GPT -> HF model (fp32, eval mode) with weights copied."""
    kind = export_kind(m)
    model = model.float().eval()
    hf, sd = _gpt2(model, m, eot_id) if kind == "gpt2" else _llama_like(model, m, eot_id, kind)
    missing, unexpected = hf.load_state_dict({k: v.contiguous().clone() for k, v in sd.items()}, strict=False)
    missing = [k for k in missing if not k.endswith((".attn.bias", ".attn.masked_bias", "rotary_emb.inv_freq"))
               and not (k == "lm_head.weight" and m.tie_embeddings)]
    if missing or unexpected:
        raise RuntimeError(f"weight mapping mismatch: missing={missing} unexpected={unexpected}")
    if m.tie_embeddings:
        hf.tie_weights()
    return hf.float().eval(), kind


@torch.no_grad()
def max_logit_diff(model, hf, m: ModelConfig, batch: int = 2, seq: int | None = None, seed: int = 0) -> float:
    g = torch.Generator().manual_seed(seed)
    seq = seq or min(m.block_size, 64)
    ids = torch.randint(0, m.vocab_size, (batch, seq), generator=g)
    ours, _ = model.float().eval()(ids)
    theirs = hf(input_ids=ids).logits
    return (ours - theirs).abs().max().item()


def build_hf_tokenizer(name_or_path: str, max_len: int):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    if name_or_path == "gpt2":
        tok = AutoTokenizer.from_pretrained("openai-community/gpt2")  # needs network once
    else:
        file = os.path.join(name_or_path, "tokenizer.json") if os.path.isdir(name_or_path) else name_or_path
        tok = PreTrainedTokenizerFast(tokenizer_file=file, bos_token=EOT, eos_token=EOT)
    tok.pad_token = tok.eos_token
    tok.model_max_length = max_len
    tok.chat_template = CHAT_TEMPLATE
    return tok


def export(model, m: ModelConfig, out_dir: str, tokenizer: str | None, dtype: str = "fp32",
           tol: float = 1e-3, extra: dict | None = None) -> dict:
    tok = build_hf_tokenizer(tokenizer, m.block_size) if tokenizer else None
    eot_id = tok.eos_token_id if tok is not None else 0
    hf, kind = convert(model, m, eot_id)
    diff = max_logit_diff(model, hf, m)
    if not diff < tol:
        raise RuntimeError(f"exported {kind} logits differ from native by {diff:.3e} (tolerance {tol})")
    if tok is not None:
        hf.generation_config.bos_token_id = hf.generation_config.eos_token_id = eot_id
        hf.generation_config.pad_token_id = eot_id
    if dtype == "bf16":
        hf = hf.to(torch.bfloat16)
    os.makedirs(out_dir, exist_ok=True)
    hf.save_pretrained(out_dir)
    if tok is not None:
        tok.save_pretrained(out_dir)
    info = {"architecture": type(hf).__name__, "kind": kind, "max_abs_logit_diff_fp32": diff,
            "params": sum(p.numel() for p in model.parameters()), **(extra or {})}
    with open(os.path.join(out_dir, "mygpt_export.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, default=str)
    return info


def main(argv=None):
    p = argparse.ArgumentParser(description="Export a mygpt checkpoint to Hugging Face format.")
    p.add_argument("--ckpt", required=True, help="checkpoint file, or a run dir (uses the latest ckpt)")
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", default=None, help="gpt2 | DIR (default: tokenizer recorded in the shards' meta.json)")
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32", help="dtype of the saved weights")
    p.add_argument("--push", default=None, help="optional HF repo id to upload to, e.g. you/my-gpt-350m")
    args = p.parse_args(argv)

    from .checkpoint import latest_checkpoint, load_checkpoint
    from .config import config_from_dict
    from .model import GPT

    path = latest_checkpoint(args.ckpt) if os.path.isdir(args.ckpt) else args.ckpt
    if not path:
        raise SystemExit(f"no checkpoint found in {args.ckpt}")
    ck = load_checkpoint(path)
    cfg = config_from_dict(ck["config"])
    export_kind(cfg.model)  # fail fast before loading anything heavy
    model = GPT(cfg.model)
    model.load_state_dict(ck["model"])
    tokenizer = args.tokenizer
    if tokenizer is None:  # same lookup as training: gpt2, or tokenizer.json next to the shards
        from .data import resolve_tokenizer
        meta = ck.get("meta") or {}
        tokenizer = resolve_tokenizer(cfg.train.data_dir, meta) if os.path.isdir(cfg.train.data_dir) \
            else meta.get("tokenizer_resolved") or meta.get("tokenizer")
        if tokenizer not in (None, "gpt2", "synthetic") and not os.path.exists(tokenizer):
            raise SystemExit(f"tokenizer {tokenizer!r} not found on this box; pass --tokenizer DIR")
    if tokenizer in (None, "synthetic"):
        print("[export] no tokenizer recorded; exporting weights only (pass --tokenizer gpt2|DIR)")
        tokenizer = None
    info = export(model, cfg.model, args.out, tokenizer, args.dtype,
                  extra={"source_checkpoint": path, "step": ck["step"], "tokens": ck["step"] * cfg.train.total_batch_tokens})
    print(json.dumps(info, indent=2, default=str))
    if args.push:
        from huggingface_hub import HfApi
        HfApi().create_repo(args.push, repo_type="model", exist_ok=True)
        HfApi().upload_folder(folder_path=args.out, repo_id=args.push, repo_type="model")
        print(f"[export] uploaded to https://huggingface.co/{args.push}")
    return info


if __name__ == "__main__":
    main()
