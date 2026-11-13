"""Check a model folder before you upload it to the Hugging Face Hub (lesson 11.1).

    python tools/check_release.py release/my-gpt-350m
    python tools/check_release.py release/my-gpt-350m --load      # also load it and generate a few tokens

Fails (exit 1) on anything that should not be published or that will break loading:
pickle files (*.bin, *.pt, *.pth, *.pkl, training_args.bin), trainer checkpoint-* folders,
optimizer/scheduler state, secrets-looking files (.env, *token*), a missing model.safetensors,
tokenizer or config, and a README.md without the metadata the Hub needs (license, library_name).
Warns about missing model-card sections and leftover placeholders like <you> or ____.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

PICKLE = (".bin", ".pt", ".pth", ".pkl", ".pickle", ".ckpt")
SECRETS = re.compile(r"(^\.env$|secret|token(?!izer)|credential|\.pem$|id_rsa)", re.I)
REQUIRED = ("config.json", "tokenizer.json", "tokenizer_config.json", "README.md")
SECTIONS = ("Training data", "Evaluation", "Limitations", "Intended use", "License", "Compute")
PLACEHOLDER = re.compile(r"<you>|your-username|____|<fill|TODO", re.I)


def front_matter(text: str) -> dict:
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end < 0:
        return {}
    import yaml
    return yaml.safe_load(text[3:end]) or {}


def check(folder: str) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    files = []
    for root, dirs, names in os.walk(folder):
        rel_root = os.path.relpath(root, folder)
        for d in dirs:
            if d.startswith("checkpoint-") or d in (".git", "runs", "wandb"):
                errors.append(f"folder {os.path.join(rel_root, d)} must not be published (trainer state / logs)")
        for n in names:
            files.append(os.path.normpath(os.path.join(rel_root, n)))
    total = 0
    for f in files:
        name = os.path.basename(f)
        size = os.path.getsize(os.path.join(folder, f))
        total += size
        if name.endswith(PICKLE):
            errors.append(f"{f}: pickle-based file (can run code when loaded); publish safetensors only")
        if name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"):
            errors.append(f"{f}: trainer state, not part of a model release")
        if SECRETS.search(name):
            errors.append(f"{f}: file name looks like a secret; remove it")
    if not any(os.path.basename(f).endswith(".safetensors") for f in files):
        errors.append("no *.safetensors weights")
    for r in REQUIRED:
        if r not in files:
            errors.append(f"missing {r}")
    readme = os.path.join(folder, "README.md")
    if os.path.exists(readme):
        text = open(readme, encoding="utf-8").read()
        meta = front_matter(text)
        for key in ("license", "library_name", "datasets", "language", "pipeline_tag"):
            if key not in meta:
                (errors if key in ("license", "library_name") else warnings).append(f"README.md metadata has no `{key}`")
        for s in SECTIONS:
            if s.lower() not in text.lower():
                warnings.append(f"README.md has no '{s}' section")
        for m in sorted(set(PLACEHOLDER.findall(text))):
            warnings.append(f"README.md still contains placeholder {m!r}")
        if re.search(r"hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}", text):
            errors.append("README.md contains something that looks like an API token")
    cfg = os.path.join(folder, "config.json")
    if os.path.exists(cfg):
        c = json.load(open(cfg, encoding="utf-8"))
        print(f"architecture: {c.get('architectures')}  dtype: {c.get('dtype') or c.get('torch_dtype')}")
    print(f"{len(files)} files, {total / 1e9:.2f} GB")
    return errors, warnings


def try_load(folder: str) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(folder)
    model = AutoModelForCausalLM.from_pretrained(folder).eval()
    ids = tok("The three states of matter are", return_tensors="pt")["input_ids"]
    out = model.generate(ids, max_new_tokens=12, do_sample=False, pad_token_id=tok.eos_token_id)
    print("generation:", repr(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)))
    print("chat template:", "yes" if tok.chat_template else "no")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder")
    p.add_argument("--load", action="store_true", help="also load with transformers and generate")
    a = p.parse_args(argv)
    errors, warnings = check(a.folder)
    for w in warnings:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    if not errors and a.load:
        try_load(a.folder)
    print("RELEASE OK" if not errors else f"NOT READY: {len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
