"""Tokenizers behind one tiny interface: encode / decode / eot_id / vocab_size / name.

- get_tokenizer("gpt2")  -> tiktoken GPT-2 BPE (50257 tokens, eot = 50256)
- get_tokenizer(DIR)     -> a byte-level BPE trained by `train_bpe` (DIR/tokenizer.json)
"""
from __future__ import annotations

import json
import os
from typing import Iterable, Iterator

EOT = "<|endoftext|>"


class TiktokenGPT2:
    name = "gpt2"

    def __init__(self):
        import tiktoken
        self._enc = tiktoken.get_encoding("gpt2")
        self.eot_id = self._enc.eot_token
        self.vocab_size = self._enc.n_vocab

    def encode(self, text: str) -> list[int]:
        return self._enc.encode_ordinary(text)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return self._enc.encode_ordinary_batch(texts)

    def decode(self, ids: Iterable[int]) -> str:
        return self._enc.decode(list(ids))


class HFBPETokenizer:
    def __init__(self, path: str):
        from tokenizers import Tokenizer
        file = os.path.join(path, "tokenizer.json") if os.path.isdir(path) else path
        self.name = os.path.abspath(os.path.dirname(file))
        self._tok = Tokenizer.from_file(file)
        # A literal "<|endoftext|>" inside web text must stay ordinary bytes, never the real EOT
        # id, or such pages get fake document breaks. EOT is always inserted by id instead.
        self._tok.encode_special_tokens = True
        self.eot_id = self._tok.token_to_id(EOT)
        assert self.eot_id is not None, f"{file} has no {EOT} token"
        self.vocab_size = self._tok.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self._tok.encode_batch(texts, add_special_tokens=False)]

    def decode(self, ids: Iterable[int]) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=False)


def get_tokenizer(name_or_path: str):
    if name_or_path == "gpt2":
        return TiktokenGPT2()
    return HFBPETokenizer(name_or_path)


def train_bpe(texts: Iterator[str], vocab_size: int, out_dir: str, min_frequency: int = 2) -> str:
    """Train a GPT-2-style byte-level BPE (GPT-2 pre-tokenizer regex, no unknown tokens,
    lossless round-trip). Writes out_dir/tokenizer.json and returns its path."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    assert vocab_size < 2 ** 16, "token ids must fit in uint16 shards"
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "tokenizer.json")
    tok.save(path)
    with open(os.path.join(out_dir, "tokenizer_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"vocab_size": tok.get_vocab_size(), "eot": EOT, "eot_id": tok.token_to_id(EOT)}, f, indent=2)
    return path
