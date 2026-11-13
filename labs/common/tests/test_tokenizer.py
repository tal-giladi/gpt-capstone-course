import pytest

from mygpt.tokenizer import get_tokenizer

from conftest import fake_texts


def test_trained_bpe_round_trip(tiny_tokenizer):
    tok = get_tokenizer(tiny_tokenizer)
    assert tok.vocab_size <= 400 and tok.eot_id is not None
    for text in fake_texts(20, seed=7) + ["  leading spaces\nnew\tlines ünïcödé 日本語 🙂", ""]:
        assert tok.decode(tok.encode(text)) == text
    # training text compresses: fewer tokens than bytes
    sample = fake_texts(1, seed=3)[0]
    assert len(tok.encode(sample)) < len(sample.encode()) / 2
    assert tok.encode_batch(["a b", "c"]) == [tok.encode("a b"), tok.encode("c")]
    assert "<|endoftext|>" in tok.decode([tok.eot_id])
    # a literal EOT string inside web text is ordinary text, not a document break
    literal = "x <|endoftext|> y"
    assert tok.eot_id not in tok.encode(literal)
    assert tok.eot_id not in tok.encode_batch([literal])[0]
    assert tok.decode(tok.encode(literal)) == literal


def test_gpt2_tiktoken():
    try:
        tok = get_tokenizer("gpt2")
    except Exception as e:  # first use downloads the BPE file
        pytest.skip(f"tiktoken gpt2 encoding not available offline: {e}")
    assert tok.eot_id == 50256 and tok.vocab_size == 50257
    assert tok.encode("Hello world") == [15496, 995]
    assert tok.decode(tok.encode("round trip ✓")) == "round trip ✓"
