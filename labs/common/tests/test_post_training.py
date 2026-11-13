"""sft.py / dpo.py / judge_winrate.py: fp32 master weights, dataset normalization, --resume,
--save-final-only, --dataset-revision, judge token usage/cost, Wilson CI and declined verdicts.
CPU only, tiny exported models, local JSONL data, no network (the judge API is mocked)."""
import json
import os
import re
from types import SimpleNamespace

import pytest
import torch

from conftest import SCRIPTS

pytest.importorskip("trl")

LLAMA = dict(pos_emb="rope", norm="rmsnorm", mlp="swiglu", qk_norm=False, bias=False, rope_theta=10000.0)
TINY = ["--batch-size", "2", "--grad-accum", "1", "--max-length", "64", "--set", "logging_steps=1",
        "--set", "save_steps=1", "--set", "save_total_limit=5", "--set", "dataloader_num_workers=0"]


@pytest.fixture(scope="module")
def hf_model(tiny_tokenizer, tmp_path_factory):
    from mygpt.config import ModelConfig
    from mygpt.export_hf import export
    from mygpt.model import GPT
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=448, n_layer=1, n_head=2, n_embd=32, block_size=64, tie_embeddings=True, **LLAMA)
    out = str(tmp_path_factory.mktemp("hf_tiny"))
    export(GPT(cfg), cfg, out, tokenizer=tiny_tokenizer)
    return out


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(path)


def sft_rows(n=8):
    return [{"messages": [{"role": "user", "content": f"what is {i} plus {i}"},
                          {"role": "assistant", "content": f"the answer is {2 * i}"}], "source": "x"} for i in range(n)]


def test_fp32_master_weights_literal():
    """Lessons 09.2 / 10.1 grep for this exact text."""
    for name in ("sft.py", "dpo.py"):
        src = open(os.path.join(SCRIPTS, name), encoding="utf-8").read()
        loads = re.findall(r"from_pretrained\(args\.model, dtype=[^,)]+", src)
        assert loads and all(m.endswith("dtype=torch.float32") for m in loads), (name, loads)
        assert "bf16=bf16" in src


def test_sft_fp32_save_final_only(hf_model, tmp_path):
    import sft
    data = write_jsonl(tmp_path / "sft.jsonl", sft_rows())
    out = str(tmp_path / "sft")
    trainer = sft.main(["--model", hf_model, "--dataset", data, "--out", out, "--epochs", "1",
                        "--save-final-only", "--set", "max_steps=2", *TINY])
    assert next(trainer.model.parameters()).dtype == torch.float32
    assert trainer.state.global_step == 2
    assert not [d for d in os.listdir(out) if d.startswith("checkpoint-")]
    assert os.path.exists(os.path.join(out, "model.safetensors"))


def test_sft_resume(hf_model, tmp_path, capsys):
    import sft
    data = write_jsonl(tmp_path / "sft.jsonl", sft_rows())
    out = str(tmp_path / "sft")
    common = ["--model", hf_model, "--dataset", data, "--out", out, "--epochs", "1", *TINY]
    sft.main([*common, "--set", "max_steps=2"])
    assert sorted(os.listdir(out)).count("checkpoint-2") == 1
    trainer = sft.main([*common, "--resume", "--set", "max_steps=3"])
    assert "resuming from" in capsys.readouterr().out and trainer.state.global_step == 3


def test_sft_packing_needs_flash_attention(hf_model, tmp_path):
    import sft
    with pytest.raises(SystemExit, match="flash_attention"):
        sft.main(["--model", hf_model, "--dataset", "x.jsonl", "--out", str(tmp_path), "--packing"])


@pytest.mark.parametrize("style", ["full_conversation", "answer_only", "plain"])
def test_dpo_normalizes_string_prompt(hf_model, tmp_path, style):
    import dpo
    rows = []
    for i in range(6):
        user = {"role": "user", "content": f"say {i}"}
        good, bad = {"role": "assistant", "content": f"{i}"}, {"role": "assistant", "content": "no"}
        if style == "full_conversation":   # H4 / argilla: prompt string + [user, assistant] lists
            rows.append({"prompt": user["content"], "chosen": [user, good], "rejected": [user, bad], "score": 1.0})
        elif style == "answer_only":       # prompt string + [assistant] lists
            rows.append({"prompt": user["content"], "chosen": [good], "rejected": [bad], "id": i})
        else:                              # trl-lib style, already fine
            rows.append({"chosen": [user, good], "rejected": [user, bad]})
    data = write_jsonl(tmp_path / "prefs.jsonl", rows)
    out = str(tmp_path / "dpo")
    trainer = dpo.main(["--model", hf_model, "--dataset", data, "--out", out, "--save-final-only",
                        "--set", "max_steps=2", *TINY])
    assert next(trainer.model.parameters()).dtype == torch.float32
    assert trainer.ref_model is None or next(trainer.ref_model.parameters()).dtype == torch.float32
    assert trainer.state.global_step == 2
    assert not [d for d in os.listdir(out) if d.startswith("checkpoint-")]


def test_dpo_resume(hf_model, tmp_path, capsys):
    import dpo
    user = {"role": "user", "content": "hi"}
    rows = [{"chosen": [user, {"role": "assistant", "content": f"hello {i}"}],
             "rejected": [user, {"role": "assistant", "content": "go away"}]} for i in range(6)]
    data = write_jsonl(tmp_path / "prefs.jsonl", rows)
    common = ["--model", hf_model, "--dataset", data, "--out", str(tmp_path / "dpo"), *TINY]
    dpo.main([*common, "--set", "max_steps=1"])
    trainer = dpo.main([*common, "--resume", "--set", "max_steps=2"])
    assert "resuming from" in capsys.readouterr().out and trainer.state.global_step == 2


def test_dataset_revision_passed_to_load_dataset(monkeypatch):
    import datasets

    import _common
    seen = {}
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: seen.update(args=a, kwargs=k) or "ds")
    assert _common.load_hf_dataset("org/name", None, "train", "abc123") == "ds"
    assert seen["kwargs"]["revision"] == "abc123" and seen["args"] == ("org/name", None)
    with pytest.raises(SystemExit, match="hub datasets"):
        _common.load_hf_dataset("local.jsonl", None, "train", "abc123")


def test_resume_checkpoint_helpers(tmp_path):
    import _common
    for step in (10, 500, 1000):
        os.makedirs(tmp_path / f"checkpoint-{step}")
    (tmp_path / "checkpoint-notes.txt").write_text("x")
    assert _common.resume_checkpoint(str(tmp_path), False) is None
    assert _common.resume_checkpoint(str(tmp_path), True).endswith("checkpoint-1000")   # numeric, not lexical
    assert len(_common.remove_checkpoints(str(tmp_path))) == 3
    assert _common.resume_checkpoint(str(tmp_path), True) is None
    assert os.listdir(tmp_path) == ["checkpoint-notes.txt"]


# ---- judge_winrate.py -------------------------------------------------------------------------

def judge_files(tmp_path, n):
    prompts = [f"question {i}?" for i in range(n)]
    (tmp_path / "p.txt").write_text("\n".join(prompts), encoding="utf-8")
    for name in ("a", "b"):
        write_jsonl(tmp_path / f"{name}.jsonl", [{"prompt": p, "answer": f"{name} answers {p}"} for p in prompts])
    return ["--answers-a", str(tmp_path / "a.jsonl"), "--answers-b", str(tmp_path / "b.jsonl"),
            "--prompts", str(tmp_path / "p.txt"), "--n", str(n), "--out", str(tmp_path / "out")]


def test_wilson_inside_unit_interval():
    import judge_winrate as jw
    for k, n in ((0, 3), (3, 3), (0.5, 2), (1, 1), (50, 100)):
        lo, hi = jw.wilson(k, n)
        assert 0.0 <= lo <= k / n <= hi <= 1.0
    lo, hi = jw.wilson(50, 100)
    assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3


def test_judge_mocked_anthropic_tokens_cost_declined(tmp_path, monkeypatch):
    """Fake Anthropic client: one prompt declined, one reply without a verdict, the rest B wins."""
    import sys

    import judge_winrate as jw
    calls = []

    class Messages:
        def create(self, **kw):
            text = kw["messages"][0]["content"]
            calls.append(kw)
            first_is_b = "[Answer A]\nb answers" in text
            stop, reply = "end_turn", ("[[A]]" if first_is_b else "[[B]]")
            if "question 0?" in text:
                stop, reply = "refusal", ""
            elif "question 1?" in text and not first_is_b:
                reply = "I cannot decide."
            return SimpleNamespace(stop_reason=stop, content=[SimpleNamespace(type="text", text=f"reason. {reply}")],
                                   usage=SimpleNamespace(input_tokens=100, output_tokens=40))

    fake = SimpleNamespace(Anthropic=lambda: SimpleNamespace(beta=SimpleNamespace(messages=Messages())))
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    s = jw.main(judge_files(tmp_path, 5))
    assert len(calls) == 10 and calls[0]["model"] == "claude-opus-5-5"
    assert s["judge_calls"] == 10 and s["judge_input_tokens"] == 1000 and s["judge_output_tokens"] == 400
    assert s["estimated_cost_usd"] == pytest.approx(1000 / 1e6 * 4.0 + 400 / 1e6 * 20.0)
    assert (s["invalid"], s["invalid_declined"], s["invalid_no_verdict"]) == (2, 1, 1)
    assert s["judge_calls_declined"] == 2 and s["judge_calls_no_verdict"] == 1
    assert s["n_judged"] == 3 and s["b_wins"] == 3 and s["b_win_rate"] == 1.0
    assert 0.0 <= s["ci95"][0] <= s["ci95"][1] == 1.0
    rows = {r["prompt"]: r for r in map(json.loads, open(tmp_path / "out" / "judgments.jsonl", encoding="utf-8"))}
    assert [rows[f"question {i}?"].get("invalid_reason") for i in range(3)] == ["declined", "no_verdict", None]
    assert rows["question 0?"]["judge_status"] == ["declined", "declined"]
    assert "judge_replies" in rows["question 1?"]


def test_judge_mocked_openai_unknown_price(tmp_path, monkeypatch):
    import sys

    import judge_winrate as jw

    def create(**kw):
        msg = SimpleNamespace(content="fine. [[TIE]]", refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3))

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kw: client))
    args = judge_files(tmp_path, 2) + ["--judge", "openai", "--judge-model", "local-model"]
    s = jw.main(args)
    assert s["ties"] == 2 and s["judge_input_tokens"] == 28 and s["estimated_cost_usd"] is None
    s = jw.main(args + ["--price-in", "1", "--price-out", "2"])
    assert s["estimated_cost_usd"] == pytest.approx(28e-6 + 12 * 2e-6)
