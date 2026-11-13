"""SIGTERM / NaN handling in training, and the thin scripts' CLIs (no heavy deps needed)."""
import json
import os
import signal

import pytest

import mygpt.train as train_mod
from mygpt.checkpoint import list_checkpoints
from mygpt.data import DistributedTokenLoader
from mygpt.model import GPT

from conftest import ROOT, run_script

SMOKE = os.path.join(ROOT, "configs", "smoke_cpu.yaml")


def test_sigterm_saves_and_exits(tmp_path, monkeypatch):
    calls = {"n": 0}
    orig = DistributedTokenLoader.next_batch

    def next_batch(self):
        if self.ds.paths[0].endswith("train_000000.bin"):
            calls["n"] += 1
            if calls["n"] == 5:  # during step 3 (2 micro-batches per step)
                signal.raise_signal(signal.SIGTERM)
        return orig(self)

    monkeypatch.setattr(DistributedTokenLoader, "next_batch", next_batch)
    out = str(tmp_path / "s")
    res = train_mod.main(["--config", SMOKE, f"train.out_dir={out}"])
    assert res["stopped"] and res["exit_code"] == 0
    assert [os.path.basename(p) for p in list_checkpoints(out)] == ["ckpt_00000003.pt"]
    assert signal.getsignal(signal.SIGTERM) is not None  # handler restored


def test_nan_loss_stops_and_names_checkpoint(tmp_path, monkeypatch, capsys):
    class Exploding(GPT):
        calls = 0

        def forward(self, idx, targets=None):
            logits, loss = super().forward(idx, targets)
            if targets is not None and self.training:
                Exploding.calls += 1
                if Exploding.calls > 2 * 12:  # after step 12
                    loss = loss * float("nan")
            return logits, loss

    monkeypatch.setattr(train_mod, "GPT", Exploding)
    out = str(tmp_path / "n")
    with pytest.raises(SystemExit) as e:
        train_mod.main(["--config", SMOKE, f"train.out_dir={out}"])
    assert e.value.code == 1
    text = capsys.readouterr().out
    assert "non-finite loss" in text and "ckpt_00000010.pt" in text


@pytest.mark.parametrize("script", ["download_fineweb.py", "train_tokenizer.py", "tokenize_shards.py", "token_report.py",
                                    "decontam.py", "bench.py", "sft.py", "dpo.py", "chat.py", "judge_winrate.py"])
def test_script_help(script):
    res = run_script(script, "--help", timeout=60)
    assert "usage:" in res.stdout


def test_judge_dry_run(tmp_path):
    prompts = ["What is 2+2?", "Name a prime number.", "Say hi."]
    (tmp_path / "p.txt").write_text("\n".join(prompts), encoding="utf-8")
    for name, prefix in (("a", "A says"), ("b", "B says")):
        with open(tmp_path / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for p in prompts:
                f.write(json.dumps({"prompt": p, "answer": f"{prefix}: {p}"}) + "\n")
    out = str(tmp_path / "judge")
    res = run_script("judge_winrate.py", "--answers-a", str(tmp_path / "a.jsonl"), "--answers-b",
                     str(tmp_path / "b.jsonl"), "--prompts", str(tmp_path / "p.txt"), "--dry-run", "--out", out)
    assert "nothing sent" in res.stdout
    s = json.load(open(os.path.join(out, "summary.json"), encoding="utf-8"))
    assert s["judge"] == "dry-run" and s["n_judged"] + s["invalid"] == 3
    assert sum(1 for _ in open(os.path.join(out, "judgments.jsonl"), encoding="utf-8")) == 3
