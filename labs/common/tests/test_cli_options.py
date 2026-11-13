"""export_hf --push visibility, mygpt.evaluate --batch / --eval-tokens all, W&B alerts,
decontam WinoGrande/LAMBADA, [eval] extra. CPU only, no network (Hub and datasets mocked)."""
import os
import sys
import types

import pytest
import torch

import mygpt.train as train_mod
from mygpt import evaluate as ev
from mygpt.data import make_synthetic_shards, read_meta
from mygpt.export_hf import main as export_main, push_to_hub
from mygpt.model import GPT

from conftest import ROOT

SMOKE = os.path.join(ROOT, "configs", "smoke_cpu.yaml")


# ------------------------------------------------------------------ export_hf --push
class FakeApi:
    def __init__(self, existing: dict | None = None):
        self.repos = dict(existing or {})   # repo_id -> private
        self.created, self.uploaded = [], []

    def create_repo(self, repo_id, repo_type="model", private=None, exist_ok=False):
        self.created.append((repo_id, repo_type, private))
        self.repos.setdefault(repo_id, bool(private))

    def upload_folder(self, folder_path, repo_id, repo_type="model"):
        self.uploaded.append((folder_path, repo_id, repo_type))

    def repo_info(self, repo_id, repo_type="model"):
        return types.SimpleNamespace(private=self.repos[repo_id])


def test_push_private_by_default(tmp_path, capsys):
    api = FakeApi()
    res = push_to_hub(str(tmp_path), "me/m", api=api)
    assert api.created == [("me/m", "model", True)] and api.uploaded == [(str(tmp_path), "me/m", "model")]
    assert res == {"repo_id": "me/m", "private": True}
    assert "hf repos settings me/m --public" in capsys.readouterr().out


def test_push_public_and_existing_public_warns(tmp_path, capsys):
    api = FakeApi()
    assert push_to_hub(str(tmp_path), "me/pub", private=False, api=api)["private"] is False
    assert api.created[-1][2] is False
    api = FakeApi(existing={"me/old": False})
    push_to_hub(str(tmp_path), "me/old", private=True, api=api)
    assert "WARNING" in capsys.readouterr().out


def _smoke_ckpt(tmp_path, steps=1):
    out = str(tmp_path / "run")
    data = str(tmp_path / "data")
    make_synthetic_shards(data)
    train_mod.main(["--config", SMOKE, f"train.out_dir={out}", f"train.data_dir={data}",
                    f"train.train_tokens={steps * 4096}", "train.eval_every=0"])
    return out, data


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    return _smoke_ckpt(tmp_path_factory.mktemp("cli"))


def test_export_cli_passes_private_flag(smoke_run, tmp_path, monkeypatch):
    calls = []
    import mygpt.export_hf as ex
    monkeypatch.setattr(ex, "push_to_hub", lambda folder, repo, private=True: calls.append((repo, private)) or {})
    ex.main(["--ckpt", smoke_run[0], "--out", str(tmp_path / "a"), "--push", "me/a"])
    ex.main(["--ckpt", smoke_run[0], "--out", str(tmp_path / "b"), "--push", "me/b", "--no-private"])
    export_main(["--ckpt", smoke_run[0], "--out", str(tmp_path / "c")])   # no push: no call
    assert calls == [("me/a", True), ("me/b", False)]


# ------------------------------------------------------------------ evaluate
def test_full_window_tokens_and_parse():
    assert ev.full_window_tokens([10 * 64 * 4 + 1, 64 * 4], seq_len=4) == 10 * 64 * 4
    assert ev.parse_eval_tokens("all") == ev.parse_eval_tokens("-1") == -1
    assert ev.parse_eval_tokens("1e6") == 1_000_000
    with pytest.raises(Exception):
        ev.parse_eval_tokens("0")


def test_evaluate_batch_and_whole_split(smoke_run):
    run, data = smoke_run
    T = 64  # smoke_cpu seq_len
    from mygpt.config import load_config
    assert load_config(SMOKE).train.seq_len == T
    a = ev.main(["--ckpt", run, "--data-dir", data, "--eval-tokens", str(64 * T * 2), "--batch", "16"])
    b = ev.main(["--ckpt", run, "--data-dir", data, "--eval-tokens", str(64 * T * 2), "--batch", "64"])
    assert (a["batch"], b["batch"]) == (16, 64)
    assert a["eval_tokens"] == b["eval_tokens"] == 64 * T * 2
    assert abs(a["val_loss"] - b["val_loss"]) < 1e-4      # same windows, different micro-batch
    whole = ev.main(["--ckpt", run, "--data-dir", data, "--eval-tokens", "all", "--batch", "32"])
    n_val = read_meta(data)["val_tokens"]
    assert whole["eval_tokens"] == ev.full_window_tokens([n_val], T, 64)
    assert ev.main(["--ckpt", run, "--data-dir", data, "--eval-tokens", "-1", "--batch", "16"])["eval_tokens"] \
        == whole["eval_tokens"]


# ------------------------------------------------------------------ W&B alerts
class FakeRun:
    def __init__(self):
        self.alerts, self.finished = [], False

    def log(self, rec, step=None):
        pass

    def alert(self, title, text, level=None):
        self.alerts.append((title, level))

    def finish(self):
        self.finished = True


@pytest.fixture
def fake_wandb(monkeypatch):
    run = FakeRun()
    mod = types.SimpleNamespace(init=lambda **kw: run,
                                AlertLevel=types.SimpleNamespace(INFO="INFO", WARN="WARN", ERROR="ERROR"))
    monkeypatch.setitem(sys.modules, "wandb", mod)
    return run


def test_wandb_alert_on_finish(tmp_path, fake_wandb):
    make_synthetic_shards(str(tmp_path / "d"))
    train_mod.main(["--config", SMOKE, f"train.out_dir={tmp_path / 'r'}", f"train.data_dir={tmp_path / 'd'}",
                    "train.train_tokens=8192", "train.eval_every=0", "train.wandb_project=p"])
    assert fake_wandb.finished and len(fake_wandb.alerts) == 1
    assert "finished" in fake_wandb.alerts[0][0] and fake_wandb.alerts[0][1] == "INFO"


def test_wandb_alert_off(tmp_path, fake_wandb):
    make_synthetic_shards(str(tmp_path / "d"))
    train_mod.main(["--config", SMOKE, f"train.out_dir={tmp_path / 'r'}", f"train.data_dir={tmp_path / 'd'}",
                    "train.train_tokens=8192", "train.eval_every=0", "train.wandb_project=p",
                    "train.wandb_alerts=false"])
    assert fake_wandb.alerts == []


def test_wandb_alert_on_nan(tmp_path, fake_wandb, monkeypatch):
    class Nan(GPT):
        def forward(self, idx, targets=None):
            logits, loss = super().forward(idx, targets)
            return logits, (loss * float("nan") if loss is not None else loss)

    monkeypatch.setattr(train_mod, "GPT", Nan)
    make_synthetic_shards(str(tmp_path / "d"))
    with pytest.raises(SystemExit):
        train_mod.main(["--config", SMOKE, f"train.out_dir={tmp_path / 'r'}", f"train.data_dir={tmp_path / 'd'}",
                        "train.eval_every=0", "train.wandb_project=p"])
    assert len(fake_wandb.alerts) == 1
    assert "non-finite" in fake_wandb.alerts[0][0] and fake_wandb.alerts[0][1] == "ERROR"


# ------------------------------------------------------------------ decontam + extras
def test_decontam_winogrande_lambada(monkeypatch):
    import datasets
    import decontam

    def fake_load(path, name=None, split=None):
        if path == "allenai/winogrande":
            return [{"sentence": "The cup fell because _ was heavy.", "option1": "it", "option2": "the shelf",
                     "answer": "2"}]
        if path == "EleutherAI/lambada_openai":
            return [{"text": "a whole passage ending in word"}]
        raise AssertionError(path)

    monkeypatch.setattr(datasets, "load_dataset", fake_load)
    out = decontam.load_benchmarks(["winogrande", "lambada"])
    assert out == {"winogrande": ["The cup fell because the shelf was heavy."],
                   "lambada_openai": ["a whole passage ending in word"]}
    with pytest.raises(ValueError, match="known:"):
        decontam.load_benchmarks(["mmlu"])


def test_eval_extra_pulls_hf_backend():
    import tomllib
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
        extras = tomllib.load(f)["project"]["optional-dependencies"]
    assert extras["eval"] == ["lm-eval[hf]>=0.4.13,<0.5"]
