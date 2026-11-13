"""train.tracker: trackio adapter and validation (trackio is mocked; no network)."""
import sys
import types

import pytest

from mygpt.config import load_config
from mygpt.train import train


def _cfg(tmp_path, *over):
    return load_config("configs/smoke_cpu.yaml", [f"train.out_dir={tmp_path}", "train.train_tokens=4096",
                                                   "train.eval_every=0", *over])


def test_trackio_tracker_logs_and_finishes(tmp_path, monkeypatch):
    calls = {"init": [], "log": 0, "finish": 0}
    fake = types.ModuleType("trackio")
    fake.init = lambda **kw: calls["init"].append(kw)
    fake.log = lambda metrics, step=None: calls.__setitem__("log", calls["log"] + 1)
    fake.finish = lambda: calls.__setitem__("finish", calls["finish"] + 1)
    monkeypatch.setitem(sys.modules, "trackio", fake)
    train(_cfg(tmp_path, "train.tracker=trackio", "train.wandb_project=capstone"))
    assert calls["init"] and calls["init"][0]["project"] == "capstone"
    assert calls["log"] > 0 and calls["finish"] == 1


def test_tracker_none_and_invalid(tmp_path):
    train(_cfg(tmp_path / "a", "train.tracker=none", "train.wandb_project=capstone"))
    with pytest.raises(ValueError):
        train(_cfg(tmp_path / "b", "train.tracker=mlflow", "train.wandb_project=capstone"))
