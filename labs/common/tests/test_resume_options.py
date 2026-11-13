"""Resume-time options and bookkeeping: evaluate on a run dir, skip_batches, WSD branching with a
smaller train_tokens, per-optimizer weight decay, MFU FLOPs for untied embeddings, log.jsonl
mirroring, and the race-free HellaSwag download."""
import json
import os
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import torch

import mygpt.checkpoint as ckpt_mod
import mygpt.evaluate as eval_mod
import mygpt.train as train_mod
from mygpt.checkpoint import ckpt_path, list_checkpoints, load_checkpoint
from mygpt.config import ModelConfig, TrainConfig, load_config
from mygpt.model import GPT
from mygpt.muon import Muon
from mygpt.train import lr_mult, main

from conftest import ROOT

SMOKE = os.path.join(ROOT, "configs", "smoke_cpu.yaml")


def read_log(out_dir):
    with open(os.path.join(out_dir, "log.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def by_step(out_dir, key):
    return {r["step"]: r[key] for r in read_log(out_dir) if key in r}


def branch_dir(src_run, step, dst):
    """A new run dir holding one checkpoint of src_run plus its synthetic data."""
    os.makedirs(dst)
    shutil.copy(ckpt_path(src_run, step), dst)
    shutil.copytree(os.path.join(src_run, "synthetic_data"), os.path.join(dst, "synthetic_data"))
    return dst


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("smoke") / "a")
    main(["--config", SMOKE, f"train.out_dir={out}", "train.log_every=1", "train.keep_last=5"])
    return out


# ---------------------------------------------------------------- evaluate CLI
def test_evaluate_accepts_run_dir(smoke_run, capsys):
    data = os.path.join(smoke_run, "synthetic_data")
    out = eval_mod.main(["--ckpt", smoke_run, "--data-dir", data, "--eval-tokens", "2048"])
    assert out["ckpt"].endswith("ckpt_00000020.pt") and out["step"] == 20
    by_file = eval_mod.main(["--ckpt", out["ckpt"], "--data-dir", data, "--eval-tokens", "2048"])
    assert out["val_loss"] == pytest.approx(by_file["val_loss"])
    with pytest.raises(SystemExit):
        eval_mod.main(["--ckpt", data])  # dir without checkpoints


# ---------------------------------------------------------------- skip_batches
def test_skip_batches_advances_loader_once(smoke_run, tmp_path):
    b = branch_dir(smoke_run, 10, str(tmp_path / "b"))
    main(["--config", SMOKE, f"train.out_dir={b}", "train.log_every=1", "train.skip_batches=3"])
    skip = [r for r in read_log(b) if "skip_batches" in r]
    assert skip == [{"step": 10, "skip_batches": 3, "skipped_tokens": 3 * 1024}]
    accum, chunk = 2, 8 * 64  # smoke: 2 micro-batches of 8x64 per optimizer step
    pos_a = load_checkpoint(ckpt_path(smoke_run, 20))["loader"]["pos"]
    pos_b = load_checkpoint(ckpt_path(b, 20))["loader"]["pos"]
    assert pos_b - pos_a == 3 * accum * chunk
    # Same step count and LR schedule, only the data moved.
    assert sorted(by_step(b, "loss")) == list(range(11, 21))
    assert by_step(b, "lr") == {s: by_step(smoke_run, "lr")[s] for s in range(11, 21)}

    # Resuming again with the same override (still on the command line) must not skip again.
    main(["--config", SMOKE, f"train.out_dir={b}", "train.train_tokens=25600", "train.skip_batches=3"])
    assert len([r for r in read_log(b) if "skip_batches" in r]) == 1
    pos_25 = load_checkpoint(list_checkpoints(b)[-1])["loader"]["pos"]
    assert pos_25 - pos_b == 5 * accum * chunk


def test_skip_batches_ignored_on_fresh_run(tmp_path):
    out = str(tmp_path / "f")
    main(["--config", SMOKE, f"train.out_dir={out}", "train.train_tokens=2048", "train.skip_batches=4"])
    assert not [r for r in read_log(out) if "skip_batches" in r]


# ---------------------------------------------------------------- WSD branching
def test_wsd_branch_with_smaller_train_tokens(tmp_path):
    """Resume a copy of a long WSD run from a stable-phase checkpoint with a smaller
    train_tokens: the decay must start at the step derived from the new total and the branch
    must reproduce a run that was planned that short from the start."""
    wsd = ["train.schedule=wsd", "train.min_lr_ratio=0.0", "train.log_every=1", "train.keep_last=10",
           "train.eval_every=0"]
    long = str(tmp_path / "long")  # 40 steps, decay_frac 1/3 -> decay starts at step index 26
    main(["--config", SMOKE, f"train.out_dir={long}", "train.train_tokens=40960", "train.decay_frac=0.3333", *wsd])
    lr_long = by_step(long, "lr")
    assert lr_long[20] == lr_long[21] == pytest.approx(1e-2)  # step 20 is in the stable phase

    branch = branch_dir(long, 20, str(tmp_path / "branch"))  # 30 steps, decay over the last 10
    main(["--config", SMOKE, f"train.out_dir={branch}", "train.train_tokens=30720", "train.decay_frac=0.3333", *wsd])
    lr_b, loss_b = by_step(branch, "lr"), by_step(branch, "loss")
    assert sorted(lr_b) == list(range(21, 31))
    t = SimpleNamespace(schedule="wsd", min_lr_ratio=0.0, decay_frac=0.3333)
    for s in range(21, 31):
        assert lr_b[s] == pytest.approx(1e-2 * lr_mult(s - 1, 30, 2, t))
    assert lr_b[21] == pytest.approx(1e-2) and lr_b[30] == pytest.approx(1e-3)
    assert all(lr_b[s + 1] < lr_b[s] for s in range(21, 30))
    assert os.path.basename(list_checkpoints(branch)[-1]) == "ckpt_00000030.pt"

    short = str(tmp_path / "short")  # the same 30-step schedule, uninterrupted
    main(["--config", SMOKE, f"train.out_dir={short}", "train.train_tokens=30720", "train.decay_frac=0.3333", *wsd])
    lr_s, loss_s = by_step(short, "lr"), by_step(short, "loss")
    for s in range(21, 31):
        assert lr_b[s] == pytest.approx(lr_s[s])
        assert loss_b[s] == pytest.approx(loss_s[s], abs=1e-5)


# ---------------------------------------------------------------- weight decay
def _groups(**train_kw):
    torch.manual_seed(0)
    model = GPT(ModelConfig(vocab_size=97, n_layer=2, n_head=2, n_embd=32, block_size=16))
    t = TrainConfig(**train_kw)
    return build(model, t)


def build(model, t):
    opts = train_mod.build_optimizer(model, t, "cpu")
    adamw = [o for o in opts if isinstance(o, torch.optim.AdamW)][0]
    muon = [o for o in opts if isinstance(o, Muon)]
    return adamw, (muon[0] if muon else None)


def test_weight_decay_split_between_optimizers():
    adamw, muon = _groups(optimizer="muon", weight_decay=0.1)
    assert [g["weight_decay"] for g in adamw.param_groups] == [0.1, 0.0]
    assert [g["weight_decay"] for g in muon.param_groups] == [0.0]  # muon_weight_decay default
    adamw, muon = _groups(optimizer="muon", weight_decay=0.0, muon_weight_decay=0.05)
    assert [g["weight_decay"] for g in adamw.param_groups] == [0.0, 0.0]
    assert [g["weight_decay"] for g in muon.param_groups] == [0.05]
    adamw, muon = _groups(optimizer="adamw", weight_decay=0.1, muon_weight_decay=0.05)
    assert muon is None and [g["weight_decay"] for g in adamw.param_groups] == [0.1, 0.0]


def test_configs_weight_decay():
    main350 = load_config(os.path.join(ROOT, "configs", "main_350m.yaml")).train
    assert main350.optimizer == "muon" and main350.weight_decay == 0.1 and main350.muon_weight_decay == 0.0


# ---------------------------------------------------------------- MFU FLOPs
def test_flops_per_token_excludes_untied_input_embedding():
    base = load_config(os.path.join(ROOT, "configs", "ablation_base.yaml")).model
    with torch.device("meta"):
        tied = GPT(base)
        untied = GPT(ModelConfig(**{**base.__dict__, "tie_embeddings": False}))
    assert untied.num_params() - tied.num_params() == 32768 * 768
    # Untied adds a separate lm_head but the input table is a lookup: same matmul FLOPs.
    assert untied.flops_per_token(1024) == tied.flops_per_token(1024)
    assert tied.flops_per_token(1024) / 1e6 == pytest.approx(775, abs=1)
    n_tied = tied.num_params() - tied.transformer.wpe.weight.numel()  # wte stays: it is the lm_head
    assert tied.flops_per_token(1024) == 6 * n_tied + 12 * 12 * 768 * 1024


# ---------------------------------------------------------------- rclone mirror
def test_log_pushed_with_every_checkpoint(tmp_path, monkeypatch):
    pushed = []

    class Done:
        def wait(self):
            return 0

    def fake_push(path, remote, wait=False):
        pushed.append(os.path.basename(path))
        return Done()

    monkeypatch.setattr(train_mod, "rclone_push", fake_push)
    monkeypatch.setattr(train_mod, "rclone_pull_latest", lambda remote, out_dir: None)
    main(["--config", SMOKE, f"train.out_dir={tmp_path / 'r'}", "train.ckpt_remote=fake:bucket/r"])
    assert pushed == ["ckpt_00000010.pt", "log.jsonl", "ckpt_00000020.pt", "log.jsonl", "log.jsonl"]


def test_pull_latest_also_pulls_log(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if cmd[1] == "lsf":
            return subprocess.CompletedProcess(cmd, 0, stdout="ckpt_00000010.pt\nckpt_00000020.pt\nlog.jsonl\n")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(ckpt_mod, "have_rclone", lambda: True)
    monkeypatch.setattr(ckpt_mod.subprocess, "run", fake_run)
    out = str(tmp_path / "o")
    assert ckpt_mod.rclone_pull_latest("r2:b/run", out).endswith("ckpt_00000020.pt")
    assert [c[2] for c in calls[1:]] == ["r2:b/run/ckpt_00000020.pt", "r2:b/run/log.jsonl"]

    calls.clear()
    open(os.path.join(out, "log.jsonl"), "w").close()  # a local log is never overwritten
    ckpt_mod.rclone_pull_latest("r2:b/run", out)
    assert [c[2] for c in calls[1:]] == ["r2:b/run/ckpt_00000020.pt"]


# ---------------------------------------------------------------- HellaSwag download
def test_hellaswag_download_is_atomic(tmp_path, monkeypatch):
    seen = []

    def fake_retrieve(url, path):
        seen.append(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"ctx": "a", "endings": ["b", "c", "d", "e"], "label": 0}) + "\n")

    monkeypatch.setattr(eval_mod.urllib.request, "urlretrieve", fake_retrieve)
    ex = eval_mod.load_hellaswag(cache_dir=str(tmp_path))
    assert len(ex) == 1 and seen[0].endswith(".tmp") and seen[0] != str(tmp_path / "hellaswag_val.jsonl")
    assert os.listdir(tmp_path) == ["hellaswag_val.jsonl"]
    eval_mod.load_hellaswag(cache_dir=str(tmp_path))
    assert len(seen) == 1  # cached

    def failing(url, path):
        open(path, "w").write("{partial")
        raise OSError("network down")

    monkeypatch.setattr(eval_mod.urllib.request, "urlretrieve", failing)
    with pytest.raises(OSError):
        eval_mod.load_hellaswag(cache_dir=str(tmp_path / "x"))
    assert os.listdir(tmp_path / "x") == []  # no half-written file left behind
