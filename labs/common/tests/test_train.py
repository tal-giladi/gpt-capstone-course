import json
import os
import shutil

import pytest

from mygpt.checkpoint import list_checkpoints, load_checkpoint
from mygpt.train import lr_mult, main

from conftest import ROOT, run_script

SMOKE = os.path.join(ROOT, "configs", "smoke_cpu.yaml")


def read_log(out_dir):
    with open(os.path.join(out_dir, "log.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def losses(out_dir):
    return {r["step"]: r["loss"] for r in read_log(out_dir) if "loss" in r}


def test_smoke_train_and_exact_resume(tmp_path):
    a = str(tmp_path / "a")
    res = main(["--config", SMOKE, f"train.out_dir={a}", "train.log_every=1", "train.keep_last=5"])
    la = losses(a)
    assert sorted(la) == list(range(1, 21))
    assert la[20] < la[1] - 1.0, f"loss did not decrease: {la[1]:.3f} -> {la[20]:.3f}"
    vals = [r["val_loss"] for r in read_log(a) if "val_loss" in r]
    assert len(vals) == 2 and vals[1] < vals[0]
    assert [os.path.basename(p) for p in list_checkpoints(a)] == ["ckpt_00000010.pt", "ckpt_00000020.pt"]
    assert res["last_checkpoint"].endswith("ckpt_00000020.pt")
    ck = load_checkpoint(list_checkpoints(a)[0])
    assert ck["step"] == 10 and ck["loader"]["pos"] > 0 and ck["config"]["model"]["vocab_size"] == 256

    # Fresh out_dir holding only the step-10 checkpoint: resuming must reproduce steps 11..20.
    b = str(tmp_path / "b")
    os.makedirs(b)
    shutil.copy(list_checkpoints(a)[0], b)
    shutil.copytree(os.path.join(a, "synthetic_data"), os.path.join(b, "synthetic_data"))
    main(["--config", SMOKE, f"train.out_dir={b}", "train.log_every=1", "train.keep_last=5"])
    lb = losses(b)
    assert sorted(lb) == list(range(11, 21))
    for s in range(11, 21):
        assert lb[s] == pytest.approx(la[s], abs=1e-5), f"step {s}: {lb[s]} vs {la[s]}"

    # A finished run resumes to "nothing left to do" instead of retraining.
    again = main(["--config", SMOKE, f"train.out_dir={b}"])
    assert again["last_checkpoint"].endswith("ckpt_00000020.pt")


def test_muon_and_modern_arch_train(tmp_path):
    out = str(tmp_path / "m")
    main(["--config", SMOKE, f"train.out_dir={out}", "train.optimizer=muon", "train.schedule=wsd",
          "model.pos_emb=rope", "model.norm=rmsnorm", "model.mlp=swiglu", "model.qk_norm=true", "model.bias=false",
          "train.log_every=1", "train.muon_lr=0.02", "train.dtype=bf16"])
    lm = losses(out)
    assert lm[20] < lm[1] - 1.0
    ck = load_checkpoint(list_checkpoints(out)[-1])
    assert len(ck["optimizers"]) == 2  # AdamW + Muon


def test_lr_schedules():
    class T:
        schedule, min_lr_ratio, decay_frac = "cosine", 0.1, 0.2
    assert lr_mult(0, 100, 10, T) == pytest.approx(0.1)
    assert lr_mult(10, 100, 10, T) == pytest.approx(1.0)
    assert lr_mult(99, 100, 10, T) == pytest.approx(0.1, abs=1e-3)
    T.schedule = "wsd"
    assert lr_mult(50, 100, 10, T) == 1.0 and lr_mult(79, 100, 10, T) == 1.0
    assert lr_mult(90, 100, 10, T) == pytest.approx(0.55)


def test_bench_runs_on_cpu(tmp_path):
    out = str(tmp_path / "bench.jsonl")
    res = run_script("bench.py", "--config", SMOKE, "--steps", "3", "--warmup", "1", "--price-per-gpu-hour", "2.5",
                     "--json", out, "train.micro_batch_size=4")
    r = json.loads(open(out, encoding="utf-8").read())
    assert r["tokens_per_s_per_gpu"] > 0 and r["usd_per_1b_tokens"] > 0 and r["gpu"] == "cpu"
    assert r["micro_batch"] == 4 and r["warmup_steps"] == 1 and r["data"] == "random tokens"
    assert "random tokens" in res.stdout and "per 1B tokens" in res.stdout
