import os
import random
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
sys.path.insert(0, SCRIPTS)

WORDS = ("the model learns tokens from text and the loss goes down while the gradient flows "
         "through attention layers residual streams embeddings norms optimizer schedule data "
         "shards checkpoints resume évaluation naïve 東京 42 3.14 hello world").split()


def fake_texts(n: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    return [" ".join(rng.choice(WORDS) for _ in range(rng.randint(20, 120))) + "." for _ in range(n)]


def run_script(name: str, *args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    res = subprocess.run([sys.executable, os.path.join(SCRIPTS, name), *args], capture_output=True,
                         text=True, encoding="utf-8", timeout=timeout, cwd=ROOT)
    assert res.returncode == 0, f"{name} failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    return res


@pytest.fixture(scope="session")
def tiny_tokenizer(tmp_path_factory):
    from mygpt.tokenizer import train_bpe
    out = str(tmp_path_factory.mktemp("tok"))
    train_bpe(iter(fake_texts(400)), vocab_size=400, out_dir=out, min_frequency=1)
    return out


@pytest.fixture(scope="session")
def parquet_dir(tmp_path_factory):
    import pyarrow as pa
    import pyarrow.parquet as pq
    d = tmp_path_factory.mktemp("parquet")
    for i in range(2):
        pq.write_table(pa.table({"text": fake_texts(150, seed=i), "id": list(range(150))}), d / f"part_{i:02d}.parquet")
    return str(d)
