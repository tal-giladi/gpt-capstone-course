import json
import os

import numpy as np
import pytest
import torch

from mygpt.data import (MAGIC, DistributedTokenLoader, read_header, read_meta, read_shard, write_meta,
                        write_shard)
from mygpt.tokenizer import get_tokenizer

from conftest import run_script


def make_dir(tmp_path, n_train=3, size=1000):
    d = tmp_path / "shards"
    d.mkdir()
    for i in range(n_train):
        write_shard(str(d / f"train_{i:06d}.bin"), np.arange(size, dtype=np.int64) % 500 + i, 512)
    write_shard(str(d / "val_000000.bin"), np.arange(size) % 300, 512)
    write_meta(str(d), {"tokenizer": "synthetic", "vocab_size": 512})
    return str(d)


def test_shard_roundtrip(tmp_path):
    toks = np.array([0, 5, 65535, 7], dtype=np.int64)
    p = str(tmp_path / "train_000000.bin")
    write_shard(p, toks, 65536)
    assert read_header(p) == {"n_tokens": 4, "vocab_size": 65536}
    assert np.fromfile(p, dtype=np.int32, count=1)[0] == MAGIC
    assert os.path.getsize(p) == 256 * 4 + 4 * 2
    assert read_shard(p).tolist() == toks.tolist()
    (tmp_path / "bad.bin").write_bytes(b"\0" * 2048)
    with pytest.raises(ValueError, match="magic"):
        read_header(str(tmp_path / "bad.bin"))


def test_loader_deterministic_resume_and_ranks(tmp_path):
    d = make_dir(tmp_path)
    a = DistributedTokenLoader(d, "train", B=2, T=8)
    seq = [a.next_batch() for _ in range(200)]  # crosses shard boundaries and wraps around
    for x, y in seq[:5]:
        assert x.shape == (2, 8) and x.dtype == torch.long and torch.equal(x[:, 1:], y[:, :-1])
    b = DistributedTokenLoader(d, "train", B=2, T=8)
    for _ in range(77):
        b.next_batch()
    state = b.state_dict()
    c = DistributedTokenLoader(d, "train", B=2, T=8)
    c.load_state_dict(json.loads(json.dumps(state)))
    for i in range(77, 120):
        x, y = c.next_batch()
        assert torch.equal(x, seq[i][0]) and torch.equal(y, seq[i][1])
    # two ranks see disjoint consecutive windows of the same global chunk
    r0 = DistributedTokenLoader(d, "train", B=2, T=8, rank=0, world_size=2)
    r1 = DistributedTokenLoader(d, "train", B=2, T=8, rank=1, world_size=2)
    x0, _ = r0.next_batch()
    x1, _ = r1.next_batch()
    full = torch.from_numpy(read_shard(os.path.join(d, "train_000000.bin"))[:33].astype(np.int64))
    assert torch.equal(torch.cat([x0.flatten(), x1.flatten()]), full[:32])
    assert r0.state_dict() == r1.state_dict()


def test_tokenize_report_decontam_scripts(parquet_dir, tiny_tokenizer, tmp_path):
    out = str(tmp_path / "shards")
    run_script("tokenize_shards.py", "--input", parquet_dir, "--tokenizer", tiny_tokenizer, "--out", out,
               "--shard-tokens", "4000", "--val-tokens", "2000", "--workers", "2", "--docs-per-task", "16")
    meta = read_meta(out)
    tok = get_tokenizer(tiny_tokenizer)
    assert meta["eot_id"] == tok.eot_id and meta["val_tokens"] >= 2000 and meta["train_tokens"] > 10000
    assert meta["train_docs"] + meta["val_docs"] == 300
    from mygpt.data import resolve_tokenizer
    assert resolve_tokenizer(out) == out and os.path.exists(os.path.join(out, "tokenizer.json"))
    vals = sorted(f for f in os.listdir(out) if f.startswith("val_"))
    trains = sorted(f for f in os.listdir(out) if f.startswith("train_"))
    assert vals and len(trains) >= 3
    assert sum(read_header(os.path.join(out, f))["n_tokens"] for f in trains) == meta["train_tokens"]
    first = read_shard(os.path.join(out, vals[0]))
    assert first[0] == tok.eot_id  # every document starts with EOT
    # decoding the first val document gives back the first parquet text
    import pyarrow.parquet as pq
    text0 = pq.read_table(os.path.join(parquet_dir, "part_00.parquet"))["text"][0].as_py()
    ends = np.flatnonzero(np.asarray(first) == tok.eot_id)
    assert tok.decode(first[1:ends[1]].tolist()) == text0

    res = run_script("token_report.py", "--shards", out)
    report = json.load(open(os.path.join(out, "token_report.json"), encoding="utf-8"))
    assert report["total_tokens"] == meta["total_tokens"] and "bytes_per_token" in report
    assert "total:" in res.stdout

    # decontam: one benchmark item copied from the data, one unseen
    bench = tmp_path / "bench.jsonl"
    words = text0.split()
    leaked = " ".join(words[:30])
    bench.write_text(json.dumps({"benchmark": "toy", "text": leaked}) + "\n" +
                     json.dumps({"benchmark": "toy", "text": "zzzz qqqq " * 20}) + "\n", encoding="utf-8")
    rep = str(tmp_path / "decon.json")
    run_script("decontam.py", "--shards-or-parquet", out, "--tokenizer", tiny_tokenizer, "--split", "val",
               "--benchmark-file", str(bench), "--out", rep)
    r = json.load(open(rep, encoding="utf-8"))
    assert r["benchmarks"]["toy"]["contaminated_items"] == 1 and r["flagged_docs"] >= 1
    drop = str(tmp_path / "drop.json")
    run_script("decontam.py", "--shards-or-parquet", parquet_dir, "--tokenizer", tiny_tokenizer,
               "--benchmark-file", str(bench), "--out", rep, "--drop", drop)
    flagged = json.load(open(drop, encoding="utf-8"))
    assert ["part_00.parquet", 0] in flagged
    out2 = str(tmp_path / "shards2")
    run_script("tokenize_shards.py", "--input", parquet_dir, "--tokenizer", tiny_tokenizer, "--out", out2,
               "--shard-tokens", "4000", "--val-tokens", "2000", "--workers", "1", "--exclude", drop)
    assert read_meta(out2)["train_docs"] + read_meta(out2)["val_docs"] == 300 - len(flagged)
