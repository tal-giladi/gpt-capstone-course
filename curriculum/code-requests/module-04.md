# Code requests from Module 4 (for labs/common owner)

Lessons 04.2 and 04.3 call these exactly as written below. Please implement or tell the module
author what to change in the lessons.

1. **`scripts/autoshutdown.sh` interface.** Lesson 04.2 runs:
   `bash scripts/autoshutdown.sh --idle-minutes 5 --stop-cmd 'runpodctl pod stop $RUNPOD_POD_ID'`
   - Poll `nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits` every ~30 s; "idle"
     = all GPUs below ~5% for `--idle-minutes` consecutive minutes; log each decision to stdout.
   - `--stop-cmd` default when omitted: if `RUNPOD_POD_ID` is set, `runpodctl pod stop $RUNPOD_POD_ID`
     (fall back to `runpodctl stop pod $RUNPOD_POD_ID` if the first fails); otherwise
     `sudo shutdown -h now`, printing a warning that on Lambda/Hetzner/Vast a shutdown does **not**
     stop billing (they need an API call: Lambda terminate endpoint, `vastai stop instance`).
   - Never stop during the first `--grace-minutes` (default 10) after start, so setup time is not
     counted as idle.
   Important fact: `poweroff`/`shutdown` inside a RunPod container does not stop the pod, and on
   Lambda it does not stop billing (Lambda docs warn against it).
2. **`scripts/bench.py` behaviour the lesson assumes.**
   - Runs single-process on one GPU with the config's `micro_batch_size` and `seq_len`, even if
     the config is written for 8 GPUs, and reports **per-GPU** tokens/s.
   - Prints tokens/s, MFU (against the dense BF16 peak of the detected GPU: H100 SXM 989, H100 NVL
     835, H100 PCIe 756, H200 989, B200 2250, A100 312 TFLOP/s) and `$ per 1B tokens` at
     `--price-per-gpu-hour`; excludes compile warm-up steps from the timing (state how many).
   - Reads real shards from the config's `data_dir` if present; falls back to random tokens with a
     printed notice if not (throughput does not depend on content).
   - Nice to have: accept `key=value` overrides like `train.py` (e.g. `train.compile=false`), and a
     `--json OUT` flag so `budget_calc.py` can read results directly.
3. **Configs' `data_dir`.** Lesson 04.3 tells learners to copy shards to "the `data_dir` written in
   the config". Please use paths that match the bucket layout from lesson 02.2, e.g.
   `data/tok/gpt2-10BT` (baseline) and `data/tok/bpe32k-100BT-30` (ablation/main), and keep
   `tok/<same-name>/` in the bucket. Module 3 must write shards to those same bucket prefixes.
4. **`pyproject.toml`.** Torch from PyPI (CUDA 13.0 wheels, driver >= 580) is assumed; lesson 04.2
   checks the driver for this. `pytest -q` must pass on the GPU box as well as on CPU.
5. **Extras.** `setup_box.sh` (module-04 lab) runs `uv pip install -e ".[dev]"` and `pytest -q`; please
   keep the `[dev]` extra and the < 2 min CPU test budget.
