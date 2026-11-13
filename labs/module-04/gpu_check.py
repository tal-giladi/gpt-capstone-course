"""Five-minute health check for a freshly rented GPU box.

    python tools/gpu_check.py            # all visible GPUs
    python tools/gpu_check.py --quick    # skip the matmul benchmark

Checks: NVIDIA driver version (CUDA 13 wheels need >= 580), torch sees every GPU, bf16 works,
memory is free, and a bf16 matmul reaches a sane share of the GPU's peak. A host that fails
here will fail your training run too; release it and take another.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time

import torch

# Dense BF16 tensor-core peaks in TFLOP/s (vendor sparse figures halved). H100 SXM and H100 NVL:
# NVIDIA H100 page; B200: third-party spec sheets. Used only to judge "healthy or not".
PEAK_TFLOPS = {"H100 80GB HBM3": 989, "H100 SXM": 989, "H200": 989, "H100 NVL": 835,
               "H100 PCIe": 756, "B200": 2250, "A100": 312, "RTX 4090": 165, "L40S": 362}
HEALTHY_SHARE = 0.55  # a big bf16 matmul usually reaches 60-80% of dense peak on a healthy GPU


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"<{e}>"


def peak_for(name: str) -> int | None:
    for key, val in PEAK_TFLOPS.items():
        if key.lower() in name.lower():
            return val
    return None


def matmul_tflops(dev: int, n: int = 8192, iters: int = 50) -> float:
    torch.cuda.set_device(dev)
    a = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    b = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    for _ in range(5):
        a @ b
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize(dev)
    return 2 * n**3 * iters / (time.perf_counter() - t0) / 1e12


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    problems: list[str] = []

    if not shutil.which("nvidia-smi"):
        print("nvidia-smi not found: this is not a GPU box (or the driver is missing).")
        return 1
    driver = sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader").splitlines()[0]
    print(f"driver            : {driver}")
    print(f"torch             : {torch.__version__} (built for CUDA {torch.version.cuda})")
    cuda_major = int((torch.version.cuda or "0").split(".")[0])
    if cuda_major >= 13 and int(driver.split(".")[0]) < 580:
        problems.append(f"driver {driver} < 580 but torch is built for CUDA {torch.version.cuda}")

    if not torch.cuda.is_available():
        problems.append("torch.cuda.is_available() is False")
        print("\n".join("PROBLEM: " + x for x in problems))
        return 1

    n = torch.cuda.device_count()
    expected = len(sh("nvidia-smi -L").splitlines())
    print(f"GPUs              : torch sees {n}, nvidia-smi lists {expected}")
    if n != expected:
        problems.append("torch sees fewer GPUs than nvidia-smi (check CUDA_VISIBLE_DEVICES)")
    print(f"bf16 supported    : {torch.cuda.is_bf16_supported()}")

    for i in range(n):
        name = torch.cuda.get_device_name(i)
        free, total = torch.cuda.mem_get_info(i)
        line = f"GPU {i}: {name}, {total / 2**30:.0f} GiB, {free / 2**30:.0f} GiB free"
        if free < 0.9 * total:
            problems.append(f"GPU {i} has only {free / 2**30:.0f} GiB free: another process is using it")
        if not a.quick:
            tf = matmul_tflops(i)
            peak = peak_for(name)
            line += f", bf16 matmul {tf:.0f} TFLOP/s"
            if peak:
                line += f" ({tf / peak:.0%} of {peak} dense peak)"
                if tf < HEALTHY_SHARE * peak:
                    problems.append(f"GPU {i} reaches only {tf / peak:.0%} of peak (throttled, shared or misreported)")
        print(line)

    if n > 1:
        print("\nGPU interconnect (NV# = NVLink, SYS/PHB/NODE = PCIe only):")
        print(sh("nvidia-smi topo -m"))
    du = shutil.disk_usage(".")
    print(f"\ndisk free here    : {du.free / 1e9:.0f} GB")
    if du.free < 60e9:
        problems.append("less than 60 GB free disk: not enough for the main-run token shards")

    print()
    for x in problems:
        print("PROBLEM: " + x)
    print("HEALTHY" if not problems else f"{len(problems)} problem(s)")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
