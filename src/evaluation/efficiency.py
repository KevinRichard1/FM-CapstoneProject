"""Lightweight efficiency measurements for MOMENT runs."""

from __future__ import annotations

import time
from pathlib import Path

import torch


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_gpu_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024**2))


def checkpoint_size_mb(path: str | Path) -> float:
    path = Path(path)
    return float(path.stat().st_size / (1024**2)) if path.exists() else 0.0


@torch.no_grad()
def measure_inference(
    model,
    x: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    warmup: int = 2,
    repeats: int = 5,
) -> dict[str, float]:
    """Measure batch latency and throughput for one fixed input shape."""
    model.eval()
    x = x.to(device)
    mask = mask.to(device)

    for _ in range(warmup):
        model(x_enc=x, input_mask=mask)
    if device.type == "cuda":
        torch.cuda.synchronize()

    started = time.perf_counter()
    for _ in range(repeats):
        model(x_enc=x, input_mask=mask)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    batch_size = int(x.shape[0])
    return {
        "repeats": float(repeats),
        "batch_size": float(batch_size),
        "mean_batch_ms": round(1000.0 * elapsed / repeats, 3),
        "samples_per_second": round(batch_size * repeats / elapsed, 2),
    }
