"""Seeding helpers shared by all experiments."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch RNGs.

    Deterministic CUDA kernels are enabled when requested; this may reduce
    throughput but keeps repeated runs comparable.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_generator(seed: int, device: Optional[str] = None) -> torch.Generator:
    """Create a seeded generator for DataLoader workers or torch.randperm."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    """Seed NumPy/Python RNG inside each DataLoader worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
