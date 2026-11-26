"""Deterministic PyTorch dataset for univariate weather forecasting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.reproducibility import make_generator, seed_worker


class WeatherDataset(Dataset):
    """Rolling windows from one chronological Hugging Face weather split."""

    def __init__(
        self,
        processed_dir: str | Path,
        split: str,
        context_len: int | None = None,
        horizon: int | None = None,
    ) -> None:
        processed_dir = Path(processed_dir)
        manifest_path = processed_dir / "weather_manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        hf_split = "validation" if split == "val" else split
        info = self.manifest["splits"][hf_split]
        self.context_len = context_len or int(self.manifest["context_len"])
        self.horizon = horizon or int(self.manifest["forecast_horizon"])
        self.split = split

        self.values = np.load(info["array"])
        if self.context_len <= 0 or self.horizon <= 0:
            raise ValueError("context_len and horizon must be positive")
        self.start_indices = np.arange(
            self.context_len, self.values.size - self.horizon + 1, dtype=np.int64
        )

    def __len__(self) -> int:
        return int(self.start_indices.size)

    def __getitem__(self, index: int) -> dict:
        target_index = int(self.start_indices[index])
        history = self.values[target_index - self.context_len : target_index]
        future = self.values[target_index : target_index + self.horizon]
        x = np.ascontiguousarray(history[None, :], dtype=np.float32)
        target = np.ascontiguousarray(future[None, :], dtype=np.float32)
        mask = np.ones(self.context_len, dtype=np.float32)
        return {
            "x_enc": torch.from_numpy(x),
            "input_mask": torch.from_numpy(mask),
            "target": torch.from_numpy(target),
            "target_index": target_index,
            "split": self.split,
        }


def make_weather_loaders(
    processed_dir: str | Path,
    batch_size: int = 16,
    seed: int = 42,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    for split, offset in (("train", 0), ("val", 1), ("test", 2)):
        dataset = WeatherDataset(processed_dir, split)
        generator = make_generator(seed + offset, device="cpu")
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            worker_init_fn=seed_worker if num_workers > 0 else None,
            generator=generator,
            drop_last=False,
        )
    return loaders
