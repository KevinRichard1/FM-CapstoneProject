"""Deterministic PyTorch dataset for chronological StockNet windows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.reproducibility import make_generator, seed_worker


class StockNetDataset(Dataset):
    """Returns [features, context_len] tensors for one chronological split."""

    def __init__(self, processed_dir: str | Path, split: str) -> None:
        processed_dir = Path(processed_dir)
        self.manifest = json.loads(
            (processed_dir / "stocknet_manifest.json").read_text(encoding="utf-8")
        )
        metadata = pd.read_csv(processed_dir / "window_metadata.csv")
        self.context_len = int(self.manifest["context_len"])
        self.split = split
        self.indices = np.flatnonzero(metadata["split"].to_numpy() == split)
        if self.indices.size == 0:
            raise ValueError(f"No StockNet windows for split {split}")
        self.x = np.load(processed_dir / "X.npy", mmap_mode="r")
        self.y = np.load(processed_dir / "y.npy", mmap_mode="r")

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, index: int) -> dict:
        row = int(self.indices[index])
        values = np.asarray(self.x[row], dtype=np.float32)
        x = np.ascontiguousarray(values.T)
        mask = np.ones(self.context_len, dtype=np.float32)
        return {
            "x_enc": torch.from_numpy(x),
            "input_mask": torch.from_numpy(mask),
            "label": torch.tensor(int(self.y[row]), dtype=torch.long),
            "window_index": row,
            "split": self.split,
        }


def make_stocknet_loaders(
    processed_dir: str | Path,
    batch_size: int = 16,
    seed: int = 42,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    for split, offset in (("train", 0), ("val", 1), ("test", 2)):
        dataset = StockNetDataset(processed_dir, split)
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
