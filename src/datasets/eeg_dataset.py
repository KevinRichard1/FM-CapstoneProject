"""Deterministic PyTorch dataset for 200 Hz ADFTD-RS EEG windows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.reproducibility import make_generator, seed_worker


class EEGDataset(Dataset):
    """Returns [channels, seq_len] tensors with a padding mask.

    The stored windows are [400, 19] (time, channels). They are z-scored with
    training-subject statistics, transposed to [19, 400], and zero-padded to
    ``seq_len`` (512 by default). Mask zeros mark padded positions.
    """

    def __init__(
        self,
        eeg_dir: str | Path,
        split: str,
        manifest_path: str | Path,
        scaler_path: str | Path,
        seq_len: int = 512,
        sample_rate_hz: int = 200,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")

        eeg_dir = Path(eeg_dir)
        self.meta = json.loads((eeg_dir / "meta.json").read_text(encoding="utf-8"))
        n, self.source_len, self.n_channels = (
            self.meta["N"],
            self.meta["T"],
            self.meta["C"],
        )
        if self.source_len > seq_len:
            raise ValueError("Stored EEG windows are longer than the model sequence length")
        self.seq_len = seq_len
        self.x = np.memmap(
            eeg_dir / "X-001.dat",
            dtype="float32",
            mode="r",
            shape=(n, self.source_len, self.n_channels),
        )
        y = np.memmap(eeg_dir / "y.dat", dtype="float32", mode="r", shape=(n, 3))

        manifest = pd.read_csv(manifest_path)
        split_subjects = set(
            manifest.loc[manifest["split"] == split, "subject_id"].astype(int)
        )
        subjects = np.rint(y[:, 1]).astype(int)
        rates = np.rint(y[:, 2]).astype(int)
        self.indices = np.flatnonzero(
            (rates == sample_rate_hz) & np.isin(subjects, list(split_subjects))
        )
        self.labels = np.rint(y[self.indices, 0]).astype(np.int64)
        self.subject_ids = subjects[self.indices]

        scaler = np.load(scaler_path)
        self.mean = scaler["mean"].astype(np.float32)
        self.std = scaler["std"].astype(np.float32)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, index: int) -> dict:
        row = int(self.indices[index])
        window = np.asarray(self.x[row], dtype=np.float32)
        window = (window - self.mean) / self.std

        values = np.zeros((self.n_channels, self.seq_len), dtype=np.float32)
        values[:, : self.source_len] = window.T
        mask = np.zeros(self.seq_len, dtype=np.float32)
        mask[: self.source_len] = 1.0

        return {
            "x_enc": torch.from_numpy(values),
            "input_mask": torch.from_numpy(mask),
            "label": torch.tensor(int(self.labels[index]), dtype=torch.long),
            "subject_id": int(self.subject_ids[index]),
            "window_index": row,
        }


def make_eeg_loaders(
    eeg_dir: str | Path,
    manifest_path: str | Path,
    scaler_path: str | Path,
    batch_size: int = 16,
    seed: int = 42,
    num_workers: int = 0,
    seq_len: int = 512,
) -> dict[str, DataLoader]:
    """Create train/val/test loaders with seeded generators."""
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        dataset = EEGDataset(
            eeg_dir=eeg_dir,
            split=split,
            manifest_path=manifest_path,
            scaler_path=scaler_path,
            seq_len=seq_len,
        )
        generator = make_generator(
            seed + {"train": 0, "val": 1, "test": 2}[split], device="cpu"
        )
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
