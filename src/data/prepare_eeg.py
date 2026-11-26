"""Fit EEG normalization statistics on training subjects only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compute_train_statistics(
    eeg_dir: Path, manifest: pd.DataFrame, chunk_size: int = 4096
) -> tuple[np.ndarray, np.ndarray, int]:
    meta = json.loads((eeg_dir / "meta.json").read_text(encoding="utf-8"))
    n, time_steps, channels = meta["N"], meta["T"], meta["C"]
    x = np.memmap(
        eeg_dir / "X-001.dat", dtype="float32", mode="r", shape=(n, time_steps, channels)
    )
    y = np.memmap(eeg_dir / "y.dat", dtype="float32", mode="r", shape=(n, 3))

    rates = np.rint(y[:, 2]).astype(int)
    subjects = np.rint(y[:, 1]).astype(int)
    train_subjects = set(
        manifest.loc[manifest["split"] == "train", "subject_id"].astype(int)
    )
    indices = np.flatnonzero((rates == 200) & np.isin(subjects, list(train_subjects)))
    if indices.size == 0:
        raise ValueError("No 200 Hz training windows found")

    total = np.zeros(channels, dtype=np.float64)
    total_squares = np.zeros(channels, dtype=np.float64)
    for start in range(0, indices.size, chunk_size):
        batch = np.asarray(x[indices[start : start + chunk_size]], dtype=np.float64)
        total += batch.sum(axis=(0, 1))
        total_squares += np.square(batch).sum(axis=(0, 1))

    count = indices.size * time_steps
    mean = total / count
    variance = np.maximum(total_squares / count - np.square(mean), 1e-12)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32), int(indices.size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eeg-dir", default="data/ADFTD-RS")
    parser.add_argument("--manifest", default="data/manifests/eeg_subject_split.csv")
    parser.add_argument("--output", default="data/processed/eeg_scaler.npz")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    mean, std, n_windows = compute_train_statistics(Path(args.eeg_dir), manifest)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        mean=mean,
        std=std,
        n_train_windows=np.array(n_windows),
        sample_rate_hz=np.array(200),
    )
    print(f"Saved EEG scaler to {output} using {n_windows} training windows")
    print(f"mean={mean}")
    print(f"std={std}")


if __name__ == "__main__":
    main()
