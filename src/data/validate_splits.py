"""Executable leakage and shape checks for all three datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.datasets.eeg_dataset import EEGDataset, make_eeg_loaders
from src.datasets.stocknet_dataset import StockNetDataset, make_stocknet_loaders
from src.datasets.weather_dataset import WeatherDataset, make_weather_loaders


def check_eeg(args: argparse.Namespace) -> dict:
    manifest = pd.read_csv(args.eeg_manifest)
    split_sets = {
        split: set(manifest.loc[manifest["split"] == split, "subject_id"].astype(int))
        for split in ("train", "val", "test")
    }
    assert split_sets["train"].isdisjoint(split_sets["val"])
    assert split_sets["train"].isdisjoint(split_sets["test"])
    assert split_sets["val"].isdisjoint(split_sets["test"])

    summary: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        dataset = EEGDataset(args.eeg_dir, split, args.eeg_manifest, args.eeg_scaler)
        batch = dataset[0]
        subjects = set(int(value) for value in dataset.subject_ids)
        assert subjects.issubset(split_sets[split])
        assert batch["x_enc"].shape == (19, 512)
        assert batch["input_mask"].shape == (512,)
        assert torch.isfinite(batch["x_enc"]).all()
        assert torch.isfinite(batch["input_mask"]).all()
        summary[split] = {
            "windows": len(dataset),
            "subjects": len(subjects),
            "x_shape": list(batch["x_enc"].shape),
            "finite": True,
        }
    loaders = make_eeg_loaders(
        args.eeg_dir, args.eeg_manifest, args.eeg_scaler, batch_size=4, seed=42
    )
    for split, loader in loaders.items():
        batch = next(iter(loader))
        batch_size = batch["x_enc"].shape[0]
        assert batch["x_enc"].shape == (batch_size, 19, 512)
        assert batch["input_mask"].shape == (batch_size, 512)
        assert torch.isfinite(batch["x_enc"]).all()
        assert torch.isfinite(batch["input_mask"]).all()
        summary[split]["loader_batch_shape"] = list(batch["x_enc"].shape)
    return {"subject_sets_disjoint": True, "splits": summary}


def check_weather(args: argparse.Namespace) -> dict:
    manifest = json.loads((Path(args.weather_processed) / "weather_manifest.json").read_text(encoding="utf-8"))
    summary: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        dataset = WeatherDataset(args.weather_processed, split)
        info = manifest["splits"]["validation" if split == "val" else split]
        starts = dataset.start_indices
        assert starts.min() >= info["first_target_index"]
        assert starts.max() <= info["last_target_index"]
        assert len(dataset) == info["window_count"]
        batch = dataset[0]
        assert batch["x_enc"].shape == (1, manifest["context_len"])
        assert batch["target"].shape == (1, manifest["forecast_horizon"])
        assert torch.isfinite(batch["x_enc"]).all()
        assert torch.isfinite(batch["target"]).all()
        summary[split] = {
            "windows": len(dataset),
            "x_shape": list(batch["x_enc"].shape),
            "target_shape": list(batch["target"].shape),
            "finite": True,
        }
    loaders = make_weather_loaders(args.weather_processed, batch_size=4, seed=42)
    for split, loader in loaders.items():
        batch = next(iter(loader))
        batch_size = batch["x_enc"].shape[0]
        assert batch["x_enc"].shape == (batch_size, 1, manifest["context_len"])
        assert batch["target"].shape == (batch_size, 1, manifest["forecast_horizon"])
        assert torch.isfinite(batch["x_enc"]).all()
        assert torch.isfinite(batch["target"]).all()
        summary[split]["loader_batch_shape"] = list(batch["x_enc"].shape)
    return {"targets_within_split": True, "splits": summary}


def check_stocknet(args: argparse.Namespace) -> dict:
    metadata = pd.read_csv(Path(args.stocknet_processed) / "window_metadata.csv")
    manifest = json.loads(
        (Path(args.stocknet_processed) / "stocknet_manifest.json").read_text(encoding="utf-8")
    )
    history = pd.to_datetime(metadata["history_end_date"])
    target = pd.to_datetime(metadata["target_date"])
    assert (history < target).all(), "History reaches the target date"
    assert (metadata["history_end_row"] < metadata["row_index"]).all(), "History rows overlap the target"
    assert metadata["split"].isin(manifest["boundaries"]).all()

    summary: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        start, end = manifest["boundaries"][split]
        subset = metadata[metadata["split"] == split]
        assert (
            (pd.to_datetime(subset["target_date"]) >= pd.Timestamp(start))
            & (pd.to_datetime(subset["target_date"]) < pd.Timestamp(end))
        ).all()
        dataset = StockNetDataset(args.stocknet_processed, split)
        batch = dataset[0]
        assert batch["x_enc"].shape == (len(manifest["features"]), manifest["context_len"])
        assert torch.isfinite(batch["x_enc"]).all()
        assert torch.isfinite(batch["input_mask"]).all()
        summary[split] = {
            "windows": len(dataset),
            "x_shape": list(batch["x_enc"].shape),
            "finite": True,
        }
    loaders = make_stocknet_loaders(args.stocknet_processed, batch_size=4, seed=42)
    for split, loader in loaders.items():
        batch = next(iter(loader))
        batch_size = batch["x_enc"].shape[0]
        assert batch["x_enc"].shape == (batch_size, len(manifest["features"]), manifest["context_len"])
        assert torch.isfinite(batch["x_enc"]).all()
        assert torch.isfinite(batch["input_mask"]).all()
        summary[split]["loader_batch_shape"] = list(batch["x_enc"].shape)
    return {
        "history_before_target": True,
        "history_rows_before_target": True,
        "targets_within_boundaries": True,
        "splits": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eeg-dir", default="data/ADFTD-RS")
    parser.add_argument("--eeg-manifest", default="data/manifests/eeg_subject_split.csv")
    parser.add_argument("--eeg-scaler", default="data/processed/eeg_scaler.npz")
    parser.add_argument("--weather-processed", default="data/processed/weather")
    parser.add_argument("--stocknet-processed", default="data/processed/stocknet")
    args = parser.parse_args()

    result = {
        "eeg": check_eeg(args),
        "weather": check_weather(args),
        "stocknet": check_stocknet(args),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
