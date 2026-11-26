"""Train and evaluate the three required non-foundation-model baselines."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader

from src.datasets.eeg_dataset import EEGDataset
from src.datasets.stocknet_dataset import StockNetDataset
from src.datasets.weather_dataset import WeatherDataset
from src.evaluation.metrics import classification_metrics, forecasting_metrics
from src.models.baselines import (
    eeg_spectral_features,
    persistence_forecast,
    stocknet_lagged_features,
)
from src.utils.reproducibility import set_seed


def append_metrics(rows: list[dict], path: Path) -> None:
    frame = pd.DataFrame(rows)
    if path.exists():
        existing = pd.read_csv(path)
        keys = ["dataset", "model", "split", "aggregation", "metric", "config"]
        overlap = existing.set_index(keys).index.isin(frame.set_index(keys).index)
        frame = pd.concat([existing[~overlap], frame], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def record_classification(
    rows: list[dict],
    result: dict,
    dataset_name: str,
    model: str,
    split: str,
    aggregation: str,
    config: dict,
    seed: int,
) -> None:
    for metric, value in result.items():
        rows.append(
            {
                "dataset": dataset_name,
                "model": model,
                "split": split,
                "aggregation": aggregation,
                "metric": metric,
                "value": json.dumps(value) if isinstance(value, list) else value,
                "seed": seed,
                "config": json.dumps(config, sort_keys=True),
            }
        )


def record_forecasting(
    rows: list[dict],
    result: dict,
    split: str,
    config: dict,
    seed: int,
) -> None:
    for metric, value in result.items():
        rows.append(
            {
                "dataset": "weather",
                "model": "persistence",
                "split": split,
                "aggregation": "window",
                "metric": metric,
                "value": json.dumps(value) if isinstance(value, list) else value,
                "seed": seed,
                "config": json.dumps(config, sort_keys=True),
            }
        )


def eeg_features(dataset: EEGDataset, batch_size: int = 2048) -> np.ndarray:
    blocks = []
    for start in range(0, len(dataset.indices), batch_size):
        rows = dataset.indices[start : start + batch_size]
        windows = np.asarray(dataset.x[rows], dtype=np.float32)
        blocks.append(eeg_spectral_features(windows, sample_rate=200))
    return np.concatenate(blocks, axis=0)


def aggregate_subject_probabilities(
    probabilities: np.ndarray,
    subjects: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    columns = [f"p_{int(value)}" for value in classes]
    frame = pd.DataFrame(probabilities, columns=columns)
    frame["subject_id"] = subjects
    frame["label"] = labels
    grouped = frame.groupby("subject_id", sort=True)
    mean_probabilities = grouped[columns].mean().to_numpy()
    subject_labels = grouped["label"].first().to_numpy()
    subject_predictions = classes[mean_probabilities.argmax(axis=1)]
    return subject_predictions, subject_labels, mean_probabilities


def save_classification_predictions(
    path: Path,
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    identifiers: dict[str, np.ndarray],
) -> None:
    data = {"label": labels, "prediction": predictions}
    for index, class_value in enumerate(classes):
        data[f"p_{int(class_value)}"] = probabilities[:, index]
    data.update(identifiers)
    pd.DataFrame(data).to_csv(path, index=False)


def run_eeg_baseline(args: argparse.Namespace, rows: list[dict]) -> None:
    config = {
        "features": "relative_band_power_plus_log_total_power",
        "bands_hz": {"delta": [0.5, 4], "theta": [4, 8], "alpha": [8, 13], "beta": [13, 30], "gamma": [30, 45]},
        "model": "SVC_rbf",
        "train_subset": 6000,
        "grid": {"C": [1.0, 10.0], "gamma": ["scale", 0.01]},
        "selection": "macro_f1 on validation only",
    }
    datasets = {
        split: EEGDataset(
            args.eeg_dir, split, args.eeg_manifest, args.eeg_scaler
        )
        for split in ("train", "val", "test")
    }
    features = {split: eeg_features(datasets[split]) for split in datasets}
    labels = {split: datasets[split].labels for split in datasets}

    scaler = StandardScaler().fit(features["train"])
    scaled = {split: scaler.transform(features[split]) for split in datasets}
    rng = np.random.default_rng(args.seed)
    subset = rng.choice(len(scaled["train"]), size=min(config["train_subset"], len(scaled["train"])), replace=False)

    best_score = -1.0
    best: tuple[SVC, float, float | str] | None = None
    started = time.perf_counter()
    for c_value in config["grid"]["C"]:
        for gamma_value in config["grid"]["gamma"]:
            candidate = SVC(
                kernel="rbf",
                C=c_value,
                gamma=gamma_value,
                class_weight="balanced",
                probability=True,
                random_state=args.seed,
            )
            candidate.fit(scaled["train"][subset], labels["train"][subset])
            validation_probabilities = candidate.predict_proba(scaled["val"])
            validation_predictions = candidate.classes_[validation_probabilities.argmax(axis=1)]
            score = f1_score(labels["val"], validation_predictions, average="macro", zero_division=0)
            if score > best_score:
                best_score = score
                best = (candidate, c_value, gamma_value)
    if best is None:
        raise RuntimeError("No EEG baseline candidate was fitted")
    model, best_c, best_gamma = best
    config["selected"] = {"C": best_c, "gamma": best_gamma, "validation_macro_f1": float(best_score)}
    config["selection_seconds"] = round(time.perf_counter() - started, 2)

    for split in ("train", "val", "test"):
        probabilities = model.predict_proba(scaled[split])
        predictions = model.classes_[probabilities.argmax(axis=1)]
        identifiers = {
            "subject_id": datasets[split].subject_ids,
            "window_index": datasets[split].indices,
        }
        save_classification_predictions(
            Path(args.prediction_dir) / f"eeg_svm_{split}.csv",
            labels[split],
            predictions,
            probabilities,
            model.classes_,
            identifiers,
        )
        window_result = classification_metrics(labels[split], predictions, probabilities)
        record_classification(rows, window_result, "eeg", "rbf_svm", split, "window", config, args.seed)
        subject_predictions, subject_labels, subject_probabilities = aggregate_subject_probabilities(
            probabilities, datasets[split].subject_ids, labels[split], model.classes_
        )
        subject_result = classification_metrics(
            subject_labels, subject_predictions, subject_probabilities
        )
        record_classification(
            rows, subject_result, "eeg", "rbf_svm", split, "subject", config, args.seed
        )
    print(f"EEG RBF-SVM selected {config['selected']}")


def run_weather_baseline(args: argparse.Namespace, rows: list[dict]) -> None:
    manifest = json.loads(
        (Path(args.weather_processed) / "weather_manifest.json").read_text(encoding="utf-8")
    )
    config = {
        "model": "persistence",
        "horizon": manifest["forecast_horizon"],
        "units": "standardized",
    }
    prediction_dir = Path(args.prediction_dir)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        dataset = WeatherDataset(args.weather_processed, split)
        loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
        predictions, targets = [], []
        for batch in loader:
            predictions.append(
                persistence_forecast(batch["x_enc"].numpy(), config["horizon"])
            )
            targets.append(batch["target"].numpy())
        predicted = np.concatenate(predictions, axis=0)
        actual = np.concatenate(targets, axis=0)
        np.savez_compressed(
            prediction_dir / f"weather_persistence_{split}.npz",
            predictions=predicted,
            targets=actual,
        )
        result = forecasting_metrics(actual, predicted)
        record_forecasting(rows, result, split, config, args.seed)
    print(f"Weather persistence completed with horizon={config['horizon']}")


def stocknet_features(dataset: StockNetDataset, batch_size: int = 2048) -> np.ndarray:
    blocks = []
    for start in range(0, len(dataset.indices), batch_size):
        rows = dataset.indices[start : start + batch_size]
        windows = np.asarray(dataset.x[rows], dtype=np.float32)
        blocks.append(stocknet_lagged_features(windows, recent_steps=5))
    return np.concatenate(blocks, axis=0)


def run_stocknet_baseline(args: argparse.Namespace, rows: list[dict]) -> None:
    config = {
        "features": "last_5_steps_flattened_plus_window_mean_std",
        "model": "logistic_regression",
        "grid": {"C": [0.01, 0.1, 1.0]},
        "selection": "macro_f1 on validation only",
    }
    datasets = {
        split: StockNetDataset(args.stocknet_processed, split)
        for split in ("train", "val", "test")
    }
    features = {split: stocknet_features(datasets[split]) for split in datasets}
    labels = {split: datasets[split].y[datasets[split].indices].astype(int) for split in datasets}

    scaler = StandardScaler().fit(features["train"])
    scaled = {split: scaler.transform(features[split]) for split in datasets}

    best_score, best = -1.0, None
    for c_value in config["grid"]["C"]:
        candidate = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=1000,
            random_state=args.seed,
        )
        candidate.fit(scaled["train"], labels["train"])
        validation_probabilities = candidate.predict_proba(scaled["val"])
        validation_predictions = candidate.classes_[validation_probabilities.argmax(axis=1)]
        score = f1_score(labels["val"], validation_predictions, average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best = (candidate, c_value)
    if best is None:
        raise RuntimeError("No StockNet baseline candidate was fitted")
    model, best_c = best
    config["selected"] = {"C": best_c, "validation_macro_f1": float(best_score)}

    for split in ("train", "val", "test"):
        probabilities = model.predict_proba(scaled[split])
        predictions = model.classes_[probabilities.argmax(axis=1)]
        save_classification_predictions(
            Path(args.prediction_dir) / f"stocknet_logreg_{split}.csv",
            labels[split],
            predictions,
            probabilities,
            model.classes_,
            {"window_index": datasets[split].indices},
        )
        result = classification_metrics(labels[split], predictions, probabilities)
        record_classification(
            rows, result, "stocknet", "logistic_regression", split, "window", config, args.seed
        )
    print(f"StockNet logistic regression selected {config['selected']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["eeg", "weather", "stocknet", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eeg-dir", default="data/ADFTD-RS")
    parser.add_argument("--eeg-manifest", default="data/manifests/eeg_subject_split.csv")
    parser.add_argument("--eeg-scaler", default="data/processed/eeg_scaler.npz")
    parser.add_argument("--weather-dir", default="data/raw/weather_hf")
    parser.add_argument("--weather-processed", default="data/processed/weather")
    parser.add_argument("--stocknet-processed", default="data/processed/stocknet")
    parser.add_argument("--output-dir", default="outputs/metrics")
    parser.add_argument("--prediction-dir", default="outputs/predictions")
    args = parser.parse_args()

    set_seed(args.seed)
    Path(args.prediction_dir).mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    if args.dataset in {"eeg", "all"}:
        run_eeg_baseline(args, rows)
    if args.dataset in {"weather", "all"}:
        run_weather_baseline(args, rows)
    if args.dataset in {"stocknet", "all"}:
        run_stocknet_baseline(args, rows)
    append_metrics(rows, Path(args.output_dir) / "baselines.csv")
    print(f"Saved {len(rows)} metric rows to {args.output_dir}/baselines.csv")


if __name__ == "__main__":
    main()
