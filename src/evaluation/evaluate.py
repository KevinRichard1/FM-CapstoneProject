"""Inference helpers shared by MOMENT experiment runs."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from src.evaluation.metrics import classification_metrics, forecasting_metrics
from src.models.moment_classification import extract_logits
from src.models.moment_forecasting import extract_forecast


def _to_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


@torch.no_grad()
def collect_predictions(
    model,
    loader,
    device: torch.device,
    task: str,
    identifier_keys: Iterable[str] = (),
) -> dict:
    """Run inference once and return predictions, labels, and identifiers."""
    model.eval()
    probabilities: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    identifiers: dict[str, list[np.ndarray]] = defaultdict(list)

    for batch in loader:
        x = batch["x_enc"].to(device, non_blocking=True)
        mask = batch["input_mask"].to(device, non_blocking=True)
        outputs = model(x_enc=x, input_mask=mask)

        if task == "classification":
            logits = extract_logits(outputs).float()
            probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
            predictions.append(logits.argmax(dim=-1).cpu().numpy())
            labels.append(_to_numpy(batch["label"]))
        elif task == "forecasting":
            predictions.append(extract_forecast(outputs).float().cpu().numpy())
            targets.append(_to_numpy(batch["target"]))
        else:
            raise ValueError(f"Unknown task: {task}")

        for key in identifier_keys:
            if key in batch:
                identifiers[key].append(_to_numpy(batch[key]))

    result: dict = {
        "predictions": np.concatenate(predictions),
        "identifiers": {key: np.concatenate(values) for key, values in identifiers.items()},
    }
    if task == "classification":
        result["probabilities"] = np.concatenate(probabilities)
        result["labels"] = np.concatenate(labels)
    else:
        result["targets"] = np.concatenate(targets)
    return result


def aggregate_subject_probabilities(
    probabilities: np.ndarray,
    subjects: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate window probabilities by subject using a predeclared mean rule."""
    n_classes = probabilities.shape[1]
    columns = [f"p_{index}" for index in range(n_classes)]
    frame = pd.DataFrame(probabilities, columns=columns)
    frame["subject_id"] = np.asarray(subjects)
    frame["label"] = np.asarray(labels)
    grouped = frame.groupby("subject_id", sort=True)
    mean_probabilities = grouped[columns].mean().to_numpy()
    subject_labels = grouped["label"].first().to_numpy().astype(int)
    subject_predictions = mean_probabilities.argmax(axis=1).astype(int)
    return subject_predictions, subject_labels, mean_probabilities


def evaluate_predictions(task: str, results: dict) -> dict:
    """Compute task metrics from one inference result dictionary."""
    if task == "classification":
        return classification_metrics(
            results["labels"], results["predictions"], results["probabilities"]
        )
    return forecasting_metrics(results["targets"], results["predictions"])


def save_classification_predictions(
    path,
    results: dict,
    extra_columns: dict[str, np.ndarray] | None = None,
) -> None:
    probabilities = results["probabilities"]
    data = {
        "label": results["labels"],
        "prediction": results["predictions"],
    }
    for index in range(probabilities.shape[1]):
        data[f"p_{index}"] = probabilities[:, index]
    data.update(results["identifiers"])
    if extra_columns:
        data.update(extra_columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(path, index=False)


def save_forecast_predictions(path, results: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"predictions": results["predictions"], "targets": results["targets"]}
    payload.update(results["identifiers"])
    np.savez_compressed(path, **payload)
