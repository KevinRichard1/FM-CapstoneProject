"""Task metrics for classification and forecasting experiments."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray | None = None
) -> dict[str, float | list[list[int]]]:
    """Compute the primary and supporting classification metrics."""
    labels = np.asarray(labels).astype(int)
    predictions = np.asarray(predictions).astype(int)
    classes = np.unique(np.concatenate([labels, predictions]))
    result: dict[str, float | list[list[int]]] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "per_class_precision": precision_score(
            labels, predictions, average=None, labels=classes, zero_division=0
        ).tolist(),
        "per_class_recall": recall_score(
            labels, predictions, average=None, labels=classes, zero_division=0
        ).tolist(),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=classes).tolist(),
    }
    if probabilities is not None and len(classes) == 2:
        positive_index = int(np.flatnonzero(classes == 1)[0])
        result["roc_auc"] = float(
            roc_auc_score((labels == 1).astype(int), probabilities[:, positive_index])
        )
    return result


def forecasting_metrics(
    targets: np.ndarray, predictions: np.ndarray
) -> dict[str, float | list[float]]:
    """Compute aggregate and per-horizon forecasting errors."""
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if targets.shape != predictions.shape:
        raise ValueError(f"Shape mismatch: {targets.shape} vs {predictions.shape}")

    absolute_errors = np.abs(targets - predictions)
    squared_errors = np.square(targets - predictions)
    per_horizon_mae = absolute_errors.mean(axis=0)
    per_horizon_rmse = np.sqrt(squared_errors.mean(axis=0))
    return {
        "mae": float(absolute_errors.mean()),
        "rmse": float(np.sqrt(squared_errors.mean())),
        "per_horizon_mae": per_horizon_mae.tolist(),
        "per_horizon_rmse": per_horizon_rmse.tolist(),
    }
