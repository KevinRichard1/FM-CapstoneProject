"""Builders and helpers for MOMENT forecasting experiments."""

from __future__ import annotations

import torch
from momentfm import MOMENTPipeline

from src.models.moment_classification import resolve_device


def build_moment_forecaster(
    forecast_horizon: int,
    n_channels: int = 1,
    checkpoint: str = "AutonLab/MOMENT-1-small",
    seq_len: int = 512,
    patch_len: int = 8,
    freeze_encoder: bool = True,
    freeze_embedder: bool = True,
    device: str | torch.device = "auto",
) -> MOMENTPipeline:
    """Load MOMENT-1-Small with a newly initialized forecasting head."""
    model_kwargs = {
        "task_name": "forecasting",
        "n_channels": n_channels,
        "forecast_horizon": forecast_horizon,
        "seq_len": seq_len,
        "patch_len": patch_len,
        "freeze_encoder": freeze_encoder,
        "freeze_embedder": freeze_embedder,
    }
    model = MOMENTPipeline.from_pretrained(checkpoint, model_kwargs=model_kwargs)
    model.init()
    model.to(resolve_device(device))
    return model


def extract_forecast(outputs) -> torch.Tensor:
    """Return forecast values from a TimeseriesOutputs object."""
    forecast = outputs.forecast
    if not torch.is_tensor(forecast):
        forecast = torch.as_tensor(forecast)
    return forecast
