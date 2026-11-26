"""Feature extraction and simple non-foundation-model predictors."""

from __future__ import annotations

import numpy as np

EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def eeg_spectral_features(windows: np.ndarray, sample_rate: int = 200) -> np.ndarray:
    """Relative band powers plus log total power for [batch, time, channels]."""
    windows = np.asarray(windows, dtype=np.float64)
    n_time, n_channels = windows.shape[1], windows.shape[2]
    spectrum = np.fft.rfft(windows, axis=1)
    power = np.square(np.abs(spectrum)) / max(n_time, 1)
    frequencies = np.fft.rfftfreq(n_time, d=1.0 / sample_rate)

    total_mask = (frequencies >= 0.5) & (frequencies < 45.0)
    total = power[:, total_mask, :].sum(axis=1) + 1e-12
    features = []
    for low, high in EEG_BANDS.values():
        mask = (frequencies >= low) & (frequencies < high)
        band_power = power[:, mask, :].sum(axis=1)
        features.append(band_power / total)
    features.append(np.log1p(total))
    return np.concatenate(features, axis=-1).astype(np.float32)


def stocknet_lagged_features(windows: np.ndarray, recent_steps: int = 5) -> np.ndarray:
    """Flatten recent steps plus window statistics for [batch, time, features]."""
    windows = np.asarray(windows, dtype=np.float32)
    recent = windows[:, -recent_steps:, :].reshape(windows.shape[0], -1)
    statistics = np.concatenate(
        [windows.mean(axis=1), windows.std(axis=1)], axis=1
    )
    return np.concatenate([recent, statistics], axis=1).astype(np.float32)


def persistence_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat the last observed value for every forecast step.

    ``history`` is [batch, channels, context_len].
    """
    history = np.asarray(history, dtype=np.float32)
    last = history[:, :, -1:]
    return np.repeat(last, horizon, axis=2)
