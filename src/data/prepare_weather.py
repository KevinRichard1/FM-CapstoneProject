"""Save weather target arrays and the chronological window manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data.inspect_datasets import load_weather_dataset


def prepare_weather(
    weather_dir: Path, output_dir: Path, target: str, context_len: int, horizon: int
) -> dict:
    dataset = load_weather_dataset(weather_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_info: dict[str, dict] = {}

    for split in dataset:
        values = np.asarray(dataset[split][target], dtype=np.float32)
        if values.ndim != 1:
            raise ValueError(f"Expected a 1D target series, got {values.shape}")
        if values.size <= context_len + horizon:
            raise ValueError(f"Split {split} is too short for the requested windows")
        destination = output_dir / f"{split}_target.npy"
        np.save(destination, values)
        window_count = values.size - context_len - horizon + 1
        split_info[split] = {
            "rows": int(values.size),
            "first_target_index": int(context_len),
            "last_target_index": int(values.size - horizon),
            "window_count": int(window_count),
            "array": str(destination),
        }

    manifest = {
        "source": "ts-arena/weather",
        "target": target,
        "context_len": context_len,
        "forecast_horizon": horizon,
        "standardized": True,
        "rescale_applied": False,
        "window_rule": "input=[t-context:t], target=[t:t+horizon] within the same split",
        "splits": split_info,
    }
    (output_dir / "weather_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather-dir", default="data/raw/weather_hf")
    parser.add_argument("--output-dir", default="data/processed/weather")
    parser.add_argument("--target", default="T (degC)")
    parser.add_argument("--context-len", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=144)
    args = parser.parse_args()

    manifest = prepare_weather(
        Path(args.weather_dir), Path(args.output_dir), args.target, args.context_len, args.horizon
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
