"""Create chronological StockNet windows with leakage-safe target alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PAPER_BOUNDARIES: dict[str, tuple[str, str]] = {
    "train": ("2014-01-01", "2015-08-01"),
    "val": ("2015-08-01", "2015-10-01"),
    "test": ("2015-10-01", "2016-01-01"),
}
FEATURES = ["open", "high", "low", "volume"]


def date_split(date: pd.Timestamp) -> str | None:
    for split, (start, end) in PAPER_BOUNDARIES.items():
        if pd.Timestamp(start) <= date < pd.Timestamp(end):
            return split
    return None


def prepare_stocknet(
    parquet_path: Path,
    output_dir: Path,
    context_len: int = 512,
    features: list[str] | None = None,
) -> dict:
    features = features or FEATURES
    frame = pd.read_parquet(parquet_path).sort_values(
        ["stock_symbol", "date"]
    ).reset_index(drop=True)
    if frame.duplicated(["stock_symbol", "date"]).any():
        raise ValueError("Duplicate (stock_symbol, date) rows found")

    targets = frame["date"]
    train_mask = (targets >= pd.Timestamp(PAPER_BOUNDARIES["train"][0])) & (
        targets < pd.Timestamp(PAPER_BOUNDARIES["train"][1])
    )
    train_values = frame.loc[train_mask, features].to_numpy(dtype=np.float64)
    if train_values.size == 0:
        raise ValueError("No rows in the training date range")

    mean = train_values.mean(axis=0)
    scale = train_values.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    scaled = frame[features].to_numpy(dtype=np.float64)
    scaled = ((scaled - mean) / scale).astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "stocknet_scaler.npz",
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        features=np.array(features),
    )

    labels = frame["label"].to_numpy(dtype=np.int64)
    dates = frame["date"].to_numpy()
    symbols = frame["stock_symbol"].to_numpy()
    split_names = np.array([date_split(pd.Timestamp(value)) for value in dates], dtype=object)

    records: list[dict] = []
    for symbol in np.unique(symbols):
        positions = np.flatnonzero(symbols == symbol)
        for local_index, row_index in enumerate(positions):
            if local_index < context_len:
                continue
            split = split_names[row_index]
            if split is None:
                continue
            history_rows = positions[local_index - context_len : local_index]
            history_end = pd.Timestamp(dates[history_rows[-1]])
            target_date = pd.Timestamp(dates[row_index])
            if history_end >= target_date:
                raise AssertionError(f"Future information in window for {symbol}")
            if not np.all(symbols[history_rows] == symbol):
                raise AssertionError(f"Window crosses symbols for {symbol}")
            records.append(
                {
                    "stock_symbol": str(symbol),
                    "target_date": str(target_date.date()),
                    "split": str(split),
                    "history_start_date": str(pd.Timestamp(dates[history_rows[0]]).date()),
                    "history_end_date": str(history_end.date()),
                    "history_start_row": int(history_rows[0]),
                    "history_end_row": int(history_rows[-1]),
                    "row_index": int(row_index),
                    "_history_rows": history_rows,
                }
            )

    if not records:
        raise ValueError("No usable StockNet windows")
    metadata = pd.DataFrame(records)
    count = len(metadata)
    windows = np.empty((count, context_len, len(features)), dtype=np.float32)
    targets_array = np.empty(count, dtype=np.int64)
    for position, record in enumerate(records):
        windows[position] = scaled[record["_history_rows"]]
        targets_array[position] = labels[record["row_index"]]

    np.save(output_dir / "X.npy", windows)
    np.save(output_dir / "y.npy", targets_array)
    metadata = metadata.drop(columns=["_history_rows"])
    metadata.to_csv(output_dir / "window_metadata.csv", index=False)

    split_counts = metadata["split"].value_counts().to_dict()
    manifest = {
        "source": str(parquet_path),
        "boundaries": {k: list(v) for k, v in PAPER_BOUNDARIES.items()},
        "features": features,
        "excluded_columns": [
            "movement_percent",
            "tweets",
            "num_tweets",
            "company_name",
            "sector",
            "close",
        ],
        "context_len": context_len,
        "feature_order": features,
        "target": "label (sign of same-day movement_percent)",
        "prediction_time": "inputs end on the previous trading day before the target date",
        "history_policy": "history may predate the split boundary but must be strictly before the target date",
        "split_counts": {key: int(value) for key, value in split_counts.items()},
        "scaler": {
            "fit_dates": list(PAPER_BOUNDARIES["train"]),
            "mean": mean.astype(np.float32).tolist(),
            "scale": scale.astype(np.float32).tolist(),
        },
        "window_shape": list(windows.shape),
        "target_shape": list(targets_array.shape),
        "leakage_checks": {
            "history_end_strictly_before_target": bool(
                (pd.to_datetime(metadata["history_end_date"]) < pd.to_datetime(metadata["target_date"])).all()
            ),
            "history_rows_before_target_row": bool(
                (metadata["history_end_row"] < metadata["row_index"]).all()
            ),
            "no_target_outside_declared_splits": bool(
                metadata["split"].isin(PAPER_BOUNDARIES).all()
            ),
        },
    }
    (output_dir / "stocknet_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", default="data/Stocknet/train.parquet")
    parser.add_argument("--output-dir", default="data/processed/stocknet")
    parser.add_argument("--context-len", type=int, default=512)
    args = parser.parse_args()
    manifest = prepare_stocknet(Path(args.parquet), Path(args.output_dir), args.context_len)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
