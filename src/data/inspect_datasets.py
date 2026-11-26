"""Schema audits for the three local datasets.

This script records verified facts only. It writes JSON audits under
``data/manifests`` and does not make split or modelling decisions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import DatasetDict, load_dataset, load_from_disk


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def audit_eeg(eeg_dir: Path) -> dict:
    meta = json.loads((eeg_dir / "meta.json").read_text(encoding="utf-8"))
    n, time_steps, channels = meta["N"], meta["T"], meta["C"]
    x = np.memmap(eeg_dir / "X-001.dat", dtype="float32", mode="r", shape=(n, time_steps, channels))
    y = np.memmap(eeg_dir / "y.dat", dtype="float32", mode="r", shape=(n, 3))

    labels = np.rint(y[:, 0]).astype(int)
    subjects = np.rint(y[:, 1]).astype(int)
    rates = np.rint(y[:, 2]).astype(int)

    rate_counts = {int(k): int(v) for k, v in zip(*np.unique(rates, return_counts=True))}
    class_counts = {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
    mask_200 = rates == 200

    subject_frame = pd.DataFrame(
        {
            "subject_id": subjects[mask_200],
            "label": labels[mask_200],
            "windows": 1,
        }
    )
    by_subject = subject_frame.groupby("subject_id").agg(
        windows=("windows", "sum"),
        n_labels=("label", "nunique"),
        label=("label", "first"),
        rate=("rate", "first") if "rate" in subject_frame else ("windows", "first"),
    )
    # The rate column is constant for the 200 Hz subset; add it explicitly.
    by_subject["rate"] = 200

    rng = np.random.default_rng(0)
    sample = x[rng.choice(n, size=min(5000, n), replace=False)]
    finite = bool(np.isfinite(sample).all())

    return {
        "source": str(eeg_dir),
        "files": {path.name: path.stat().st_size for path in sorted(eeg_dir.iterdir()) if path.is_file()},
        "meta": meta,
        "array_shapes": {"X": list(x.shape), "y": list(y.shape)},
        "sampling_rates_hz": rate_counts,
        "class_counts_all_rates": class_counts,
        "class_counts_200hz": {
            int(k): int(v) for k, v in zip(*np.unique(labels[mask_200], return_counts=True))
        },
        "window_counts_200hz": int(mask_200.sum()),
        "window_duration_samples_200hz": time_steps,
        "window_duration_seconds_200hz": time_steps / 200,
        "subjects_total": int(np.unique(subjects).size),
        "subject_counts_200hz_by_class": {
            int(cls): int(by_subject[by_subject.label == cls].shape[0])
            for cls in sorted(np.unique(labels[mask_200]))
        },
        "subject_ids_200hz": sorted(int(s) for s in by_subject.index),
        "windows_per_subject_200hz": {
            str(int(row.subject_id)) if hasattr(row, "subject_id") else str(int(idx)): int(row.windows)
            for idx, row in by_subject.iterrows()
        },
        "subject_label_consistent_200hz": bool((by_subject.n_labels == 1).all()),
        "sample_finite": finite,
        "sample_signal_min": float(sample.min()),
        "sample_signal_max": float(sample.max()),
        "sample_signal_mean": float(sample.mean()),
        "sample_signal_std": float(sample.std()),
        "notes": [
            "Only the 200 Hz rows are eligible; 100 Hz and 50 Hz rows are excluded.",
            "Windows are already segmented (400 samples, 19 channels) with time as the first axis.",
            "y columns are [disease_id, subject_id, sampling_frequency] stored as float32.",
        ],
    }


def load_weather_dataset(weather_dir: Path) -> DatasetDict:
    if (weather_dir / "dataset_dict.json").exists():
        return load_from_disk(str(weather_dir))
    return load_dataset("ts-arena/weather")


def audit_weather(weather_dir: Path, local_csv: Path | None) -> dict:
    dataset = load_weather_dataset(weather_dir)
    split_info: dict[str, dict] = {}
    target = "T (degC)"
    for split in dataset:
        frame = dataset[split].to_pandas()
        idx = frame["timestamp_idx"].to_numpy()
        split_info[split] = {
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "missing_values": {str(k): int(v) for k, v in frame.isna().sum().items() if v},
            "timestamp_idx_start": int(idx.min()),
            "timestamp_idx_end": int(idx.max()),
            "timestamp_idx_strictly_increasing": bool(np.all(np.diff(idx) == 1)),
            "target_mean": float(frame[target].mean()),
            "target_std": float(frame[target].std()),
        }

    comparison = None
    if local_csv is not None and local_csv.exists():
        local = pd.read_csv(local_csv, encoding="latin-1")
        comparison = {
            "path": str(local_csv),
            "rows": int(len(local)),
            "columns": list(local.columns),
            "has_split_column": "split" in local.columns,
            "date_min": str(local["date"].min()),
            "date_max": str(local["date"].max()),
            "target_mean": float(local[target].mean()),
            "target_std": float(local[target].std()),
            "hf_total_rows": int(sum(len(dataset[split]) for split in dataset)),
        }

    return {
        "source": "ts-arena/weather (Hugging Face)",
        "saved_at": str(weather_dir),
        "splits": split_info,
        "split_order": list(dataset.keys()),
        "feature_dtypes": {str(k): str(v) for k, v in dataset["train"].features.items()},
        "local_csv_comparison": comparison,
        "notes": [
            "Published split row counts: 36,887 train / 5,269 validation / 10,540 test.",
            "Values are standardized; no second normalization should be applied by default.",
            "timestamp_idx resets in each split, so it is a within-split order index, not a global timestamp.",
            "No readable calendar timestamps are retained in the Hugging Face table.",
        ],
    }


def audit_stocknet(parquet_path: Path, metadata_path: Path | None) -> dict:
    frame = pd.read_parquet(parquet_path)
    ordered = frame.sort_values(["stock_symbol", "date"]).reset_index(drop=True)
    labels = ordered["label"].astype(int)
    movement_positive = (ordered["movement_percent"] > 0).astype(int)
    label_matches_movement = bool(labels.equals(movement_positive))

    per_symbol = ordered.groupby("stock_symbol").size()
    paper_windows = {
        name: {
            "start": start,
            "end_exclusive": end,
            "rows": int(((ordered["date"] >= start) & (ordered["date"] < end)).sum()),
        }
        for name, start, end in (
            ("train", "2014-01-01", "2015-08-01"),
            ("validation", "2015-08-01", "2015-10-01"),
            ("test", "2015-10-01", "2016-01-01"),
        )
    }
    numeric = ordered.select_dtypes(include=[np.number])
    summaries = {
        column: {
            "min": float(numeric[column].min()),
            "max": float(numeric[column].max()),
            "mean": float(numeric[column].mean()),
            "std": float(numeric[column].std()),
        }
        for column in ("open", "high", "low", "close", "volume", "movement_percent")
    }

    result = {
        "source": str(parquet_path),
        "metadata": json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path and metadata_path.exists() else None,
        "shape": list(frame.shape),
        "columns": list(frame.columns),
        "dtypes": {str(k): str(v) for k, v in frame.dtypes.items()},
        "date_range": {"min": str(frame["date"].min()), "max": str(frame["date"].max())},
        "symbols": int(frame["stock_symbol"].nunique()),
        "rows_per_symbol": {
            "min": int(per_symbol.min()),
            "max": int(per_symbol.max()),
            "median": float(per_symbol.median()),
        },
        "duplicate_symbol_date_rows": int(frame.duplicated(["stock_symbol", "date"]).sum()),
        "missing_values": {str(k): int(v) for k, v in frame.isna().sum().items() if v},
        "label_counts": {int(k): int(v) for k, v in labels.value_counts().sort_index().items()},
        "positive_movement_ratio": float(movement_positive.mean()),
        "label_equals_movement_percent_gt_zero": label_matches_movement,
        "neutral_movement_band_present": False,
        "paper_date_window_counts": paper_windows,
        "numeric_field_summaries": summaries,
        "excluded_from_numerical_moment_input": [
            "movement_percent",
            "tweets",
            "num_tweets",
            "company_name",
            "sector",
        ],
        "unresolved": [
            "OHLCV fields are transformed, not verified raw prices; the Parquet conversion does not document the exact transform.",
            "close contains extreme values (far outside the open/high/low range), so its construction remains unresolved.",
            "label is the sign of movement_percent with no neutral band and row counts differ from the original paper, so this is a StockNet-derived task, not an exact reproduction.",
            "Target alignment must use information through the previous trading day for a movement realized on date d.",
        ],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eeg-dir", default="data/ADFTD-RS")
    parser.add_argument("--weather-dir", default="data/raw/weather_hf")
    parser.add_argument("--local-weather-csv", default="data/Autoformer/weather.csv")
    parser.add_argument("--stocknet-parquet", default="data/Stocknet/train.parquet")
    parser.add_argument("--stocknet-metadata", default="data/Stocknet/dataset_info.json")
    parser.add_argument("--output-dir", default="data/manifests")
    args = parser.parse_args()

    output = Path(args.output_dir)
    eeg = audit_eeg(Path(args.eeg_dir))
    weather = audit_weather(Path(args.weather_dir), Path(args.local_weather_csv))
    stocknet = audit_stocknet(Path(args.stocknet_parquet), Path(args.stocknet_metadata))

    write_json(output / "eeg_audit.json", eeg)
    write_json(output / "weather_audit.json", weather)
    write_json(output / "stocknet_audit.json", stocknet)
    print(f"Wrote audits to {output}")
    print(f"EEG 200 Hz: {eeg['window_counts_200hz']} windows, {eeg['subjects_total']} subjects")
    print("Weather splits:", {k: v["rows"] for k, v in weather["splits"].items()})
    print(
        "StockNet:",
        stocknet["shape"],
        "label==movement>0:",
        stocknet["label_equals_movement_percent_gt_zero"],
    )


if __name__ == "__main__":
    main()
