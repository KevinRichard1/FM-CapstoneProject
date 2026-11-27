"""Build the final comparison table and summary figures from saved outputs."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

SCALAR_METRICS = {
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "mcc",
    "roc_auc",
    "mae",
    "rmse",
}
PER_HORIZON_METRICS = {"per_horizon_mae", "per_horizon_rmse"}
HEADLINE_MODELS = {
    "eeg": "moment_headlr1e3_lora",
    "weather": "moment_lora",
    "stocknet": "moment_linear_probe",
}
PRIMARY_METRICS = {
    "eeg": ("subject", "macro_f1"),
    "weather": ("window", "rmse"),
    "stocknet": ("window", "macro_f1"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, default=Path("outputs/metrics"))
    parser.add_argument("--figure-dir", type=Path, default=Path("outputs/figures"))
    return parser.parse_args()


def classify_model(dataset: str, model: str) -> str:
    if model == HEADLINE_MODELS[dataset]:
        return "headline"
    if "random" in model:
        return "random_backbone_ablation"
    if any(tag in model for tag in ("headlr", "snet_", "wx_lora", "feat_", "ctx")):
        return "tuning_variant"
    return "standard_moment"


def load_history_lookup(metrics_dir: Path) -> dict[tuple[str, str, int], list[dict]]:
    lookup: dict[tuple[str, str, int], list[dict]] = {}
    for path in metrics_dir.glob("moment_history_*_seed*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload["config"]
        key = (config["dataset"], config["model_name"], int(config["seed"]))
        lookup[key] = payload["epochs"]
    return lookup


def efficiency_fields(config: dict | None, history: list[dict] | None = None) -> dict:
    if not config:
        return {
            "total_parameters": None,
            "trainable_parameters": None,
            "total_training_seconds": None,
            "mean_epoch_seconds": None,
            "peak_gpu_memory_mb": None,
            "inference_mean_batch_ms": None,
            "inference_samples_per_second": None,
            "checkpoint_size_mb": None,
            "batch_size": None,
            "learning_rate": None,
            "class_weight": None,
            "best_epoch": None,
        }
    counts = config.get("parameter_counts", {})
    inference = config.get("inference", {})
    return {
        "total_parameters": counts.get("total"),
        "trainable_parameters": counts.get("trainable"),
        "total_training_seconds": config.get("total_training_seconds"),
        "mean_epoch_seconds": (
            round(
                sum(epoch["epoch_seconds"] for epoch in history) / len(history),
                2,
            )
            if history
            else None
        ),
        "peak_gpu_memory_mb": config.get("peak_gpu_memory_mb"),
        "inference_mean_batch_ms": inference.get("mean_batch_ms"),
        "inference_samples_per_second": inference.get("samples_per_second"),
        "checkpoint_size_mb": config.get("checkpoint_size_mb"),
        "batch_size": config.get("batch_size"),
        "learning_rate": json.dumps(config.get("learning_rate"), sort_keys=True),
        "class_weight": json.dumps(config.get("class_weight"), sort_keys=True)
        if config.get("class_weight") is not None
        else None,
        "best_epoch": config.get("best_epoch"),
    }


def reference_rows() -> list[dict]:
    references = []
    for dataset, num_classes, aggregation in (("eeg", 3, "subject"), ("stocknet", 2, "window")):
        references.append(
            {
                "dataset": dataset,
                "model": "chance_reference",
                "source": "reference",
                "group": "reference",
                "is_headline": False,
                "split": "test",
                "aggregation": aggregation,
                "metric": "macro_f1",
                "value": 1.0 / num_classes,
                "seed": 42,
                "train_limit": None,
                "val_limit": None,
                "test_limit": None,
                "primary_metric": True,
                "total_parameters": None,
                "trainable_parameters": None,
                "total_training_seconds": None,
                "mean_epoch_seconds": None,
                "peak_gpu_memory_mb": None,
                "inference_mean_batch_ms": None,
                "inference_samples_per_second": None,
                "checkpoint_size_mb": None,
                "batch_size": None,
                "learning_rate": None,
                "best_epoch": None,
                "horizon": None,
            }
        )
    return references


def flatten_numeric(value) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        flattened: list[float] = []
        for item in value:
            flattened.extend(flatten_numeric(item))
        return flattened
    raise TypeError(f"Expected numeric or nested numeric list, got {type(value)}")


def build_table(metrics_dir: Path) -> pd.DataFrame:
    baselines = pd.read_csv(metrics_dir / "baselines.csv")
    moment = pd.read_csv(metrics_dir / "moment.csv")
    histories = load_history_lookup(metrics_dir)
    for frame in (baselines, moment):
        frame["raw_value"] = frame["value"].astype(str)
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame["config_json"] = frame["config"].apply(json.loads)

    rows: list[dict] = []
    for source, frame in (("baseline", baselines), ("moment", moment)):
        for _, row in frame.iterrows():
            metric = row["metric"]
            if row["split"] != "test" or (
                metric not in SCALAR_METRICS and metric not in PER_HORIZON_METRICS
            ):
                continue
            config = row["config_json"]
            group = (
                "baseline"
                if source == "baseline"
                else classify_model(row["dataset"], row["model"])
            )
            history = (
                histories.get((row["dataset"], row["model"], int(row["seed"])))
                if source == "moment"
                else None
            )
            efficiency = efficiency_fields(config if source == "moment" else None, history)
            base = {
                "dataset": row["dataset"],
                "model": row["model"],
                "source": source,
                "group": group,
                "is_headline": group == "headline",
                "split": row["split"],
                "aggregation": row["aggregation"],
                "metric": metric,
                "value": row["value"],
                "seed": row["seed"],
                "train_limit": config.get("train_limit"),
                "val_limit": config.get("val_limit"),
                "test_limit": config.get("test_limit"),
                "primary_metric": (
                    row["aggregation"] == PRIMARY_METRICS[row["dataset"]][0]
                    and metric == PRIMARY_METRICS[row["dataset"]][1]
                ),
                **efficiency,
                "horizon": None,
            }
            if metric in PER_HORIZON_METRICS:
                for horizon, value in enumerate(flatten_numeric(json.loads(row["raw_value"])), start=1):
                    expanded = base.copy()
                    expanded["metric"] = f"{metric}_h{horizon:03d}"
                    expanded["value"] = float(value)
                    expanded["horizon"] = horizon
                    rows.append(expanded)
                continue
            rows.append(base)
    rows.extend(reference_rows())
    return pd.DataFrame(rows).sort_values(
        ["dataset", "group", "model", "aggregation", "metric"]
    )


def save_environment(metrics_dir: Path) -> None:
    environment = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "precision": "float32",
    }
    (metrics_dir / "final_environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )


def format_metric(value) -> str:
    return "n/a" if pd.isna(value) else f"{float(value):.4f}"


def format_count(value) -> str:
    return "n/a" if pd.isna(value) else f"{int(value):,}"


def save_summary(table: pd.DataFrame, metrics_dir: Path) -> None:
    primary = table[table["primary_metric"]].copy()
    headline = table[table["is_headline"]].drop_duplicates(subset=["dataset", "model"])
    lines = [
        "# Final evaluation and efficiency summary",
        "",
        "Generated by `src/evaluation/build_final_comparison.py` from saved metrics and configs.",
        "",
        "## Primary test metrics",
        "",
    ]
    for dataset, (aggregation, metric) in PRIMARY_METRICS.items():
        lines.extend(
            [
                f"### {dataset} ({aggregation}-level {metric})",
                "",
                "| Model | Group | Value |",
                "|---|---|---:|",
            ]
        )
        subset = primary[primary["dataset"] == dataset]
        subset = subset[subset["group"] != "tuning_variant"]
        subset = subset.drop_duplicates(subset=["model", "value"])
        for _, row in subset.sort_values(["group", "model"]).iterrows():
            lines.append(
                f"| {row['model']} | {row['group']} | {format_metric(row['value'])} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Headline efficiency",
            "",
            "| Dataset | Model | Total params | Trainable params | Total train s | Mean epoch s | Peak GPU MB | Inference ms/batch | Checkpoint MB |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in headline.sort_values("dataset").iterrows():
        lines.append(
            "| {dataset} | {model} | {total} | {trainable} | {total_s} | {epoch_s} | {memory} | {latency} | {checkpoint} |".format(
                dataset=row["dataset"],
                model=row["model"],
                total=format_count(row["total_parameters"]),
                trainable=format_count(row["trainable_parameters"]),
                total_s=format_metric(row["total_training_seconds"]),
                epoch_s=format_metric(row["mean_epoch_seconds"]),
                memory=format_metric(row["peak_gpu_memory_mb"]),
                latency=format_metric(row["inference_mean_batch_ms"]),
                checkpoint=format_metric(row["checkpoint_size_mb"]),
            )
        )

    lines.extend(
        [
            "",
            "## Notes and limitations",
            "",
            "- Headline models are selected on validation data only; test metrics are reported once per selected run.",
            "- Only seed 42 was run, so no mean/std or confidence intervals are reported.",
            "- MOMENT test sets were capped, while non-FM baseline test sets were not; cross-method comparisons are indicative rather than strictly matched.",
            "- Weather errors are in standardized units; no readable calendar dates or recoverable scaler were available for physical-unit conversion.",
            "- StockNet is HF-derived and numerical-only. It is not an exact reproduction of the StockNet paper, and per-ticker test breakdowns are omitted because MOMENT used a capped random subset.",
            "- Batch-1 latency was not measured separately; inference cost is reported as batched throughput.",
            "- Parameter, timing, memory, and checkpoint data are unavailable from saved scikit-learn baseline objects.",
            "- Validation-selected StockNet LR and engineered-feature tuning did not improve test results and are retained as negative ablations.",
        ]
    )
    (metrics_dir / "final_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def save_figures(table: pd.DataFrame, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    primary = table[table["primary_metric"]].copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, dataset in zip(axes, ("eeg", "weather", "stocknet")):
        subset = primary[primary["dataset"] == dataset].drop_duplicates(
            subset=["model", "value"]
        )
        axis.barh(subset["model"], subset["value"], color="#4C78A8")
        axis.set_title(dataset)
        axis.set_xlabel(PRIMARY_METRICS[dataset][1])
        axis.tick_params(axis="y", labelsize=7)
    fig.suptitle("Primary test metrics (lower RMSE is better)")
    fig.tight_layout()
    fig.savefig(figure_dir / "final_primary_metrics.png", dpi=150)
    plt.close(fig)

    efficiency = table[table["is_headline"]].drop_duplicates(subset=["model"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(efficiency["model"], efficiency["peak_gpu_memory_mb"], color="#F58518")
    axes[0].set_title("Peak GPU memory (MB)")
    axes[1].bar(efficiency["model"], efficiency["total_training_seconds"], color="#54A24B")
    axes[1].set_title("Total training seconds")
    for axis in axes:
        axis.tick_params(axis="x", rotation=30, labelsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "final_efficiency.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    table = build_table(args.metrics_dir)
    output_path = args.metrics_dir / "final_comparison.csv"
    table.to_csv(output_path, index=False)
    save_environment(args.metrics_dir)
    save_summary(table, args.metrics_dir)
    save_figures(table, args.figure_dir)
    print(f"Wrote {len(table)} rows to {output_path}")


if __name__ == "__main__":
    main()
