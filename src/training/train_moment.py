"""Train MOMENT linear-probing and LoRA experiments for all three tasks."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.data.inspect_datasets import load_weather_dataset
from src.datasets.eeg_dataset import EEGDataset
from src.datasets.stocknet_dataset import StockNetDataset
from src.datasets.weather_dataset import WeatherDataset
from src.evaluation.efficiency import (
    checkpoint_size_mb,
    measure_inference,
    peak_gpu_memory_mb,
    reset_peak_memory,
)
from src.evaluation.evaluate import (
    aggregate_subject_probabilities,
    collect_predictions,
    evaluate_predictions,
    save_classification_predictions,
    save_forecast_predictions,
)
from src.models.moment_classification import build_moment_classifier, resolve_device
from src.models.moment_forecasting import build_moment_forecaster
from src.training.adapters import attach_lora, freeze_for_linear_probing, parameter_counts
from src.utils.logging import get_logger
from src.utils.reproducibility import make_generator, set_seed

DATASET_SETTINGS = {
    "eeg": {"task": "classification", "num_class": 3, "channels": 19},
    "weather": {"task": "forecasting", "channels": 1},
    "stocknet": {"task": "classification", "num_class": 2, "channels": 4},
}


def append_metric_rows(rows: list[dict], path: Path) -> None:
    frame = pd.DataFrame(rows)
    keys = ["dataset", "model", "split", "aggregation", "metric", "seed", "config"]
    if path.exists():
        existing = pd.read_csv(path)
        overlap = existing.set_index(keys).index.isin(frame.set_index(keys).index)
        frame = pd.concat([existing[~overlap], frame], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def record_result(
    rows: list[dict],
    dataset_name: str,
    model_name: str,
    split: str,
    aggregation: str,
    metrics: dict,
    config: dict,
    seed: int,
) -> None:
    serialized = json.dumps(config, sort_keys=True, default=str)
    for metric, value in metrics.items():
        rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "split": split,
                "aggregation": aggregation,
                "metric": metric,
                "value": json.dumps(value) if isinstance(value, list) else value,
                "seed": seed,
                "config": serialized,
            }
        )


def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    limit: int | None = None,
) -> DataLoader:
    indices = np.arange(len(dataset))
    if limit is not None and limit < len(dataset):
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        indices = indices[:limit]
    subset = Subset(dataset, indices.tolist())
    generator = make_generator(seed, device="cpu")
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
        drop_last=False,
    )


def weather_horizon(args: argparse.Namespace) -> int:
    manifest = json.loads(
        (Path(args.weather_processed) / "weather_manifest.json").read_text(encoding="utf-8")
    )
    return int(manifest["forecast_horizon"])


def stocknet_channels(args: argparse.Namespace) -> int:
    manifest = json.loads(
        (Path(args.stocknet_processed) / "stocknet_manifest.json").read_text(encoding="utf-8")
    )
    return len(manifest["features"])


def build_loaders(args: argparse.Namespace, dataset_name: str) -> dict[str, DataLoader]:
    train_limit = args.train_limit if dataset_name != "weather" or args.train_limit else None
    val_limit = args.val_limit if dataset_name != "weather" or args.val_limit else None
    test_limit = args.test_limit if dataset_name != "weather" or args.test_limit else None

    if dataset_name == "eeg":
        dataset_cls = lambda split: EEGDataset(
            args.eeg_dir, split, args.eeg_manifest, args.eeg_scaler
        )
        identifier_keys = ("subject_id", "window_index")
    elif dataset_name == "weather":
        dataset_cls = lambda split: WeatherDataset(args.weather_processed, split)
        identifier_keys = ("target_index",)
    elif dataset_name == "stocknet":
        dataset_cls = lambda split: StockNetDataset(args.stocknet_processed, split)
        identifier_keys = ("window_index",)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return {
        "train": make_loader(
            dataset_cls("train"), args.batch_size, True, args.seed, train_limit
        ),
        "val": make_loader(
            dataset_cls("val"), args.batch_size, False, args.seed + 1, val_limit
        ),
        "test": make_loader(
            dataset_cls("test"), args.batch_size, False, args.seed + 2, test_limit
        ),
        "_identifier_keys": identifier_keys,  # type: ignore[dict-item]
    }


def build_model(args: argparse.Namespace, dataset_name: str, method: str, device):
    settings = DATASET_SETTINGS[dataset_name]
    channels = stocknet_channels(args) if dataset_name == "stocknet" else settings["channels"]
    common = {
        "checkpoint": args.checkpoint,
        "seq_len": args.seq_len,
        "patch_len": args.patch_len,
        "freeze_encoder": True,
        "freeze_embedder": True,
        "randomly_initialize_backbone": args.random_backbone,
        "device": device,
    }
    if dataset_name == "weather":
        model = build_moment_forecaster(
            forecast_horizon=weather_horizon(args),
            n_channels=channels,
            **common,
        )
    else:
        model = build_moment_classifier(
            num_class=settings["num_class"],
            n_channels=channels,
            reduction="concat",
            **common,
        )

    if method == "linear":
        freeze_for_linear_probing(model)
        counts = parameter_counts(model)
        checkpointing_disabled = False
    elif method == "lora":
        model, counts = attach_lora(
            model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
        checkpointing_disabled = True
    else:
        raise ValueError(f"Unknown method: {method}")
    return model, counts, checkpointing_disabled


def set_module_modes(model, method: str) -> None:
    if method == "linear":
        model.patch_embedding.eval()
        model.encoder.eval()
        model.head.train()


def prepare_batch(batch: dict, task: str, device):
    x = batch["x_enc"].to(device, non_blocking=True)
    mask = batch["input_mask"].to(device, non_blocking=True)
    if task == "classification":
        target = batch["label"].to(device, non_blocking=True).long()
    else:
        target = batch["target"].to(device, non_blocking=True).float()
    return x, mask, target


def compute_loss(model, x, mask, target, task: str) -> torch.Tensor:
    outputs = model(x_enc=x, input_mask=mask)
    if task == "classification":
        from src.models.moment_classification import extract_logits

        return nn.functional.cross_entropy(extract_logits(outputs).float(), target)
    from src.models.moment_forecasting import extract_forecast

    forecast = extract_forecast(outputs).float()
    if forecast.shape != target.shape:
        raise ValueError(f"Forecast {tuple(forecast.shape)} != target {tuple(target.shape)}")
    return nn.functional.mse_loss(forecast, target)


def build_optimizer(
    args: argparse.Namespace, method: str, model: torch.nn.Module
) -> tuple[torch.optim.Optimizer, dict[str, int]]:
    """Use distinct LRs for LoRA adapters and the newly initialized head."""
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if method == "linear":
        return (
            torch.optim.AdamW(
                trainable_params,
                lr=args.lr_linear,
                weight_decay=args.weight_decay,
            ),
            {"head": len(trainable_params)},
        )

    head_ids = {id(parameter) for parameter in model.head.parameters() if parameter.requires_grad}
    head_params = [parameter for parameter in trainable_params if id(parameter) in head_ids]
    adapter_params = [parameter for parameter in trainable_params if id(parameter) not in head_ids]
    if not head_params or not adapter_params:
        raise RuntimeError("LoRA requires both trainable adapter and head parameters")
    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": args.lr_head},
            {"params": adapter_params, "lr": args.lr_lora},
        ],
        weight_decay=args.weight_decay,
    )
    return optimizer, {"head": len(head_params), "adapter": len(adapter_params)}


def evaluate_split(
    model,
    loader: DataLoader,
    device,
    dataset_name: str,
    task: str,
    identifier_keys,
) -> tuple[dict, dict, dict | None]:
    results = collect_predictions(model, loader, device, task, identifier_keys)
    metrics = evaluate_predictions(task, results)
    subject_results = None
    if dataset_name == "eeg":
        predictions, labels, probabilities = aggregate_subject_probabilities(
            results["probabilities"],
            results["identifiers"]["subject_id"],
            results["labels"],
        )
        subject_results = {
            "predictions": predictions,
            "labels": labels,
            "probabilities": probabilities,
        }
        subject_results["metrics"] = evaluate_predictions("classification", subject_results)
    return results, metrics, subject_results


def selection_score(
    dataset_name: str,
    task: str,
    metrics: dict,
    subject_results: dict | None = None,
) -> tuple[str, float]:
    if dataset_name == "eeg":
        if subject_results is None:
            raise ValueError("Subject aggregation is required for EEG model selection")
        return "subject_macro_f1", float(subject_results["metrics"]["macro_f1"])
    if task == "classification":
        return "window_macro_f1", float(metrics["macro_f1"])
    return "window_rmse", -float(metrics["rmse"])


def save_eeg_subject_predictions(
    path: Path,
    identifiers: dict,
    subject_results: dict,
) -> None:
    probabilities = subject_results["probabilities"]
    frame = pd.DataFrame(
        {
            "subject_id": np.sort(np.unique(identifiers["subject_id"])),
            "label": subject_results["labels"],
            "prediction": subject_results["predictions"],
        }
    )
    for index in range(probabilities.shape[1]):
        frame[f"p_{index}"] = probabilities[:, index]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def run_experiment(args: argparse.Namespace, dataset_name: str, method: str) -> dict:
    set_seed(args.seed)
    device = resolve_device(args.device)
    task = DATASET_SETTINGS[dataset_name]["task"]
    tag_suffix = f"_{args.run_tag}" if args.run_tag else ""
    if args.skip_test:
        tag_suffix += "_tune"
    suffix = tag_suffix + ("_random" if args.random_backbone else "")
    model_name = f"moment{suffix}_{'linear_probe' if method == 'linear' else 'lora'}"
    logger = get_logger(
        f"{dataset_name}.{method}{suffix}",
        Path(args.log_dir)
        / f"moment_{dataset_name}_{method}{suffix}_seed{args.seed}.log",
    )
    logger.info("Starting %s on %s (device=%s)", model_name, dataset_name, device)

    loaders = build_loaders(args, dataset_name)
    identifier_keys = loaders.pop("_identifier_keys")
    model, counts, checkpointing_disabled = build_model(args, dataset_name, method, device)
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_params:
        raise RuntimeError(f"No trainable parameters for {dataset_name}/{method}")

    config = {
        "dataset": dataset_name,
        "task": task,
        "method": method,
        "channels": (
            stocknet_channels(args) if dataset_name == "stocknet" else DATASET_SETTINGS[dataset_name]["channels"]
        ),
        "model_name": model_name,
        "checkpoint": args.checkpoint,
        "random_backbone": args.random_backbone,
        "run_tag": args.run_tag,
        "skip_test": args.skip_test,
        "seed": args.seed,
        "device": str(device),
        "batch_size": args.batch_size,
        "learning_rate": (
            {"adapter": args.lr_lora, "head": args.lr_head}
            if method == "lora"
            else {"head": args.lr_linear}
        ),
        "weight_decay": args.weight_decay,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "grad_accumulation": args.grad_accum,
        "grad_clip": args.grad_clip,
        "train_limit": args.train_limit,
        "val_limit": args.val_limit,
        "test_limit": args.test_limit,
        "lora": (
            {"rank": args.lora_rank, "alpha": args.lora_alpha, "dropout": args.lora_dropout}
            if method == "lora"
            else None
        ),
        "gradient_checkpointing_disabled": checkpointing_disabled,
        "parameter_counts": counts,
        "selection": "validation only; test evaluated once after best checkpoint is restored",
    }
    if task == "forecasting":
        config["forecast_horizon"] = weather_horizon(args)
        config["units"] = "standardized"

    optimizer, optimizer_groups = build_optimizer(args, method, model)
    config["optimizer_parameter_tensors"] = optimizer_groups
    checkpoint_path = (
        Path(args.checkpoint_dir)
        / f"{dataset_name}_{method}{suffix}_seed{args.seed}.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    best_score = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    reset_peak_memory()

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        set_module_modes(model, method)
        running_loss = 0.0
        seen = 0
        epoch_started = time.perf_counter()

        for step, batch in enumerate(loaders["train"]):
            x, mask, target = prepare_batch(batch, task, device)
            loss = compute_loss(model, x, mask, target, task)
            (loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(loaders["train"]):
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            batch_size = int(x.shape[0])
            running_loss += float(loss.detach()) * batch_size
            seen += batch_size

        train_loss = running_loss / max(seen, 1)
        _, val_metrics, val_subjects = evaluate_split(
            model, loaders["val"], device, dataset_name, task, identifier_keys
        )
        score_name, score = selection_score(
            dataset_name, task, val_metrics, val_subjects
        )
        epoch_seconds = time.perf_counter() - epoch_started
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_metric": score_name,
                "validation_score": score,
                "epoch_seconds": round(epoch_seconds, 2),
                "peak_gpu_memory_mb": round(peak_gpu_memory_mb(), 2),
            }
        )
        logger.info(
            "epoch=%d loss=%.5f %s=%.5f seconds=%.1f",
            epoch,
            train_loss,
            score_name,
            score,
            epoch_seconds,
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "validation_score": score,
                    "validation_metric": score_name,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                logger.info("Early stopping after %d stale epochs", stale_epochs)
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    config["best_epoch"] = best_epoch
    config["best_validation_metric"] = checkpoint["validation_metric"]
    config["best_validation_score"] = float(checkpoint["validation_score"])
    config["total_training_seconds"] = round(
        sum(entry["epoch_seconds"] for entry in history), 2
    )
    config["peak_gpu_memory_mb"] = round(peak_gpu_memory_mb(), 2)
    config["checkpoint_size_mb"] = round(checkpoint_size_mb(checkpoint_path), 3)

    inference_sample = next(iter(loaders["train"]))
    config["inference"] = measure_inference(
        model,
        inference_sample["x_enc"],
        inference_sample["input_mask"],
        device,
    )

    prediction_dir = Path(args.prediction_dir)
    rows: list[dict] = []
    final_results: dict[str, dict] = {}
    final_subjects: dict[str, dict | None] = {}
    splits = ("train", "val") if args.skip_test else ("train", "val", "test")
    for split in splits:
        results, metrics, subject_results = evaluate_split(
            model, loaders[split], device, dataset_name, task, identifier_keys
        )
        final_results[split] = results
        final_subjects[split] = subject_results
        record_result(rows, dataset_name, model_name, split, "window", metrics, config, args.seed)
        if dataset_name == "eeg" and subject_results is not None:
            record_result(
                rows,
                dataset_name,
                model_name,
                split,
                "subject",
                subject_results["metrics"],
                config,
                args.seed,
            )

        prefix = f"{dataset_name}_{method}{suffix}_seed{args.seed}_{split}"
        if task == "classification":
            save_classification_predictions(
                prediction_dir / f"{prefix}.csv", results
            )
            if dataset_name == "eeg" and subject_results is not None:
                save_eeg_subject_predictions(
                    prediction_dir / f"{prefix}_subject.csv",
                    results["identifiers"],
                    subject_results,
                )
        else:
            save_forecast_predictions(prediction_dir / f"{prefix}.npz", results)

    metric_path = Path(args.output_dir) / "moment.csv"
    append_metric_rows(rows, metric_path)
    history_path = (
        Path(args.output_dir)
        / f"moment_history_{dataset_name}_{method}{suffix}_seed{args.seed}.json"
    )
    history_path.write_text(
        json.dumps({"config": config, "epochs": history}, indent=2, default=str),
        encoding="utf-8",
    )
    config_path = (
        Path(args.output_dir)
        / f"moment_config_{dataset_name}_{method}{suffix}_seed{args.seed}.json"
    )
    config_path.write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    if args.skip_test:
        if task == "classification":
            val_metrics = evaluate_predictions("classification", final_results["val"])
            primary = f"validation_macro_f1={val_metrics['macro_f1']:.4f}"
        else:
            val_metrics = evaluate_predictions("forecasting", final_results["val"])
            primary = f"validation_rmse={val_metrics['rmse']:.4f}"
    elif task == "classification":
        test_metrics = evaluate_predictions("classification", final_results["test"])
        primary = f"test_macro_f1={test_metrics['macro_f1']:.4f}"
    else:
        test_metrics = evaluate_predictions("forecasting", final_results["test"])
        primary = f"test_rmse={test_metrics['rmse']:.4f}"
    logger.info("Best epoch=%d; %s", best_epoch, primary)
    return {
        "dataset": dataset_name,
        "method": method,
        "model_name": model_name,
        "random_backbone": args.random_backbone,
        "best_epoch": best_epoch,
        "config": config_path.as_posix(),
        "history": history_path.as_posix(),
        "metrics": metric_path.as_posix(),
        "primary": primary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["eeg", "weather", "stocknet", "all"], default="all")
    parser.add_argument("--methods", nargs="+", choices=["linear", "lora"], default=["linear", "lora"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--random-backbone",
        action="store_true",
        help="Replace pretrained backbone weights with random initialization (ablation only).",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Suffix appended to output names so reruns do not overwrite prior results.",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Tuning mode: evaluate train/validation only and never touch the test split.",
    )
    parser.add_argument("--checkpoint", default="AutonLab/MOMENT-1-small")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--patch-len", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--lr-linear", type=float, default=1e-3, help="Learning rate for linear-probe head."
    )
    parser.add_argument(
        "--lr-head",
        type=float,
        default=1e-3,
        help="Learning rate for the task head during LoRA adaptation.",
    )
    parser.add_argument(
        "--lr-lora", type=float, default=1e-4, help="Learning rate for LoRA adapter parameters."
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--eeg-dir", default="data/ADFTD-RS")
    parser.add_argument("--eeg-manifest", default="data/manifests/eeg_subject_split.csv")
    parser.add_argument("--eeg-scaler", default="data/processed/eeg_scaler.npz")
    parser.add_argument("--weather-processed", default="data/processed/weather")
    parser.add_argument("--stocknet-processed", default="data/processed/stocknet")
    parser.add_argument("--output-dir", default="outputs/metrics")
    parser.add_argument("--prediction-dir", default="outputs/predictions")
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    parser.add_argument("--log-dir", default="outputs/logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    datasets = ["eeg", "weather", "stocknet"] if args.dataset == "all" else [args.dataset]
    summaries = []
    for dataset_name in datasets:
        for method in args.methods:
            summaries.append(run_experiment(args, dataset_name, method))
    print(json.dumps(summaries, indent=2))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
