"""Smoke tests for MOMENT-1-Small classification and forecasting heads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.models.moment_classification import build_moment_classifier, extract_logits
from src.models.moment_forecasting import build_moment_forecaster, extract_forecast


def model_config_dict(model) -> dict:
    """Serialize the effective model configuration and head information."""
    config = model.config
    data = dict(config.__dict__) if hasattr(config, "__dict__") else dict(config)
    data["effective_task_name"] = model.task_name
    data["head_class"] = type(model.head).__name__
    data["head_nf"] = getattr(model, "head_nf", None)
    return data


def parameter_counts(model) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total": total, "trainable": trainable}


def tensor_summary(name: str, tensor: torch.Tensor, expected_shape: list[int]) -> dict:
    return {
        "name": name,
        "shape": list(tensor.shape),
        "expected_shape": expected_shape,
        "shape_ok": list(tensor.shape) == expected_shape,
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "all_finite": bool(torch.isfinite(tensor).all().item()),
    }


def run_classification(
    device: torch.device, checkpoint: str, seq_len: int
) -> tuple[dict, dict]:
    model = build_moment_classifier(
        num_class=2,
        n_channels=1,
        checkpoint=checkpoint,
        seq_len=seq_len,
        patch_len=8,
        device=device,
    )
    model.eval()
    x = torch.randn(2, 1, seq_len, dtype=torch.float32, device=device)
    ones_mask = torch.ones(2, seq_len, dtype=torch.float32, device=device)
    partial_mask = ones_mask.clone()
    padded_steps = seq_len - 400
    partial_mask[:, -padded_steps:] = 0.0

    with torch.no_grad():
        out_no_mask = model(x_enc=x)
        out_ones = model(x_enc=x, input_mask=ones_mask)
        out_partial = model(x_enc=x, input_mask=partial_mask)

    logits = extract_logits(out_ones)
    embeddings = out_ones.embeddings
    result = {
        "task": "classification",
        "input_shape": list(x.shape),
        "parameter_counts": parameter_counts(model),
        "outputs": {
            "no_mask_logits": tensor_summary(
                "logits_no_mask", extract_logits(out_no_mask), [2, 2]
            ),
            "ones_mask_logits": tensor_summary("logits", logits, [2, 2]),
            "partial_mask_logits": tensor_summary(
                "logits_partial_mask", extract_logits(out_partial), [2, 2]
            ),
            "embeddings": tensor_summary(
                "embeddings", embeddings, [2, seq_len // 8, 512]
            ),
        },
        "prediction_extraction": "outputs.logits -> torch.Tensor [batch, num_class]",
        "mask_behavior": (
            "input_mask is optional; explicit ones and partial zeros were accepted. "
            "Zero positions suppress patches through the T5 attention mask."
        ),
    }
    return result, model_config_dict(model)


def run_forecasting(
    device: torch.device, checkpoint: str, seq_len: int, horizon: int
) -> tuple[dict, dict]:
    model = build_moment_forecaster(
        forecast_horizon=horizon,
        n_channels=1,
        checkpoint=checkpoint,
        seq_len=seq_len,
        patch_len=8,
        device=device,
    )
    model.eval()
    x = torch.randn(2, 1, seq_len, dtype=torch.float32, device=device)
    ones_mask = torch.ones(2, seq_len, dtype=torch.float32, device=device)
    partial_mask = ones_mask.clone()
    padded_steps = seq_len - 400
    partial_mask[:, -padded_steps:] = 0.0

    with torch.no_grad():
        out_no_mask = model(x_enc=x)
        out_ones = model(x_enc=x, input_mask=ones_mask)
        out_partial = model(x_enc=x, input_mask=partial_mask)

    forecast = extract_forecast(out_ones)
    result = {
        "task": "forecasting",
        "input_shape": list(x.shape),
        "parameter_counts": parameter_counts(model),
        "outputs": {
            "no_mask_forecast": tensor_summary(
                "forecast_no_mask", extract_forecast(out_no_mask), [2, 1, horizon]
            ),
            "ones_mask_forecast": tensor_summary(
                "forecast", forecast, [2, 1, horizon]
            ),
            "partial_mask_forecast": tensor_summary(
                "forecast_partial_mask",
                extract_forecast(out_partial),
                [2, 1, horizon],
            ),
        },
        "prediction_extraction": (
            "outputs.forecast -> torch.Tensor [batch, channels, forecast_horizon]"
        ),
        "mask_behavior": (
            "input_mask is optional; explicit ones and partial zeros were accepted. "
            "Zero positions suppress patches through the T5 attention mask."
        ),
    }
    return result, model_config_dict(model)


def format_log(results: dict) -> str:
    lines = ["MOMENT-1-Small smoke test", "=========================", ""]
    lines.append(f"Checkpoint: {results['checkpoint']}")
    lines.append(f"Device:     {results['device']}")
    lines.append(f"All checks: {'PASS' if results['all_checks_passed'] else 'FAIL'}")
    for task in ("classification", "forecasting"):
        block = results[task]
        lines.extend(["", f"[{task}]", f"input: {block['input_shape']}"])
        lines.append(
            "parameters: "
            f"{block['parameter_counts']['trainable']:,} trainable / "
            f"{block['parameter_counts']['total']:,} total"
        )
        for name, summary in block["outputs"].items():
            lines.append(
                f"  {name:<22} shape={summary['shape']} finite={summary['all_finite']} "
                f"device={summary['device']}"
            )
        lines.append(f"  extraction: {block['prediction_extraction']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="AutonLab/MOMENT-1-small")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="outputs/metrics")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    classification, classification_config = run_classification(
        device, args.checkpoint, args.seq_len
    )
    forecasting, forecasting_config = run_forecasting(
        device, args.checkpoint, args.seq_len, args.horizon
    )

    all_summaries = [
        summary
        for block in (classification, forecasting)
        for summary in block["outputs"].values()
    ]
    results = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "classification": classification,
        "forecasting": forecasting,
        "all_checks_passed": all(
            summary["shape_ok"] and summary["all_finite"] for summary in all_summaries
        ),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "moment_smoke_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "moment_config_classification.json").write_text(
        json.dumps(classification_config, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "moment_config_forecasting.json").write_text(
        json.dumps(forecasting_config, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "moment_smoke_test.log").write_text(
        format_log(results), encoding="utf-8"
    )
    print(format_log(results))
    if not results["all_checks_passed"]:
        raise SystemExit("Smoke test failed: check outputs/metrics/moment_smoke_results.json")


if __name__ == "__main__":
    main()
