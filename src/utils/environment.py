"""Capture and record the runtime environment for reproducibility."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import platform
import sys
from pathlib import Path

PACKAGES = (
    "torch",
    "torchvision",
    "torchaudio",
    "momentfm",
    "transformers",
    "accelerate",
    "datasets",
    "huggingface_hub",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "statsmodels",
    "xgboost",
    "peft",
    "pyyaml",
    "pyarrow",
)


def collect_environment() -> dict[str, object]:
    """Collect Python, hardware, and package version information."""
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]

    info: dict[str, object] = {
        "python_version": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch_version": getattr(torch, "__version__", "not installed"),
        "torch_cuda_build": getattr(torch, "version", None).cuda if torch else "n/a",
        "cuda_available": torch.cuda.is_available() if torch else False,
    }

    devices = []
    if torch and torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": torch.cuda.get_device_capability(index),
                    "total_memory_gib": round(
                        torch.cuda.get_device_properties(index).total_memory / 1024**3, 2
                    ),
                }
            )
    info["gpus"] = devices

    versions: dict[str, str] = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    info["packages"] = versions
    return info


def format_environment(info: dict[str, object]) -> str:
    """Render environment information as readable text."""
    lines = [
        "Environment snapshot",
        "====================",
        f"Python:         {info['python_version']}",
        f"Executable:     {info['python_executable']}",
        f"Platform:       {info['platform']}",
        f"Processor:      {info['processor']}",
        f"PyTorch:        {info['torch_version']}",
        f"CUDA build:     {info['torch_cuda_build']}",
        f"CUDA available: {info['cuda_available']}",
    ]
    for gpu in info["gpus"]:  # type: ignore[union-attr]
        lines.append(
            f"GPU {gpu['index']}:      {gpu['name']} | capability {gpu['capability']} | "
            f"{gpu['total_memory_gib']} GiB"
        )
    lines.extend(["", "Package versions", "----------------"])
    for name, version in info["packages"].items():  # type: ignore[union-attr]
        lines.append(f"{name:<20} {version}")
    return "\n".join(lines) + "\n"


def write_environment(path: str | Path) -> dict[str, object]:
    """Write the environment snapshot to ``path`` and return the raw info."""
    info = collect_environment()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(format_environment(info), encoding="utf-8")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="outputs/environment.txt",
        help="Path for the environment snapshot.",
    )
    args = parser.parse_args()
    write_environment(args.output)
    print(f"Environment written to {args.output}")


if __name__ == "__main__":
    main()
