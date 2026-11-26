"""Builders and helpers for MOMENT classification experiments."""

from __future__ import annotations

from typing import Literal

import torch
from momentfm import MOMENTPipeline
from transformers import T5Config, T5EncoderModel


def build_moment_classifier(
    num_class: int,
    n_channels: int = 1,
    checkpoint: str = "AutonLab/MOMENT-1-small",
    seq_len: int = 512,
    patch_len: int = 8,
    reduction: Literal["concat", "mean"] = "concat",
    freeze_encoder: bool = True,
    freeze_embedder: bool = True,
    randomly_initialize_backbone: bool = False,
    device: str | torch.device = "auto",
) -> MOMENTPipeline:
    """Load MOMENT-1-Small with a newly initialized classification head.

    The classification head is not pretrained; it must be trained downstream.
    """
    model_kwargs = {
        "task_name": "classification",
        "n_channels": n_channels,
        "num_class": num_class,
        "seq_len": seq_len,
        "patch_len": patch_len,
        "reduction": reduction,
        "freeze_encoder": freeze_encoder,
        "freeze_embedder": freeze_embedder,
    }
    model = MOMENTPipeline.from_pretrained(checkpoint, model_kwargs=model_kwargs)
    model.init()
    if randomly_initialize_backbone:
        randomize_encoder_weights(model)
    model.to(resolve_device(device))
    return model


def randomize_encoder_weights(model) -> None:
    """Replace MOMENT's loaded T5 encoder with freshly initialized weights.

    ``from_pretrained`` restores checkpoint weights after MOMENT constructs the
    backbone, so ``randomly_initialize_backbone`` in the model config is not
    sufficient. This function performs the ablation reinitialization explicitly.
    """
    device = next(model.parameters()).device
    t5_config = T5Config.from_dict(model.config.t5_config)
    random_encoder = T5EncoderModel(t5_config).get_encoder()
    if getattr(model.encoder, "is_gradient_checkpointing", False):
        random_encoder.gradient_checkpointing_enable()
    model.encoder = random_encoder
    model.to(device)


def resolve_device(device: str | torch.device) -> torch.device:
    """Resolve an explicit device or choose CUDA when available."""
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def extract_logits(outputs) -> torch.Tensor:
    """Return classification logits from a TimeseriesOutputs object."""
    logits = outputs.logits
    if not torch.is_tensor(logits):
        logits = torch.as_tensor(logits)
    return logits
