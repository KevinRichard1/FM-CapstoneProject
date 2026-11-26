"""Parameter-freezing and PEFT adapters for MOMENT experiments."""

from __future__ import annotations

from typing import Iterable

import torch
from peft import LoraConfig, PeftModel, get_peft_model


def parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total": total, "trainable": trainable}


def freeze_for_linear_probing(model) -> None:
    """Freeze embedder and encoder; keep only the downstream head trainable."""
    for parameter in model.patch_embedding.parameters():
        parameter.requires_grad_(False)
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in model.head.parameters():
        parameter.requires_grad_(True)


def attach_lora(
    model,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
    target_modules: Iterable[str] = ("q", "k", "v", "o"),
) -> tuple[object, dict[str, int]]:
    """Inject LoRA into MOMENT's T5 attention projections.

    The pretrained base weights, patch embedding, and normalizer remain frozen;
    LoRA parameters and the task head are trainable.
    """
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    wrapped_encoder = get_peft_model(model.encoder, lora_config)
    if not isinstance(wrapped_encoder, PeftModel):
        raise TypeError("PEFT did not return a PeftModel")
    # MOMENT enables reentrant T5 gradient checkpointing by default. With a
    # frozen embedder, checkpointed LoRA inputs do not require gradients, so
    # LoRA parameters silently receive no gradients. Disable checkpointing for
    # the adapter path rather than reporting a false LoRA run.
    wrapped_encoder.gradient_checkpointing_disable()
    model.encoder = wrapped_encoder

    for parameter in model.patch_embedding.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.encoder.named_parameters():
        parameter.requires_grad_("lora_" in name)
    for parameter in model.head.parameters():
        parameter.requires_grad_(True)

    return model, parameter_counts(model)
