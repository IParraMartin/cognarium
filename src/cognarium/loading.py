"""Turning models into subjects."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from cognarium.device import resolve_device, resolve_dtype
from cognarium.subjects.base import Subject
from cognarium.subjects.custom import CustomSubject
from cognarium.subjects.hf import HFSubject


def load(
    model: str | torch.nn.Module | Subject,
    *,
    tokenizer: Any = None,
    device: str = "auto",
    dtype: str = "auto",
    batch_size: int = 32,
    revision: str | None = None,
    pretrained: bool = True,
    adapter: Callable[..., torch.Tensor] | None = None
) -> Subject:
    """Turns a model into a subject that can run experiments.

    Args:
        model: A Hugging Face model id or local path, a model object, or a subject,
            which is returned unchanged.
        tokenizer: The tokenizer. Required for model objects.
        device: Where the model runs.
        dtype: The model's precision.
        batch_size: How many sequences go through the model at once.
        revision: The Hugging Face revision to load.
        pretrained: Whether a custom module's weights are already trained.
        adapter: How to get logits out of a custom module.

    Returns:
        The subject.

    Raises:
        TypeError: If the model isn't a supported type, or a model object comes
            without a tokenizer.
    """
    if isinstance(model, Subject):
        return model
    resolved_device = resolve_device(device)
    if isinstance(model, str):
        return _load_by_name(
            model,
            tokenizer=tokenizer,
            device=resolved_device,
            dtype=resolve_dtype(dtype, resolved_device),
            batch_size=batch_size,
            revision=revision
        )
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Can't load a model of type {type(model).__name__}.")
    if tokenizer is None:
        raise TypeError("A model object needs a tokenizer.")
    if dtype != "auto":
        model = model.to(resolve_dtype(dtype, resolved_device))
    from transformers import PreTrainedModel

    if isinstance(model, PreTrainedModel):
        return HFSubject(
            model,
            tokenizer,
            device=resolved_device,
            batch_size=batch_size
        )
    return CustomSubject(
        model,
        tokenizer,
        adapter=adapter,
        pretrained=pretrained,
        device=resolved_device,
        batch_size=batch_size
    )


def _load_by_name(
    name: str,
    *,
    tokenizer: Any,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    revision: str | None
) -> HFSubject:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=dtype)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    return HFSubject(
        model,
        tokenizer,
        device=device,
        batch_size=batch_size,
        revision=_resolve_revision(name, revision)
    )


def _resolve_revision(name: str, requested: str | None) -> str | None:
    if Path(name).exists():
        return None
    from huggingface_hub import try_to_load_from_cache

    # the cached file sits at .../snapshots/<commit>/config.json
    cached = try_to_load_from_cache(name, "config.json", revision=requested)
    return Path(cached).parent.name if isinstance(cached, str) else requested
