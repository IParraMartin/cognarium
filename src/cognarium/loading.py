"""Turning models into subjects."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

from cognarium.device import resolve_device, resolve_dtype
from cognarium.recording.layers import LayerMap
from cognarium.subjects.base import Subject
from cognarium.subjects.custom import CustomSubject
from cognarium.subjects.hf import HFSubject
from cognarium.subjects.multimodal import HFMultimodalSubject

# the auto classes tried in turn for a model that reads media, before its own class
_MULTIMODAL_AUTO_CLASSES = ("AutoModelForImageTextToText", "AutoModelForMultimodalLM")


def load(
    model: str | torch.nn.Module | Subject,
    *,
    tokenizer: Any = None,
    device: str = "auto",
    dtype: str = "auto",
    batch_size: int = 32,
    revision: str | None = None,
    pretrained: bool = True,
    adapter: Callable[..., torch.Tensor] | None = None,
    layer_map: LayerMap | None = None,
    layers: Mapping[str, str] | None = None,
    readout: str | None = None,
    chat: bool = False,
    multimodal: bool = False,
    processor: Any = None,
    media_tokens: Mapping[str, str] | None = None,
    processor_kwargs: Mapping[str, Any] | None = None
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
        layer_map: Where a Hugging Face model keeps its blocks, final norm and head,
            for a model that can't be recognised or detected.
        layers: Which submodules of a custom module to record, as names mapped to
            dotted paths.
        readout: The dotted path of a custom module's output head, which the logit
            lens reads states through.
        chat: Whether a Hugging Face model gets each prompt as one user message
            through its chat template, as instruction-tuned models expect.
        multimodal: Whether to load a Hugging Face model by name with its processor,
            so it reads the images and video of a stimulus along with text.
            Without it, a vision-language checkpoint loads as a text model.
        processor: The processor of a Hugging Face model object that reads media. A
            model object comes with either this or `tokenizer`.
        media_tokens: The text that stands for each kind of medium in a prompt, for a
            processor whose tokens can't be detected.
        processor_kwargs: Passed to every processor call, such as
            `{"images_kwargs": {"size": {"shortest_edge": 1024, "longest_edge": 4096}}}`
            to keep images small.

    Returns:
        The subject.

    Raises:
        TypeError: If the model isn't a supported type, a model object comes without
            a tokenizer or processor, or a setting doesn't apply to this kind of model.
        ValueError: If `chat` is on and there is no chat template.
    """
    if isinstance(model, Subject):
        return model
    resolved_device = resolve_device(device)
    media = {"media_tokens": media_tokens, "processor_kwargs": processor_kwargs}
    if not multimodal and processor is None:
        _refuse_settings(media, "for models that read media. Pass multimodal=True or a processor")
    if isinstance(model, str) and multimodal:
        _refuse_custom_settings(layers, readout)
        return _load_multimodal_by_name(
            model,
            processor=processor,
            device=resolved_device,
            dtype=resolve_dtype(dtype, resolved_device),
            batch_size=batch_size,
            revision=revision,
            layer_map=layer_map,
            chat=chat,
            **media
        )
    if isinstance(model, str):
        _refuse_custom_settings(layers, readout)
        return _load_by_name(
            model,
            tokenizer=tokenizer,
            device=resolved_device,
            dtype=resolve_dtype(dtype, resolved_device),
            batch_size=batch_size,
            revision=revision,
            layer_map=layer_map,
            chat=chat
        )
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Can't load a model of type {type(model).__name__}.")
    if tokenizer is None and processor is None:
        raise TypeError("A model object needs a tokenizer, or a processor for a model that reads media.")
    if dtype != "auto":
        model = model.to(resolve_dtype(dtype, resolved_device))
    from transformers import PreTrainedModel

    if isinstance(model, PreTrainedModel) and processor is not None:
        _refuse_custom_settings(layers, readout)
        return HFMultimodalSubject(
            model,
            processor,
            device=resolved_device,
            batch_size=batch_size,
            layer_map=layer_map,
            chat=chat,
            **media
        )
    if processor is not None:
        raise TypeError("processor is for Hugging Face models that read media.")
    if isinstance(model, PreTrainedModel):
        _refuse_custom_settings(layers, readout)
        return HFSubject(
            model,
            tokenizer,
            device=resolved_device,
            batch_size=batch_size,
            layer_map=layer_map,
            chat=chat
        )
    if chat:
        raise TypeError("chat is for Hugging Face models.")
    if layer_map is not None:
        raise TypeError("layer_map is for Hugging Face models. Use layers and readout on a custom module.")
    return CustomSubject(
        model,
        tokenizer,
        adapter=adapter,
        pretrained=pretrained,
        device=resolved_device,
        batch_size=batch_size,
        layers=layers,
        readout=readout
    )


def _refuse_custom_settings(layers: Mapping[str, str] | None, readout: str | None) -> None:
    _refuse_settings(
        {"layers": layers, "readout": readout},
        "for custom modules. Use layer_map on a Hugging Face model"
    )


def _refuse_settings(settings: Mapping[str, Any], reason: str) -> None:
    for name, value in settings.items():
        if value is not None:
            raise TypeError(f"{name} is {reason}.")


def _load_multimodal_by_name(
    name: str,
    *,
    processor: Any,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    revision: str | None,
    layer_map: LayerMap | None,
    chat: bool,
    media_tokens: Mapping[str, str] | None,
    processor_kwargs: Mapping[str, Any] | None
) -> HFMultimodalSubject:
    import transformers

    if processor is None:
        processor = transformers.AutoProcessor.from_pretrained(name, revision=revision)
    config = transformers.AutoConfig.from_pretrained(name, revision=revision)
    candidates = [getattr(transformers, auto) for auto in _MULTIMODAL_AUTO_CLASSES if hasattr(transformers, auto)]
    candidates += [
        getattr(transformers, architecture)
        for architecture in config.architectures or ()
        if hasattr(transformers, architecture)
    ]
    model = None
    for candidate in candidates:
        try:
            model = candidate.from_pretrained(name, revision=revision, dtype=dtype)
            break
        except ValueError:
            continue
    # an omni model that also speaks keeps the part that reads its inputs as its thinker
    model = getattr(model, "thinker", model)
    if model is None:
        raise TypeError(
            f"No Hugging Face class loads {name} as a model that reads media. Load the model yourself and pass "
            "it to cg.load with its processor."
        )
    return HFMultimodalSubject(
        model,
        processor,
        device=device,
        batch_size=batch_size,
        revision=_resolve_revision(name, revision),
        layer_map=layer_map,
        chat=chat,
        media_tokens=media_tokens,
        processor_kwargs=processor_kwargs
    )


def _load_by_name(
    name: str,
    *,
    tokenizer: Any,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    revision: str | None,
    layer_map: LayerMap | None,
    chat: bool
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
        revision=_resolve_revision(name, revision),
        layer_map=layer_map,
        chat=chat
    )


def _resolve_revision(name: str, requested: str | None) -> str | None:
    if Path(name).exists():
        return None
    from huggingface_hub import try_to_load_from_cache

    # the cached file sits at .../snapshots/<commit>/config.json
    cached = try_to_load_from_cache(name, "config.json", revision=requested)
    return Path(cached).parent.name if isinstance(cached, str) else requested
