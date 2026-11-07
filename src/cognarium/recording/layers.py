"""Where a language model keeps its blocks, final norm and output head."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cognarium.errors import MissingAbilityError


@dataclass(frozen=True)
class LayerMap:
    """Dotted paths to the parts of a model that recording needs.

    Attributes:
        layers: The list of blocks, such as `"transformer.h"`.
        norm: The norm applied after the last block, or None if there is none.
        head: The module that maps states to logits.
    """

    layers: str
    norm: str | None
    head: str


_TRANSFORMER = LayerMap("transformer.h", "transformer.ln_f", "lm_head")
_DECODER = LayerMap("model.layers", "model.norm", "lm_head")

KNOWN_LAYER_MAPS: dict[str, LayerMap] = {
    "gpt2": _TRANSFORMER,
    "gptj": _TRANSFORMER,
    "falcon": _TRANSFORMER,
    "bloom": _TRANSFORMER,
    "llama": _DECODER,
    "mistral": _DECODER,
    "gemma": _DECODER,
    "gemma2": _DECODER,
    "qwen2": _DECODER,
    "phi3": _DECODER,
    "gpt_neox": LayerMap("gpt_neox.layers", "gpt_neox.final_layer_norm", "lm_head"),
    "opt": LayerMap("model.decoder.layers", "model.decoder.final_layer_norm", "lm_head")
}


def find_layer_map(model: torch.nn.Module) -> LayerMap:
    """Finds where a model keeps its blocks, final norm and output head.

    A model's type is looked up first. Libraries sometimes rename these modules, so a
    known map whose paths no longer exist is ignored in favour of detection.

    Args:
        model: The model.

    Returns:
        The model's layer map.

    Raises:
        MissingAbilityError: If the model has no known type and detection fails.
    """
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    known = KNOWN_LAYER_MAPS.get(model_type)
    if known is not None and _paths_exist(model, known):
        return known
    return _detect(model)


def _paths_exist(model: torch.nn.Module, layer_map: LayerMap) -> bool:
    paths = [layer_map.layers, layer_map.head] + ([] if layer_map.norm is None else [layer_map.norm])
    try:
        for path in paths:
            model.get_submodule(path)
    except AttributeError:
        return False
    return True


def _detect(model: torch.nn.Module) -> LayerMap:
    find_head = getattr(model, "get_output_embeddings", None)
    head = find_head() if callable(find_head) else None
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.ModuleList) and len(module) > 0 and len({type(block) for block in module}) == 1
    ]
    if head is None or not candidates:
        raise MissingAbilityError(
            f"Can't find the blocks and output head of {type(model).__name__}. "
            "Pass a LayerMap to say where they are."
        )
    layers, _ = max(candidates, key=lambda candidate: len(candidate[1]))
    parent = layers.rpartition(".")[0]
    norms = [
        name
        for name, _ in model.get_submodule(parent).named_children()
        if "norm" in name.lower() or "ln" in name.lower()
    ]
    norm = None if not norms else f"{parent}.{norms[-1]}" if parent else norms[-1]
    head_path = next(name for name, module in model.named_modules() if module is head)
    return LayerMap(layers, norm, head_path)
