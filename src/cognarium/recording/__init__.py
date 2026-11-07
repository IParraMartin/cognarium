"""Recording what a subject computes while it responds."""

from __future__ import annotations

from cognarium.recording.capture import StateCapture, hook_layers, pool_states
from cognarium.recording.layers import KNOWN_LAYER_MAPS, LayerMap, find_layer_map

__all__ = [
    "KNOWN_LAYER_MAPS",
    "LayerMap",
    "StateCapture",
    "find_layer_map",
    "hook_layers",
    "pool_states"
]
