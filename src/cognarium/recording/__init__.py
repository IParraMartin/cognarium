"""Recording what a subject computes while it responds."""

from __future__ import annotations

from cognarium.recording.capture import StateCapture, hook_layers, pool_states
from cognarium.recording.layers import KNOWN_LAYER_MAPS, LayerMap, find_layer_map
from cognarium.recording.store import HiddenStates, StateWriter

__all__ = [
    "KNOWN_LAYER_MAPS",
    "HiddenStates",
    "LayerMap",
    "StateCapture",
    "StateWriter",
    "find_layer_map",
    "hook_layers",
    "pool_states"
]
