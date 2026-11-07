"""Metrics, usable on plain tensors or through `Results.metrics`."""

from __future__ import annotations

from cognarium.metrics.information import entropy, kl_divergence, surprisal

__all__ = ["entropy", "kl_divergence", "surprisal"]
