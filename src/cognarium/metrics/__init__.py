"""Metrics, usable on plain tensors or through `Results.metrics`."""

from __future__ import annotations

from cognarium.metrics.information import (
    conditional_entropy,
    cross_entropy,
    entropy,
    information_bottleneck,
    joint_entropy,
    js_divergence,
    kl_divergence,
    ksg_mutual_information,
    mutual_information,
    perplexity,
    surprisal
)

__all__ = [
    "conditional_entropy",
    "cross_entropy",
    "entropy",
    "information_bottleneck",
    "joint_entropy",
    "js_divergence",
    "kl_divergence",
    "ksg_mutual_information",
    "mutual_information",
    "perplexity",
    "surprisal"
]
