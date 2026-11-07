"""The building blocks every paradigm shares."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, eq=False)
class HypothesisSpace:
    """Everything a learner with explicit hypotheses needs to know about a task.

    Attributes:
        features: Each item's features, shape (n_items, n_features).
        hypotheses: Each hypothesis's option index per item, shape
            (n_hypotheses, n_items).
        options: The response options.
        groups: The group each hypothesis belongs to.
    """

    features: torch.Tensor
    hypotheses: torch.Tensor
    options: tuple[str, ...]
    groups: tuple[str, ...]
