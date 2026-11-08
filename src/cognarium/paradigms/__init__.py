"""Experimental paradigms."""

from __future__ import annotations

from cognarium.paradigms.base import (
    Effect,
    HypothesisSpace,
    Paradigm,
    Step
)
from cognarium.paradigms.learning import ArtificialGrammar, CategoryLearning, IteratedLearning

PARADIGMS: dict[str, type[Paradigm]] = {
    paradigm.name: paradigm
    for paradigm in (
        IteratedLearning,
        CategoryLearning,
        ArtificialGrammar
    )
}

__all__ = [
    "PARADIGMS",
    "ArtificialGrammar",
    "CategoryLearning",
    "Effect",
    "HypothesisSpace",
    "IteratedLearning",
    "Paradigm",
    "Step"
]
