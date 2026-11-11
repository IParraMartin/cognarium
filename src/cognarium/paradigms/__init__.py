"""Experimental paradigms."""

from __future__ import annotations

from cognarium.paradigms.base import (
    Effect,
    HypothesisSpace,
    Paradigm,
    Step
)
from cognarium.paradigms.learning import ArtificialGrammar, IteratedLearning

PARADIGMS: dict[str, type[Paradigm]] = {
    paradigm.name: paradigm
    for paradigm in (
        IteratedLearning,
        ArtificialGrammar
    )
}

__all__ = [
    "PARADIGMS",
    "ArtificialGrammar",
    "Effect",
    "HypothesisSpace",
    "IteratedLearning",
    "Paradigm",
    "Step"
]
