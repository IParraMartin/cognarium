"""Experimental paradigms."""

from __future__ import annotations

from cognarium.paradigms.base import (
    Effect,
    HypothesisSpace,
    Paradigm,
    Step
)
from cognarium.paradigms.control import (
    Flanker,
    GoNoGo,
    Simon,
    StopSignal,
    Stroop,
    TaskSwitching
)
from cognarium.paradigms.learning import (
    ArtificialGrammar,
    CategoryLearning,
    IteratedLearning,
    StatisticalLearning
)
from cognarium.paradigms.memory import (
    DRM,
    FreeRecall,
    NBack,
    Span,
    Sternberg
)

PARADIGMS: dict[str, type[Paradigm]] = {
    paradigm.name: paradigm
    for paradigm in (
        IteratedLearning,
        Stroop,
        Flanker,
        Simon,
        GoNoGo,
        StopSignal,
        TaskSwitching,
        NBack,
        Span,
        Sternberg,
        FreeRecall,
        DRM,
        CategoryLearning,
        ArtificialGrammar,
        StatisticalLearning
    )
}

__all__ = [
    "DRM",
    "PARADIGMS",
    "ArtificialGrammar",
    "CategoryLearning",
    "Effect",
    "Flanker",
    "FreeRecall",
    "GoNoGo",
    "HypothesisSpace",
    "IteratedLearning",
    "NBack",
    "Paradigm",
    "Simon",
    "Span",
    "StatisticalLearning",
    "Step",
    "Sternberg",
    "StopSignal",
    "Stroop",
    "TaskSwitching"
]
