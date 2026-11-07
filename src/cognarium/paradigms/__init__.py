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
from cognarium.paradigms.learning import IteratedLearning

PARADIGMS: dict[str, type[Paradigm]] = {
    paradigm.name: paradigm
    for paradigm in (
        IteratedLearning,
        Stroop,
        Flanker,
        Simon,
        GoNoGo,
        StopSignal,
        TaskSwitching
    )
}

__all__ = [
    "PARADIGMS",
    "Effect",
    "Flanker",
    "GoNoGo",
    "HypothesisSpace",
    "IteratedLearning",
    "Paradigm",
    "Simon",
    "Step",
    "StopSignal",
    "Stroop",
    "TaskSwitching"
]
