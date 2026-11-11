"""Run cognitive science experiments on language models and other learners."""

from __future__ import annotations

from cognarium import (
    context,
    data,
    media,
    metrics,
    models,
    paradigms,
    subjects,
    tasks
)
from cognarium.errors import (
    IdenticalParticipantsWarning,
    MissingAbilityError,
    RunConflictError
)
from cognarium.loading import load
from cognarium.paradigms.base import Effect, Paradigm, Step
from cognarium.recording import HiddenStates, LayerMap
from cognarium.responses import PromptTemplate, Query, Responses
from cognarium.results import Results
from cognarium.runner import run
from cognarium.stimuli import Observation, Stimulus, Trial
from cognarium.subjects.base import Readout

__version__ = "0.1.0"

__all__ = [
    "Effect",
    "HiddenStates",
    "IdenticalParticipantsWarning",
    "LayerMap",
    "MissingAbilityError",
    "Observation",
    "Paradigm",
    "PromptTemplate",
    "Query",
    "Readout",
    "Responses",
    "Results",
    "RunConflictError",
    "Step",
    "Stimulus",
    "Trial",
    "context",
    "data",
    "load",
    "media",
    "metrics",
    "models",
    "paradigms",
    "run",
    "subjects",
    "tasks"
]
