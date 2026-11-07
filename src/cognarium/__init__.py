"""Run cognitive science experiments on language models and other learners."""

from __future__ import annotations

from cognarium import (
    metrics,
    models,
    paradigms,
    subjects,
    tasks
)
from cognarium.errors import MissingAbilityError, RunConflictError
from cognarium.loading import load
from cognarium.paradigms.base import Paradigm, Step
from cognarium.responses import PromptTemplate, Query, Responses
from cognarium.results import Results
from cognarium.runner import run
from cognarium.stimuli import Observation, Stimulus, Trial

__version__ = "0.0.1"

__all__ = [
    "MissingAbilityError",
    "Observation",
    "Paradigm",
    "PromptTemplate",
    "Query",
    "Responses",
    "Results",
    "RunConflictError",
    "Step",
    "Stimulus",
    "Trial",
    "load",
    "metrics",
    "models",
    "paradigms",
    "run",
    "subjects",
    "tasks"
]
