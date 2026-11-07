from __future__ import annotations

import torch

from cognarium.responses import Responses
from cognarium.subjects.base import Subject


class ScriptedSubject(Subject):
    """Gives fixed probabilities or choices, and counts how often it is called."""

    def __init__(
        self,
        probabilities=(0.5, 0.5),
        choices=None,
        modes=("in_context",),
        modalities=("text",),
        fail_after=None
    ):
        self.probabilities = probabilities
        self.choices = choices
        self.learning_modes = frozenset(modes)
        self.modalities = frozenset(modalities)
        self.default_learning = next(iter(self.learning_modes), None)
        self.fail_after = fail_after
        self.calls = 0

    def respond(self, queries, *, capture_logits=False):
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError("scripted failure")
        self.calls += 1
        row = torch.tensor(self.probabilities, dtype=torch.float64).log()
        scores = row.expand(len(queries), -1).clone()
        choices = None
        if self.choices is not None:
            choices = torch.full((len(queries),), self.choices, dtype=torch.long)
        return Responses(scores=scores, choices=choices)
