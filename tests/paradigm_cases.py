"""Subjects and cases that prove what a paradigm's scoring can detect."""

from __future__ import annotations

import torch

from cognarium.paradigms.base import Effect
from cognarium.responses import Responses
from cognarium.stimuli import Trial
from cognarium.subjects.base import Subject


class EffectSubject(Subject):
    """Answers as if it showed exactly the effect it's given, in the human direction.

    Its scores come only from each trial's own labels, so a paradigm that labels a
    level wrongly makes the effect disappear. With no effect, every option is equally
    likely.

    Args:
        effect: The effect to show, or None for none.
        strength: How large a log score the effect moves.
        modes: The learning modes to claim.
    """

    def __init__(
        self,
        effect: Effect | None,
        strength: float = 2.0,
        modes: tuple[str, ...] = ("in_context",)
    ) -> None:
        self.effect = effect
        self.strength = strength
        self.modalities = frozenset({"text", "features"})
        self.learning_modes = frozenset(modes)
        self.default_learning = modes[0]

    def respond(self, queries, *, capture_logits=False):
        rows = [self._scores(query.trial) for query in queries]
        width = max(len(row) for row in rows)
        scores = torch.full((len(rows), width), -torch.inf, dtype=torch.float64)
        for position, row in enumerate(rows):
            scores[position, : len(row)] = row
        return Responses(scores=scores)

    def learn(
        self,
        observations,
        *,
        template,
        instructions=None
    ):
        pass

    def fresh(self, seed):
        return self

    def _scores(self, trial: Trial) -> torch.Tensor:
        scores = torch.zeros(len(trial.options), dtype=torch.float64)
        effect = self.effect
        if effect is None or not self._applies(trial):
            return scores
        direction = 1.0 if self._level(trial) == 0 else -1.0
        if effect.against is not None:
            scores[trial.options.index(trial.answer)] = self.strength
        elif effect.measure == "answer_surprisal":
            scores[trial.options.index(trial.answer)] = -direction * self.strength
        elif effect.measure in ("answer_probability", "answer_margin"):
            scores[trial.options.index(trial.answer)] = direction * self.strength
        elif effect.measure == "option_probability":
            option = effect.options["option"]
            position = option if isinstance(option, int) else trial.options.index(option)
            scores[position] = direction * self.strength
        elif effect.measure == "expected_option":
            scores = direction * self.strength * torch.arange(len(trial.options), dtype=torch.float64)
        else:
            raise NotImplementedError(f"No scripted effect for the measure '{effect.measure}'.")
        return scores

    def _applies(self, trial: Trial) -> bool:
        return all(_label(trial, key) == value for key, value in self.effect.where.items())

    def _level(self, trial: Trial) -> int | None:
        if self.effect.by is None:
            return 0
        label = _label(trial, self.effect.by)
        return self.effect.levels.index(label) if label in self.effect.levels else None


def _label(trial: Trial, key: str):
    return trial.block if key == "block" else trial.condition.get(key)
