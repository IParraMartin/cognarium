"""The Generalized Context Model of categorization (Nosofsky, 1986)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

from cognarium.device import resolve_device, resolve_dtype
from cognarium.responses import Query, Responses
from cognarium.stimuli import Stimulus
from cognarium.subjects.base import Subject

if TYPE_CHECKING:
    from cognarium.paradigms.base import Paradigm


class ExemplarLearner(Subject):
    """Categorizes by similarity to stored labelled exemplars.

    Similarity is `exp(-c * sum_d(w_d * |x_d - e_d|))`. An option's score is
    `gamma * log(summed similarity to its exemplars)`, which is -inf when it has
    none, so a chain whose labels all agree stays that way unless `guess` is set.

    Args:
        c: How quickly similarity falls with distance.
        attention: Weight per feature, normalised to sum to 1. Uniform by default.
        gamma: Sharpens or flattens responses.
        guess: Probability of responding uniformly at random.
        device: Where to compute.

    Raises:
        ValueError: If a parameter is out of range.
    """

    modalities = frozenset({"features"})
    learning_modes = frozenset({"exemplar"})
    default_learning = "exemplar"

    def __init__(
        self,
        *,
        c: float = 2.0,
        attention: Sequence[float] | None = None,
        gamma: float = 1.0,
        guess: float = 0.0,
        device: str = "cpu"
    ) -> None:
        if c < 0 or gamma <= 0 or not 0 <= guess <= 1:
            raise ValueError("c must not be negative, gamma must be above 0, and guess must be in [0, 1].")
        if attention is not None and (min(attention) < 0 or sum(attention) <= 0):
            raise ValueError("attention must not be negative and must sum above 0.")
        self.c = c
        self.attention = None if attention is None else list(attention)
        self.gamma = gamma
        self.guess = guess
        self.device = resolve_device(device)
        self.dtype = resolve_dtype("auto", self.device, prefer=torch.float64)

    def respond(self, queries: Sequence[Query], *, capture_logits: bool = False) -> Responses:
        """Scores each query's options from the exemplars in its context.

        Args:
            queries: The queries to answer. Their context holds the exemplars.
            capture_logits: Ignored, since scores already are the raw values.

        Returns:
            One row of scores per query.

        Raises:
            ValueError: If a stimulus has no features, or an exemplar's label isn't
                one of the options.
        """
        batches = defaultdict(list)
        for position, query in enumerate(queries):
            batches[(id(query.context), query.trial.options)].append(position)
        width = max(len(query.trial.options) for query in queries)
        scores = torch.full((len(queries), width), -torch.inf, dtype=torch.float64)
        for positions in batches.values():
            batch_scores = self._score_batch([queries[position] for position in positions])
            scores[positions, : batch_scores.shape[1]] = batch_scores.cpu().double()
        return Responses(scores=scores)

    def bind(self, paradigm: Paradigm, learning: str | None) -> ExemplarLearner:
        """Checks that the paradigm's examples can serve as exemplars.

        Args:
            paradigm: The paradigm about to run.
            learning: The learning mode the run will use.

        Returns:
            This learner.

        Raises:
            ValueError: If an example has no features.
        """
        for example in paradigm.settings.examples:
            _features(example.stimulus)
        return self

    def config(self) -> dict[str, Any]:
        """Describes the model's parameters.

        Returns:
            The class name and every parameter.
        """
        return {
            "class": type(self).__name__,
            "c": self.c,
            "attention": self.attention,
            "gamma": self.gamma,
            "guess": self.guess
        }

    def runtime(self) -> dict[str, Any]:
        """Describes where the model runs.

        Returns:
            The device.
        """
        return {"device": str(self.device)}

    def _score_batch(self, queries: list[Query]) -> torch.Tensor:
        options = queries[0].trial.options
        context = queries[0].context
        if not context:
            return torch.zeros(
                len(queries),
                len(options),
                dtype=self.dtype,
                device=self.device
            )
        for observation in context:
            if observation.label not in options:
                raise ValueError(f"Exemplar label '{observation.label}' is not one of {options}.")
        targets = self._stack([query.trial.stimulus for query in queries])
        exemplars = self._stack([observation.stimulus for observation in context])
        labels = [options.index(observation.label) for observation in context]
        labels = torch.tensor(labels, device=self.device)
        distances = (targets[:, None, :] - exemplars[None, :, :]).abs()
        # log space keeps a very large c from underflowing to "every option impossible"
        log_similarity = -self.c * (distances * self._weights(targets.shape[1])).sum(dim=-1)
        scores = torch.stack(
            [
                self.gamma * log_similarity.masked_fill(labels != index, -torch.inf).logsumexp(dim=1)
                for index in range(len(options))
            ],
            dim=1
        )
        if self.guess > 0:
            probabilities = (scores - scores.logsumexp(dim=1, keepdim=True)).exp()
            scores = ((1 - self.guess) * probabilities + self.guess / len(options)).log()
        return scores

    def _stack(self, stimuli: list[Stimulus]) -> torch.Tensor:
        return torch.stack([_features(stimulus) for stimulus in stimuli]).to(self.device, self.dtype)

    def _weights(self, n_features: int) -> torch.Tensor:
        values = [1.0] * n_features if self.attention is None else self.attention
        # a single weight would broadcast across every feature without complaint
        if len(values) != n_features:
            raise ValueError(f"attention needs one weight per feature: {n_features}, got {len(values)}.")
        weights = torch.tensor(values, dtype=self.dtype, device=self.device)
        return weights / weights.sum()


def _features(stimulus: Stimulus) -> torch.Tensor:
    if stimulus.features is None:
        raise ValueError("ExemplarLearner reads features, and this stimulus has none.")
    return stimulus.features
