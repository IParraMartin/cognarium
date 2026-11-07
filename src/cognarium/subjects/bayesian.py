"""A Bayesian learner with an explicit space of hypotheses."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from cognarium.device import resolve_device, resolve_dtype
from cognarium.paradigms.base import HypothesisSpace
from cognarium.responses import Query, Responses
from cognarium.settings import Settings
from cognarium.stimuli import Observation, Stimulus
from cognarium.subjects.base import Subject

if TYPE_CHECKING:
    from cognarium.paradigms.base import Paradigm

_DECISIONS = ("sample", "map", "average")
# posterior values this close count as tied, so rounding can't favour one hypothesis
_TIE_TOLERANCE = 1e-9


class BayesianLearner(Subject):
    """Updates a prior over hypotheses from labelled observations.

    The decision rule matters under iterated learning. Learners who `sample` a
    hypothesis from the posterior make the prior the chain's stationary
    distribution (Griffiths and Kalish, 2007), while `map` learners exaggerate it.
    `sample` and `map` commit to one hypothesis per chain, whereas `average`
    responds to each item from the posterior predictive. Scores are always the
    posterior predictive.

    Args:
        hypotheses: The hypothesis space. By default it comes from the paradigm.
        prior: Prior probability per hypothesis. Uniform by default.
        likelihood: Maps `(hypothesis_labels, label_index, noise)` to a log
            likelihood per hypothesis. By default a noisy channel, where a label
            flips with the paradigm's `noise`. The prior is only stationary when
            this matches how the paradigm transmits labels.
        decision: How a response is chosen: "sample", "map" or "average".
        device: Where to compute.

    Raises:
        ValueError: If the decision rule is unknown.
    """

    modalities = frozenset({"features"})
    learning_modes = frozenset({"posterior"})
    default_learning = "posterior"

    def __init__(
        self,
        *,
        hypotheses: HypothesisSpace | None = None,
        prior: torch.Tensor | None = None,
        likelihood: Callable[[torch.Tensor, int, float], torch.Tensor] | None = None,
        decision: str = "sample",
        device: str = "cpu"
    ) -> None:
        if decision not in _DECISIONS:
            raise ValueError(f"Unknown decision '{decision}'. Choose from {_DECISIONS}.")
        self.hypotheses = hypotheses
        self.prior = prior
        self.likelihood = likelihood
        self.decision = decision
        self.device = resolve_device(device)
        self.dtype = resolve_dtype("auto", self.device, prefer=torch.float64)
        self.noise = 0.0
        self._space = None

    def bind(self, paradigm: Paradigm, learning: str | None) -> BayesianLearner:
        """Takes the hypotheses and noise from the paradigm.

        Args:
            paradigm: The paradigm about to run.
            learning: The learning mode the run will use.

        Returns:
            A new learner, ready to respond.

        Raises:
            ValueError: If shaping is combined with a joint decision rule, the
                prior doesn't fit the hypotheses, or an example isn't one of the
                task's items.
        """
        if self.decision != "average" and paradigm.settings.shaping != Settings().shaping:
            raise ValueError(
                f"decision='{self.decision}' chooses one hypothesis per chain, so temperature, "
                "top_p and top_k can't apply. Use decision='average' to shape responses."
            )
        bound = copy.copy(self)
        bound.noise = paradigm.noise
        bound._prepare(self.hypotheses or paradigm.hypothesis_space())
        for example in paradigm.settings.examples:
            bound._item_of(example.stimulus)
        return bound

    def log_posterior(self, context: Sequence[Observation]) -> torch.Tensor:
        """Computes the posterior over hypotheses given some observations.

        Args:
            context: The observations.

        Returns:
            The log posterior, of length n_hypotheses.

        Raises:
            RuntimeError: If this learner hasn't been bound to a paradigm.
            ValueError: If the observations rule out every hypothesis.
        """
        self._require_bound()
        log_posterior = self._log_prior
        for observation in context:
            item = self._item_of(observation.stimulus)
            label = self._space.options.index(observation.label)
            log_posterior = log_posterior + self._log_likelihoods[item, label]
        total = log_posterior.logsumexp(dim=0)
        if torch.isneginf(total):
            raise ValueError(
                "These observations rule out every hypothesis. Set noise above 0 or "
                "give the conflicting hypotheses some prior probability."
            )
        return log_posterior - total

    def respond(self, queries: Sequence[Query], *, capture_logits: bool = False) -> Responses:
        """Scores each query with the posterior predictive.

        Args:
            queries: The queries to answer. Their context holds the observations.
            capture_logits: Ignored, since scores already are the raw values.

        Returns:
            One row of scores per query, with `choices` set unless the decision
            rule is "average".

        Raises:
            RuntimeError: If this learner hasn't been bound to a paradigm.
        """
        self._require_bound()
        scores = torch.empty(len(queries), len(self._space.options), dtype=torch.float64)
        choices = torch.empty(len(queries), dtype=torch.long)
        chains = {}
        for position, query in enumerate(queries):
            key = (query.group, id(query.context))
            if key not in chains:
                chains[key] = self._solve(query)
            predictive, chosen = chains[key]
            item = self._item_of(query.trial.stimulus)
            scores[position] = predictive[item]
            choices[position] = self._labels[chosen, item]
        return Responses(scores=scores, choices=None if self.decision == "average" else choices)

    def group_prior(self) -> dict[str, float]:
        """Gives the prior probability of each hypothesis group.

        Returns:
            Probability per group name.
        """
        self._require_bound()
        totals = {}
        for group, probability in zip(self._space.groups, self._prior_values.tolist(), strict=True):
            totals[group] = totals.get(group, 0.0) + probability
        return totals

    def config(self) -> dict[str, Any]:
        """Describes what determines this learner's behaviour.

        Returns:
            The decision rule, noise, likelihood name, and digests of the prior
            and hypotheses.
        """
        self._require_bound()
        likelihood = "noisy_channel" if self.likelihood is None else self.likelihood.__name__
        return {
            "class": type(self).__name__,
            "decision": self.decision,
            "noise": self.noise,
            "likelihood": likelihood,
            "prior": hashlib.sha256(self._prior_values.numpy().tobytes()).hexdigest(),
            "hypotheses": hashlib.sha256(self._labels.numpy().tobytes()).hexdigest()
        }

    def runtime(self) -> dict[str, Any]:
        """Describes where the learner runs.

        Returns:
            The device.
        """
        return {"device": str(self.device)}

    def _prepare(self, space: HypothesisSpace) -> None:
        self._space = space
        self._labels = space.hypotheses.cpu()
        self._prior_values = self._normalised_prior(len(space.hypotheses))
        self._log_prior = self._prior_values.to(self.device, self.dtype).log()
        self._items = {
            tuple(features): index
            for index, features in enumerate(space.features.tolist())
        }
        hypotheses = space.hypotheses.to(self.device)
        n_items, n_options = hypotheses.shape[1], len(space.options)
        self._log_likelihoods = torch.stack([
            torch.stack([
                self._log_likelihood(hypotheses[:, item], label).to(self.device, self.dtype)
                for label in range(n_options)
            ])
            for item in range(n_items)
        ])
        gives_option = torch.stack([hypotheses.T == label for label in range(n_options)], dim=1)
        self._option_masks = torch.where(gives_option, 0.0, -torch.inf).to(self.device, self.dtype)

    def _normalised_prior(self, n_hypotheses: int) -> torch.Tensor:
        if self.prior is None:
            return torch.full((n_hypotheses,), 1 / n_hypotheses, dtype=torch.float64)
        prior = torch.as_tensor(self.prior, dtype=torch.float64).cpu()
        if prior.shape != (n_hypotheses,) or prior.min() < 0 or prior.sum() <= 0:
            raise ValueError(
                f"The prior needs {n_hypotheses} non-negative values with a positive sum, "
                f"got shape {tuple(prior.shape)}."
            )
        return prior / prior.sum()

    def _log_likelihood(self, labels: torch.Tensor, label: int) -> torch.Tensor:
        if self.likelihood is not None:
            return self.likelihood(labels, label, self.noise)
        others = len(self._space.options) - 1
        match = torch.tensor(1 - self.noise, dtype=self.dtype, device=labels.device).log()
        miss = torch.tensor(self.noise / others, dtype=self.dtype, device=labels.device).log()
        return torch.where(labels == label, match, miss)

    def _solve(self, query: Query) -> tuple[torch.Tensor, int]:
        log_posterior = self.log_posterior(query.context)
        predictive = (log_posterior + self._option_masks).logsumexp(dim=-1).cpu().double()
        return predictive, self._choose(log_posterior, query.group_seed)

    def _choose(self, log_posterior: torch.Tensor, seed: tuple[int, ...]) -> int:
        if self.decision == "average":
            return 0
        values = log_posterior.cpu().double().numpy()
        rng = np.random.default_rng(seed)
        if self.decision == "map":
            return int(rng.choice(np.flatnonzero(values >= values.max() - _TIE_TOLERANCE)))
        probabilities = np.exp(values)
        return int(rng.choice(len(values), p=probabilities / probabilities.sum()))

    def _item_of(self, stimulus: Stimulus) -> int:
        if stimulus.features is None:
            raise ValueError("BayesianLearner reads features, and this stimulus has none.")
        item = self._items.get(tuple(stimulus.features.tolist()))
        if item is None:
            raise ValueError("This stimulus's features aren't among the task's items.")
        return item

    def _require_bound(self) -> None:
        if self._space is None:
            raise RuntimeError("Call bind(paradigm, learning) first, or let cg.run do it.")
