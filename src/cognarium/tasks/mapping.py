"""Items mapped to labels from a fixed set, such as a category system or a function read on a scale."""

from __future__ import annotations

import math
import numbers
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
import torch

from cognarium.errors import MissingAbilityError
from cognarium.responses import PromptTemplate
from cognarium.stimuli import Stimulus

# iterated learning draws from streams 0 to 5, so a chain's random start uses its own
_START_STREAM = 6
_TEMPLATE = PromptTemplate(stimulus="Features: {text} -> Label:", response="")


class Mapping:
    """Items mapped to labels from a fixed set, which a paradigm transmits or teaches.

    A colour naming system is a mapping, with one feature vector per colour and
    made-up words as labels. So is a function read on a scale, with one x value per
    item and the points of the scale as labels. Items and their labels are written
    `Features: [0.2, 0.7, 0.1] -> Label: wug`. Iterated learning passes a mapping
    down chains of learners, and category learning teaches it from feedback.

    Args:
        features: One entry per item: a sequence of numbers or words, or a single
            value, which is written without brackets.
        labels: The fixed set of labels a subject chooses from.
        start: One label per item: the first generation's labels in iterated
            learning, and the labels taught in category learning. None draws a
            random labelling from the run's seed, one per chain in iterated
            learning and one for the whole run in category learning.
        template: How an item and its label are written.
        sigma: How far each item's meaning spreads over the others, for the naming
            measures. Defaults to the median distance from an item to its nearest
            neighbour. Only numeric features have one.
        prior: How often each item comes up, for the naming measures. Defaults to
            uniform.
        group_by: Turns one labelling, a label per item in item order, into a group
            name for `chain_entropy` and `kl`. By default a labelling's group is the
            number of labels it uses.
        name: Names the task in results.

    Raises:
        ValueError: If there are no items, items of numbers differ in length, there
            are fewer than two labels or a repeated one, `start` doesn't give one
            known label per item, `sigma` isn't positive, or `prior` doesn't give
            one weight per item with a positive total and none negative.
    """

    def __init__(
        self,
        features: Sequence[Any],
        labels: Sequence[str],
        *,
        start: Sequence[str] | None = None,
        template: PromptTemplate | None = None,
        sigma: float | None = None,
        prior: Sequence[float] | None = None,
        group_by: Callable[[tuple[str, ...]], str] | None = None,
        name: str = "mapping"
    ) -> None:
        items = [_plain(item) for item in features]
        if not items:
            raise ValueError("A mapping needs at least one item.")
        labels = tuple(str(label) for label in labels)
        if len(labels) < 2 or len(set(labels)) != len(labels):
            raise ValueError(f"Give at least two different labels, got {list(labels)}.")
        if start is not None:
            start = tuple(str(label) for label in start)
            unknown = sorted(set(start) - set(labels))
            if len(start) != len(items) or unknown:
                raise ValueError(
                    f"start needs one of the labels {list(labels)} for each of the {len(items)} items, "
                    f"got {len(start)} labels{f' including {unknown}' if unknown else ''}."
                )
        numeric = all(_numeric(value) for item in items for value in _values(item))
        widths = {len(_values(item)) for item in items}
        if numeric and len(widths) > 1:
            raise ValueError(f"Every item needs the same number of features, got lengths {sorted(widths)}.")
        if sigma is not None and not sigma > 0:
            raise ValueError(f"sigma must be positive, got {sigma}.")
        if prior is not None:
            prior = [float(weight) for weight in prior]
            if len(prior) != len(items) or min(prior) < 0 or sum(prior) <= 0:
                raise ValueError(
                    f"prior needs one weight per item ({len(items)}), none negative and a positive total."
                )
        self.name = name
        self.items = tuple(items)
        self.options = labels
        self.template = _TEMPLATE if template is None else template
        self.numeric = numeric
        self.modalities = frozenset({"text", "features"}) if numeric else frozenset({"text"})
        self.sigma = sigma
        self.prior = prior
        self.group_by = group_by
        self.groups = () if group_by is not None else tuple(_count_name(count) for count in range(1, len(labels) + 1))
        self._start = start
        self.stimuli = tuple(
            Stimulus(
                text=_written(item),
                features=torch.tensor(_values(item), dtype=torch.float32) if numeric else None
            )
            for item in items
        )

    @classmethod
    def from_function(
        cls,
        x: Iterable[float],
        function: Callable[[float], float],
        scale: Iterable[float],
        **settings: Any
    ) -> Mapping:
        """Builds a function-learning task, as in Kalish, Griffiths and Lewandowsky (2007).

        Args:
            x: The input values, one item each.
            function: Gives the first generation's answer for each input.
            scale: The answers a subject chooses from, in order.
            **settings: Passed to `Mapping`, except `start`, which this sets.

        Returns:
            A mapping from each x to a point of the scale, starting from the point
            nearest to `function(x)`.
        """
        inputs = [_plain(value) for value in x]
        points = [_plain(value) for value in scale]
        start = [
            points[int(np.argmin([abs(point - function(value)) for point in points]))]
            for value in inputs
        ]
        return cls(
            inputs,
            [str(point) for point in points],
            start=[str(point) for point in start],
            **settings
        )

    def start(self, chain: int, seed: int) -> tuple[str, ...]:
        """Gives a chain's first labelling.

        Args:
            chain: The chain.
            seed: The run's seed.

        Returns:
            The given start, or a random label for each item drawn for this chain.
        """
        if self._start is not None:
            return self._start
        rng = np.random.default_rng([seed, _START_STREAM, chain])
        return tuple(self.options[index] for index in rng.integers(len(self.options), size=len(self.items)))

    def group_of(self, labels: Sequence[str]) -> str:
        """Names the group a full labelling belongs to.

        Args:
            labels: One label per item, in item order.

        Returns:
            What `group_by` gives, or how many labels the labelling uses.
        """
        if self.group_by is not None:
            return str(self.group_by(tuple(labels)))
        return _count_name(len(set(labels)))

    def hypothesis_space(self) -> None:
        """Refuses, because a mapping doesn't list its possible labellings.

        Raises:
            MissingAbilityError: Always.
        """
        raise MissingAbilityError(
            "A Bayesian learner needs every possible labelling as a hypothesis, and a Mapping "
            f"with {len(self.items)} items and {len(self.options)} labels has "
            f"{len(self.options)}^{len(self.items)} of them. Use a neural or exemplar subject."
        )

    def meaning_space(self) -> tuple[torch.Tensor, float, torch.Tensor]:
        """Gives what the naming measures need about the items.

        Returns:
            The features, shape (n_items, n_features), in float64, the meaning width,
            and the prior over items.

        Raises:
            MissingAbilityError: If the features aren't numbers.
        """
        if not self.numeric:
            raise MissingAbilityError("The naming measures need items described by numbers.")
        points = torch.tensor([_values(item) for item in self.items], dtype=torch.float64)
        prior = torch.ones(len(points), dtype=torch.float64) if self.prior is None else torch.tensor(self.prior)
        return points, self.sigma or _nearest_spacing(points), (prior / prior.sum()).double()

    def config(self) -> dict[str, Any]:
        """Describes the task in JSON-safe form.

        Returns:
            The items, labels, start, wording and meaning settings. A `group_by` rule
            appears by its name, so renaming it is the way to mark a changed rule.
        """
        return {
            "name": self.name,
            "features": list(self.items),
            "labels": list(self.options),
            "start": None if self._start is None else list(self._start),
            "template": {
                "stimulus": self.template.stimulus,
                "response": self.template.response,
                "separator": self.template.separator
            },
            "sigma": self.sigma,
            "prior": self.prior,
            "group_by": None if self.group_by is None else getattr(self.group_by, "__qualname__", repr(self.group_by))
        }


def _plain(value: Any) -> Any:
    """Turns arrays, tensors and their elements into lists and Python scalars."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        return value
    return [_plain(element) for element in value]


def _values(item: Any) -> list[Any]:
    return item if isinstance(item, list) else [item]


def _numeric(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)


def _written(item: Any) -> str:
    if isinstance(item, list):
        return "[" + ", ".join(str(value) for value in item) + "]"
    return str(item)


def _count_name(count: int) -> str:
    return f"{count} label" if count == 1 else f"{count} labels"


def _nearest_spacing(points: torch.Tensor) -> float:
    if len(points) < 2:
        return 1.0
    distances = torch.cdist(points, points)
    distances.fill_diagonal_(torch.inf)
    spacing = distances.min(dim=1).values.median().item()
    return spacing if spacing > 0 else 1.0
