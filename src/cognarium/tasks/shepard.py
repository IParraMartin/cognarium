"""The Shepard, Hovland and Jenkins (1961) category structures."""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from cognarium.paradigms.base import HypothesisSpace
from cognarium.responses import PromptTemplate
from cognarium.stimuli import Stimulus

SHEPARD_TYPES = ("I", "II", "III", "IV", "V", "VI")
GROUPS = (*SHEPARD_TYPES, "unbalanced")

_OPTIONS = ("A", "B")
_OBJECTS = tuple(itertools.product([0, 1], repeat=3))
_WORDS = (
    ("small", "large"),
    ("red", "blue"),
    ("circle", "square")
)
_CANONICAL = {
    "I": "AAAABBBB",
    "II": "AABBBBAA",
    "III": "BBBAABAA",
    "IV": "BBBABAAA",
    "V": "BBBAAAAB",
    "VI": "BAABABBA"
}


def _orbit(labels: tuple[int, ...]) -> set[tuple[int, ...]]:
    """Every labelling reachable by permuting or flipping features, or swapping labels."""
    found = set()
    for permutation in itertools.permutations(range(3)):
        for flip in itertools.product([0, 1], repeat=3):
            moved = [
                _OBJECTS.index(tuple(item[permutation[i]] ^ flip[i] for i in range(3)))
                for item in _OBJECTS
            ]
            mapped = tuple(labels[position] for position in moved)
            found.add(mapped)
            found.add(tuple(1 - label for label in mapped))
    return found


_GROUP_OF = {
    labelling: name
    for name, text in _CANONICAL.items()
    for labelling in _orbit(tuple(_OPTIONS.index(letter) for letter in text))
}
_HYPOTHESES = torch.tensor(
    [[(i >> j) & 1 for j in range(8)] for i in range(256)],
    dtype=torch.long
)
_HYPOTHESIS_GROUPS = tuple(
    _GROUP_OF.get(tuple(row), "unbalanced") for row in _HYPOTHESES.tolist()
)


class ShepardCategories:
    """Sort 8 objects with 3 binary features into two categories.

    Objects come in `itertools.product([0, 1], repeat=3)` order, over size, colour
    and shape, and read like "small red circle". Hypothesis `i` gives item `j` the
    label at bit `j` of `i`, with 0 meaning A and 1 meaning B.

    Args:
        type: The category structure that seeds a chain, from "I" to "VI".

    Raises:
        ValueError: If the type is unknown.
    """

    name = "shepard"
    options = _OPTIONS
    modalities = frozenset({"text", "features"})
    template = PromptTemplate(stimulus="Object: {text}", response="Category:")
    groups = GROUPS

    def __init__(self, type: str = "VI") -> None:
        if type not in _CANONICAL:
            raise ValueError(f"Unknown type '{type}'. Choose from {SHEPARD_TYPES}.")
        self.type = type
        self.seed_labels = tuple(_CANONICAL[type])
        self.stimuli = tuple(
            Stimulus(
                text=" ".join(words[value] for words, value in zip(_WORDS, item, strict=True)),
                features=torch.tensor(item, dtype=torch.float32)
            )
            for item in _OBJECTS
        )

    def start(self, chain: int, seed: int) -> tuple[str, ...]:
        """Gives a chain's first labelling.

        Args:
            chain: The chain.
            seed: The run's seed.

        Returns:
            The seed structure, which every chain starts from.
        """
        return self.seed_labels

    def group_of(self, labels: Sequence[str]) -> str:
        """Names the group a full labelling of the 8 objects belongs to.

        Args:
            labels: One label per object, in object order.

        Returns:
            The Shepard type, or "unbalanced" for unequal category sizes.
        """
        indices = tuple(self.options.index(label) for label in labels)
        return _GROUP_OF.get(indices, "unbalanced")

    def hypothesis_space(self) -> HypothesisSpace:
        """Gives all 256 labellings of the 8 objects as hypotheses.

        Returns:
            The hypothesis space, with the features of each object.
        """
        return HypothesisSpace(
            features=torch.tensor(_OBJECTS, dtype=torch.float32),
            hypotheses=_HYPOTHESES,
            options=self.options,
            groups=_HYPOTHESIS_GROUPS
        )

    def type_prior(self, weights: Mapping[str, float]) -> torch.Tensor:
        """Builds a prior over labellings from a weight per group.

        Args:
            weights: Weight per group name. Missing groups get zero, and weights
                are normalised to sum to 1.

        Returns:
            A float64 prior of length 256, split evenly within each group.

        Raises:
            ValueError: If a name is unknown or no weight is positive.
        """
        unknown = set(weights) - set(GROUPS)
        if unknown:
            raise ValueError(f"Unknown groups {sorted(unknown)}. Choose from {GROUPS}.")
        total = sum(weights.values())
        if total <= 0 or min(weights.values()) < 0:
            raise ValueError("Group weights must not be negative and must sum above 0.")
        counts = Counter(_HYPOTHESIS_GROUPS)
        return torch.tensor(
            [weights.get(group, 0.0) / counts[group] / total for group in _HYPOTHESIS_GROUPS],
            dtype=torch.float64
        )

    def config(self) -> dict[str, Any]:
        """Describes the task in JSON-safe form.

        Returns:
            The task's name and seed type.
        """
        return {"name": self.name, "type": self.type}
