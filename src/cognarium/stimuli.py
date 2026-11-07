"""Stimuli, trials and the labelled observations learners see."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

_MODALITIES = ("text", "image", "audio", "features")


@dataclass(frozen=True, eq=False)
class Stimulus:
    """Something shown to a subject, in one or more modalities.

    Attributes:
        text: Text for subjects that read.
        image: An image for subjects that see.
        audio: A waveform for subjects that hear.
        features: A numeric vector, which lets non-neural learners read the
            same trial a language model reads as text.

    Raises:
        ValueError: If every field is empty.
    """

    text: str | None = None
    image: Any = None
    audio: Any = None
    features: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not self.modalities:
            raise ValueError("A stimulus needs at least one of text, image, audio or features.")

    @property
    def modalities(self) -> frozenset[str]:
        """The names of the fields this stimulus has set."""
        return frozenset(name for name in _MODALITIES if getattr(self, name) is not None)


@dataclass(frozen=True, eq=False)
class Trial:
    """One stimulus with the responses a subject can give to it.

    Attributes:
        stimulus: What the subject is shown.
        options: The possible responses. Any sequence is accepted and stored as
            a tuple.
        answer: The correct response, if the trial has one.
        condition: Labels for the experimental condition.
        item: An identifier for this item across chains or participants.

    Raises:
        ValueError: If there are fewer than two options, repeated options, or an
            answer that isn't an option.
    """

    stimulus: Stimulus
    options: tuple[str, ...]
    answer: str | None = None
    condition: Mapping[str, Any] = field(default_factory=dict)
    item: int | str | None = None

    def __post_init__(self) -> None:
        options = tuple(self.options)
        object.__setattr__(self, "options", options)
        if len(options) < 2:
            raise ValueError(f"A trial needs at least two options, got {options}.")
        if len(set(options)) != len(options):
            raise ValueError(f"Trial options must be different, got {options}.")
        if self.answer is not None and self.answer not in options:
            raise ValueError(f"Answer '{self.answer}' is not one of the options {options}.")


@dataclass(frozen=True, eq=False)
class Observation:
    """A labelled stimulus shown to a learner.

    Attributes:
        stimulus: The stimulus.
        label: The label given to it.
    """

    stimulus: Stimulus
    label: str


def as_observation(example: Observation | tuple[str | Stimulus, str]) -> Observation:
    """Normalises an example into an observation.

    Args:
        example: An observation, or a `(stimulus, label)` pair whose stimulus may
            be plain text.

    Returns:
        The example as an observation.
    """
    if isinstance(example, Observation):
        return example
    stimulus, label = example
    if isinstance(stimulus, str):
        stimulus = Stimulus(text=stimulus)
    return Observation(stimulus, label)
