"""Stimuli, trials and the labelled observations learners see."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from cognarium.media import MEDIA_KINDS, Audio, Image, Media, Video, as_media

_MODALITIES = ("text", "image", "video", "audio", "features")


@dataclass(frozen=True, eq=False)
class Stimulus:
    """Something shown to a subject, in one or more modalities.

    Any mix of fields can be set. A subject reads the fields it can, so one stimulus
    can carry a colour as an image for a vision-language model, as text for a
    language model and as features for an exemplar learner. The prompt template says
    where text and media go in a prompt.

    Attributes:
        text: Text for subjects that read.
        image: An image for subjects that see: a `cg.media.Image`, or anything it
            takes, such as a path, a PIL image or an array.
        video: A video: a `cg.media.Video`, a path, or frames of shape
            (frames, height, width, channels).
        audio: A sound: a `cg.media.Audio`, a path, or a `(waveform, sampling_rate)`
            pair.
        features: A numeric vector, which lets non-neural learners read the
            same trial a language model reads as text.

    Raises:
        ValueError: If every field is empty, or a medium can't be read as its kind.
        TypeError: If a field holds a medium of another kind.
    """

    text: str | None = None
    image: Image | None = None
    audio: Audio | None = None
    features: torch.Tensor | None = None
    video: Video | None = None

    def __post_init__(self) -> None:
        for kind in MEDIA_KINDS:
            value = getattr(self, kind)
            if value is not None:
                object.__setattr__(self, kind, as_media(kind, value))
        if not self.modalities:
            raise ValueError("A stimulus needs at least one of text, image, video, audio or features.")

    @property
    def modalities(self) -> frozenset[str]:
        """The names of the fields this stimulus has set."""
        return frozenset(name for name in _MODALITIES if getattr(self, name) is not None)

    @property
    def media(self) -> dict[str, Media]:
        """The images, videos and sounds this stimulus has, by kind."""
        return {kind: getattr(self, kind) for kind in MEDIA_KINDS if getattr(self, kind) is not None}


@dataclass(frozen=True, eq=False)
class Trial:
    """One stimulus with the responses a subject can give to it.

    Attributes:
        stimulus: What the subject is shown.
        options: The possible responses. Any sequence is accepted and stored as
            a tuple. No options means the subject answers freely.
        answer: The correct response, if the trial has one.
        condition: Labels for the experimental condition.
        item: An identifier for this item across chains or participants.
        block: Which block of trials this belongs to. A new block starts with a
            fresh context.

    Raises:
        ValueError: If there is only one option, repeated options, or an answer
            that isn't an option.
    """

    stimulus: Stimulus
    options: tuple[str, ...] = ()
    answer: str | None = None
    condition: Mapping[str, Any] = field(default_factory=dict)
    item: int | str | None = None
    block: int = 0

    def __post_init__(self) -> None:
        options = tuple(self.options)
        object.__setattr__(self, "options", options)
        if len(options) == 1:
            raise ValueError(f"A trial needs at least two options, or none for a free response, got {options}.")
        if len(set(options)) != len(options):
            raise ValueError(f"Trial options must be different, got {options}.")
        if options and self.answer is not None and self.answer not in options:
            raise ValueError(f"Answer '{self.answer}' is not one of the options {options}.")

    @property
    def free(self) -> bool:
        """Whether the subject answers freely rather than choosing an option."""
        return not self.options


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
