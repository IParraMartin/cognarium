"""Settings shared by experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cognarium.stimuli import Observation, as_observation


@dataclass(frozen=True, eq=False)
class Settings:
    """Experiment settings that apply to every subject.

    Attributes:
        temperature: Divides log probabilities before a response is sampled.
        top_p: Keeps the smallest set of options covering this much probability.
        top_k: Keeps only this many of the most likely options.
        instructions: Text placed before each prompt.
        examples: Demonstrations shown before the data. Observations or
            `(stimulus, label)` pairs, stored as observations.
        seed: Seeds everything random in the experiment.

    Raises:
        ValueError: If a shaping value is out of range.
    """

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    instructions: str | None = None
    examples: tuple[Observation, ...] = ()
    seed: int = 0

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError(
                f"temperature must be above 0, got {self.temperature}. "
                "Use top_k=1 for greedy responses."
            )
        if not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}.")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {self.top_k}.")
        object.__setattr__(self, "examples", tuple(as_observation(item) for item in self.examples))

    @property
    def shaping(self) -> dict[str, Any]:
        """The keyword arguments for `shape_log_probs`."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k
        }

    def config(self) -> dict[str, Any]:
        """Describes the settings in JSON-safe form.

        Returns:
            Every setting, with feature tensors written as lists.
        """
        return {
            **self.shaping,
            "instructions": self.instructions,
            "examples": [_describe(item) for item in self.examples],
            "seed": self.seed
        }


def _describe(observation: Observation) -> dict[str, Any]:
    features = observation.stimulus.features
    return {
        "text": observation.stimulus.text,
        "features": None if features is None else features.tolist(),
        "label": observation.label
    }
