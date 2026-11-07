"""What every subject provides."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from cognarium.errors import MissingAbilityError
from cognarium.responses import PromptTemplate, Query, Responses
from cognarium.stimuli import Observation

if TYPE_CHECKING:
    from cognarium.paradigms.base import Paradigm
    from cognarium.recording.capture import StateCapture


@dataclass(frozen=True)
class Readout:
    """How to turn a subject's recorded states into logits.

    Attributes:
        head: Maps recorded states to logits over the vocabulary.
        decode: Turns a token id into text.
        device: Where `head` must run.
        dtype: The precision `head` expects, which stored states may not have.
    """

    head: Callable[[torch.Tensor], torch.Tensor]
    decode: Callable[[int], str]
    device: torch.device
    dtype: torch.dtype = torch.float32


class Subject(ABC):
    """Anything that can give probabilities over responses.

    `respond` is the only requirement. Subjects that learn through a stored state
    also implement `fresh` and `learn`.

    Attributes:
        modalities: The stimulus fields this subject can read.
        learning_modes: The ways this subject can learn.
        default_learning: The mode `learning="auto"` picks.
        state_dtype: The precision states are stored in by default.
        can_generate: Whether this subject can write free responses.
    """

    modalities: frozenset[str] = frozenset()
    learning_modes: frozenset[str] = frozenset()
    default_learning: str | None = None
    state_dtype: ClassVar[str] = "float16"
    can_generate: ClassVar[bool] = False

    @abstractmethod
    def respond(
        self,
        queries: Sequence[Query],
        *,
        capture_logits: bool = False,
        capture_states: StateCapture | None = None
    ) -> Responses:
        """Scores each query's options.

        Args:
            queries: The queries to answer.
            capture_logits: Whether to return full-vocabulary logits, for
                subjects that have them.
            capture_states: Where to send internal states, for subjects that have
                them. Subjects that can't record states may leave this argument out,
                because it is only passed when recording.

        Returns:
            One row of scores per query.
        """

    def generate(
        self,
        queries: Sequence[Query],
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int | None = None
    ) -> list[str]:
        """Writes a free response to each query.

        Args:
            queries: The queries to answer.
            max_new_tokens: How many tokens each response may use.
            temperature: Divides log probabilities before a token is sampled.
            top_p: Keeps the smallest set of tokens covering this much probability.
            top_k: Keeps only this many of the most likely tokens.

        Returns:
            One text per query.

        Raises:
            MissingAbilityError: If this subject can't write free responses.
        """
        raise MissingAbilityError(f"{type(self).__name__} can only choose between options, not write free responses.")

    def state_layers(self) -> tuple[str, ...]:
        """Names the internal states this subject can record.

        Returns:
            The layer names, which are empty for a subject that can't record states.
        """
        return ()

    def readout(self) -> Readout | None:
        """Gives what the logit lens needs to read this subject's states.

        Returns:
            The readout, or None if states can't be turned into logits.
        """
        return None

    def first_tokens(self, options: Sequence[str]) -> list[int] | None:
        """Gives the token each option starts with.

        Args:
            options: The response options.

        Returns:
            One token id per option, or None for a subject without tokens.
        """
        return None

    def bind(self, paradigm: Paradigm, learning: str | None) -> Subject:
        """Prepares this subject for a run, before anything is called.

        Args:
            paradigm: The paradigm about to run.
            learning: The learning mode the run will use.

        Returns:
            The subject to run with, which may be a new object.
        """
        return self

    def config(self) -> dict[str, Any]:
        """Describes what makes this subject's results what they are.

        Returns:
            A JSON-safe description, which identifies a run.
        """
        return {"class": type(self).__name__}

    def runtime(self) -> dict[str, Any]:
        """Describes how this subject is running, which doesn't identify a run.

        Returns:
            A JSON-safe description, such as the device and batch size.
        """
        return {}

    def group_prior(self) -> dict[str, float] | None:
        """Gives the prior over hypothesis groups, for subjects that have one.

        Returns:
            Probability per group name, or None.
        """
        return None

    def fresh(self, seed: tuple[int, ...]) -> Subject:
        """Starts a new learner from this subject's starting state.

        Args:
            seed: Seeds any random starting state.

        Returns:
            A new learner.

        Raises:
            MissingAbilityError: If this subject doesn't keep a learned state.
        """
        raise MissingAbilityError(f"{type(self).__name__} has no fresh(), so it can't train.")

    def learn(
        self,
        observations: Sequence[Observation],
        *,
        template: PromptTemplate,
        instructions: str | None = None
    ) -> None:
        """Updates this subject's stored state from labelled observations.

        Args:
            observations: What to learn from.
            template: How observations are written for subjects that read text.
            instructions: Text placed before each observation.

        Raises:
            MissingAbilityError: If this subject doesn't keep a learned state.
        """
        raise MissingAbilityError(f"{type(self).__name__} has no learn(), so it can't train.")


def weights_digest(module: torch.nn.Module) -> str:
    """Fingerprints a module's weights, so that changed weights change a run's identity.

    Args:
        module: The module to fingerprint.

    Returns:
        A sha256 hex digest of the state dict.
    """
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        digest.update(name.encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()
