"""The building blocks every paradigm shares."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import pandas as pd
import torch

from cognarium.errors import MissingAbilityError
from cognarium.responses import Query
from cognarium.settings import Settings


@dataclass(frozen=True, eq=False)
class HypothesisSpace:
    """Everything a learner with explicit hypotheses needs to know about a task.

    Attributes:
        features: Each item's features, shape (n_items, n_features).
        hypotheses: Each hypothesis's option index per item, shape
            (n_hypotheses, n_items).
        options: The response options.
        groups: The group each hypothesis belongs to.
    """

    features: torch.Tensor
    hypotheses: torch.Tensor
    options: tuple[str, ...]
    groups: tuple[str, ...]


@dataclass(frozen=True, eq=False)
class Step:
    """One batch of queries, such as one generation of iterated learning.

    Attributes:
        index: The step's position in the experiment.
        queries: The queries to answer.
        columns: Paradigm-specific result columns, with one value per query.
    """

    index: int
    queries: tuple[Query, ...]
    columns: Mapping[str, Sequence[Any]]


class Paradigm(ABC):
    """An experiment, written as a function from past results to the next step.

    A paradigm keeps no running state: `step` depends only on its arguments. A
    stopped run can therefore continue from its last saved step.

    Attributes:
        name: Identifies the paradigm in results.
        requires_learning: Whether subjects must be able to learn.
        group_by: The columns metrics are reported by.
        noise: How likely a transmitted label is to flip.
        modalities: The stimulus fields this paradigm provides.
        settings: The experiment settings.
    """

    name: ClassVar[str]
    requires_learning: ClassVar[bool] = False
    group_by: ClassVar[tuple[str, ...]] = ()
    noise: float = 0.0
    modalities: frozenset[str]
    settings: Settings

    @abstractmethod
    def step(self, index: int, previous: pd.DataFrame | None) -> Step | None:
        """Builds the next batch of queries.

        Args:
            index: The step to build.
            previous: The results table of the step before it, if any.

        Returns:
            The step, or None when the experiment is over.
        """

    @abstractmethod
    def config(self) -> dict[str, Any]:
        """Describes the paradigm in JSON-safe form.

        Returns:
            Everything that changes the results, used to identify a run.
        """

    def annotate(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Adds columns that depend on a finished step's responses.

        Args:
            frame: The results table of one step.

        Returns:
            The table with any added columns.
        """
        return frame

    def texts(self) -> list[str]:
        """Lists the text a subject could see, for building a vocabulary.

        Returns:
            The text, as a list of strings.
        """
        return []

    def hypothesis_space(self) -> HypothesisSpace:
        """Gives the hypotheses a Bayesian learner considers on this paradigm.

        Returns:
            The hypothesis space.

        Raises:
            MissingAbilityError: If the paradigm doesn't define one.
        """
        raise MissingAbilityError(f"Paradigm '{self.name}' has no hypothesis space.")
