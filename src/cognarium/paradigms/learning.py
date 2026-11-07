"""Paradigms about learning and cultural transmission."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from cognarium.paradigms.base import HypothesisSpace, Paradigm, Step
from cognarium.responses import Query
from cognarium.settings import Settings
from cognarium.stimuli import Observation, Trial

if TYPE_CHECKING:
    from cognarium.tasks.shepard import ShepardCategories

# separate seed streams, so adding draws to one never shifts another
_TRANSMIT_STREAM = 0
_RESPONSE_STREAM = 1
_GROUP_STREAM = 2
_INIT_STREAM = 3


class IteratedLearning(Paradigm):
    """Each generation learns from the last generation's responses.

    Every chain starts from the task's seed structure. Each generation sees a
    random subset of the previous generation's labels, labels every item, and
    passes those labels on.

    Args:
        task: The task being transmitted.
        generations: How many learners follow each other in a chain.
        chains: How many independent chains to run.
        bottleneck: How many labelled items each generation sees. Showing every
            item lets a subject that reads its context simply copy it.
        noise: How likely each transmitted label is to flip to another option.
        instructions: Text placed before each prompt.
        examples: Demonstrations shown every generation, before the transmitted
            data. Observations or `(stimulus, label)` pairs. They can bias a
            chain, and results record that they were used.
        temperature: Divides log probabilities before a response is sampled.
        top_p: Keeps the smallest set of options covering this much probability.
        top_k: Keeps only this many of the most likely options.
        seed: Seeds everything random in the experiment.

    Raises:
        ValueError: If a count or probability is out of range, or an example's
            label isn't one of the task's options.
    """

    name = "iterated_learning"
    requires_learning = True
    group_by = ("generation",)

    def __init__(
        self,
        task: ShepardCategories,
        generations: int = 10,
        chains: int = 50,
        bottleneck: int = 4,
        noise: float = 0.0,
        *,
        instructions: str | None = None,
        examples: Sequence[Any] = (),
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int | None = None,
        seed: int = 0
    ) -> None:
        n_items = len(task.stimuli)
        if generations < 1 or chains < 1:
            raise ValueError("generations and chains must be at least 1.")
        if not 1 <= bottleneck <= n_items:
            raise ValueError(f"bottleneck must be between 1 and {n_items}, got {bottleneck}.")
        if not 0 <= noise < 1:
            raise ValueError(f"noise must be in [0, 1), got {noise}.")
        self.task = task
        self.generations = generations
        self.chains = chains
        self.bottleneck = bottleneck
        self.noise = noise
        self.modalities = task.modalities
        self.settings = Settings(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            instructions=instructions,
            examples=tuple(examples),
            seed=seed
        )
        for example in self.settings.examples:
            if example.label not in task.options:
                raise ValueError(f"Example label '{example.label}' is not one of {task.options}.")

    def step(self, index: int, previous: pd.DataFrame | None) -> Step | None:
        """Builds one generation across every chain.

        Args:
            index: The generation to build.
            previous: The results table of the generation before it.

        Returns:
            The generation's queries, or None after the last generation.

        Raises:
            ValueError: If a later generation has no previous results.
        """
        if index >= self.generations:
            return None
        parents = self._parent_labels(index, previous)
        seed = self.settings.seed
        queries = []
        columns = {
            "chain": [],
            "generation": [],
            "parent_label": [],
            "seen": [],
            "transmitted_label": []
        }
        for chain in range(self.chains):
            transmitted = self._transmit(index, chain, parents[chain])
            context = (
                *self.settings.examples,
                *(Observation(self.task.stimuli[item], label) for item, label in transmitted.items())
            )
            for item, stimulus in enumerate(self.task.stimuli):
                queries.append(
                    Query(
                        trial=Trial(
                            stimulus,
                            options=self.task.options,
                            answer=self.task.seed_labels[item],
                            item=item
                        ),
                        context=context,
                        instructions=self.settings.instructions,
                        template=self.task.template,
                        group=chain,
                        response_seed=(
                            seed,
                            _RESPONSE_STREAM,
                            index,
                            chain,
                            item
                        ),
                        group_seed=(
                            seed,
                            _GROUP_STREAM,
                            index,
                            chain
                        ),
                        init_seed=(seed, _INIT_STREAM, chain)
                    )
                )
                columns["chain"].append(chain)
                columns["generation"].append(index)
                columns["parent_label"].append(parents[chain][item])
                columns["seen"].append(item in transmitted)
                columns["transmitted_label"].append(transmitted.get(item))
        return Step(index=index, queries=tuple(queries), columns=columns)

    def config(self) -> dict[str, Any]:
        """Describes the experiment in JSON-safe form.

        Returns:
            Everything that changes the results, with settings at the top level.
        """
        return {
            "name": self.name,
            "task": self.task.config(),
            "generations": self.generations,
            "chains": self.chains,
            "bottleneck": self.bottleneck,
            "noise": self.noise,
            "groups": list(self.task.groups),
            **self.settings.config()
        }

    def annotate(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Adds the group each chain's labelling falls into.

        Args:
            frame: The results table of one generation.

        Returns:
            The table with a `group` column.
        """
        ordered = frame.sort_values(["chain", "item"])
        labels = ordered.groupby("chain")["response"].agg(list)
        groups = {chain: self.task.group_of(values) for chain, values in labels.items()}
        return frame.assign(group=frame["chain"].map(groups))

    def texts(self) -> list[str]:
        """Lists every line of text a subject could see.

        Returns:
            The instructions, each stimulus with each option, and the examples.
        """
        template = self.task.template
        lines = [] if self.settings.instructions is None else [self.settings.instructions]
        for stimulus in self.task.stimuli:
            for option in self.task.options:
                lines.append(template.render_observation(Observation(stimulus, option)))
        for example in self.settings.examples:
            if example.stimulus.text is not None:
                lines.append(template.render_observation(example))
        return lines

    def hypothesis_space(self) -> HypothesisSpace:
        """Gives the hypotheses a Bayesian learner considers on the task.

        Returns:
            The task's hypothesis space.
        """
        return self.task.hypothesis_space()

    def _parent_labels(self, index: int, previous: pd.DataFrame | None) -> np.ndarray:
        n_items = len(self.task.stimuli)
        if index == 0:
            return np.array([self.task.seed_labels] * self.chains, dtype=object)
        if previous is None:
            raise ValueError(f"Generation {index} needs the previous generation's results.")
        labels = np.empty((self.chains, n_items), dtype=object)
        labels[previous["chain"].to_numpy(), previous["item"].to_numpy()] = previous["response"].to_numpy()
        missing = np.argwhere(pd.isna(labels))
        if len(missing) > 0:
            chain, item = missing[0].tolist()
            raise ValueError(f"Generation {index} has no label from chain {chain}, item {item} of the previous one.")
        return labels

    def _transmit(
        self,
        index: int,
        chain: int,
        labels: np.ndarray
    ) -> dict[int, str]:
        rng = np.random.default_rng([
            self.settings.seed,
            _TRANSMIT_STREAM,
            index,
            chain
        ])
        options = self.task.options
        subset = rng.choice(len(labels), size=self.bottleneck, replace=False)
        flips = rng.random(self.bottleneck) < self.noise
        transmitted = {}
        for item, flip in zip(subset.tolist(), flips.tolist(), strict=True):
            label = labels[item]
            if flip:
                others = [option for option in options if option != label]
                label = others[rng.integers(len(others))]
            transmitted[item] = label
        return transmitted
