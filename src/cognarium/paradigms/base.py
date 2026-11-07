"""The building blocks every paradigm shares."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd
import torch

from cognarium.context import ContextPolicy, isolated
from cognarium.errors import MissingAbilityError
from cognarium.responses import PromptTemplate, Query
from cognarium.settings import Settings
from cognarium.stimuli import Observation, Stimulus, Trial

if TYPE_CHECKING:
    from cognarium.results import Results

# the streams iterated learning uses are 0 to 3, and a new stream never shifts an old one
_RESPONSE_STREAM = 1
_GROUP_STREAM = 2
_INIT_STREAM = 3
_TRIAL_STREAM = 4
# trials that see no past can share a step, and this keeps each step's file small
_ISOLATED_STEP_SIZE = 256
_RESERVED_COLUMNS = frozenset({
    "step",
    "item",
    "options",
    "answer",
    "response",
    "response_index",
    "scores",
    "log_probs",
    "shaped_log_probs",
    "participant",
    "trial",
    "block",
    "history",
    "metric",
    "value",
    "reference",
    "state_shard",
    "state_offset",
    "logit_margin",
    "logit_difference",
    "target_rank",
    "vocab_entropy",
    "top_token_id",
    "option_first_tokens"
})


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


@dataclass(frozen=True)
class Effect:
    """One comparison a paradigm reports, declared as data.

    Attributes:
        name: Identifies the effect in the scores.
        measure: A measure with one value per row, such as `answer_surprisal`, or a
            numeric result column.
        by: The column naming the levels.
        levels: The levels to compare. `level_a` minus `level_b` is reported, so a
            human-like effect comes out positive.
        against: Compares with this value instead of a second level, such as 0.5 for chance.
        where: Keeps only the rows whose columns equal these values.
        options: Passed to the measure.
    """

    name: str
    measure: str
    by: str | None = None
    levels: tuple[Any, ...] | None = None
    against: float | None = None
    where: Mapping[str, Any] = field(default_factory=dict)
    options: Mapping[str, Any] = field(default_factory=dict)


class Paradigm:
    """An experiment, written as a list of trials or as a function of past results.

    Most paradigms only need `trials`. Each participant then gets those trials in
    their own order, and `context` says which past trials stay in view. A paradigm
    that needs more control overrides `step` and `config` instead.

    A paradigm keeps no running state: `step` depends only on its arguments. A
    stopped run can therefore continue from its last saved step.

    Args:
        n_participants: How many virtual participants take part. Each one draws its
            own trials and order.
        personas: Text placed before the instructions, with one per participant,
            repeating if there are fewer than participants. Only subjects that read
            text can take these.
        instructions: Text placed before each prompt.
        examples: Demonstrations shown before the trials. Observations or
            `(stimulus, label)` pairs.
        temperature: Divides log probabilities before a response is sampled.
        top_p: Keeps the smallest set of options covering this much probability.
        top_k: Keeps only this many of the most likely options.
        max_new_tokens: How many tokens a free response may use.
        seed: Seeds everything random in the experiment.
        **parameters: Anything else the paradigm uses, which becomes an attribute.

    Attributes:
        name: Identifies the paradigm in results. Defaults to the class name in snake
            case.
        requires_learning: Whether subjects must be able to learn.
        group_by: The columns metrics are reported by. Defaults to the trials' condition
            labels.
        noise: How likely a transmitted label is to flip.
        modalities: The stimulus fields this paradigm provides. Defaults to those every
            trial has.
        settings: The experiment settings.
        context: Which past trials stay in a subject's context.
        ordered: Whether participants see trials in the order `trials` gives. Otherwise
            each participant's trials are shuffled within their blocks.
        response_mode: `"choice"` picks an option, and `"generate"` answers freely.
        template: How stimuli and labels are written for subjects that read text.
        effects: What `score` reports. The first is the paradigm's primary effect.
        personas: The personas, if any.
        n_participants: How many virtual participants take part.
        max_new_tokens: How many tokens a free response may use.
        parameters: The paradigm's other parameters.

    Raises:
        ValueError: If a count is below 1, the list of personas is empty, or a parameter
            would replace one of the paradigm's methods.
    """

    name: ClassVar[str]
    requires_learning: ClassVar[bool] = False
    noise: float = 0.0
    settings: Settings
    context: ContextPolicy = isolated
    ordered: ClassVar[bool] = False
    response_mode: ClassVar[str] = "choice"
    template: PromptTemplate = PromptTemplate()
    effects: tuple[Effect, ...] = ()
    personas: tuple[str, ...] | None = None
    n_participants: int = 1
    max_new_tokens: int = 32

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "name" not in cls.__dict__:
            cls.name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", cls.__name__).lower()

    def __init__(
        self,
        *,
        n_participants: int = 1,
        personas: Sequence[str] | None = None,
        instructions: str | None = None,
        examples: Sequence[Any] = (),
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int | None = None,
        max_new_tokens: int = 32,
        seed: int = 0,
        **parameters: Any
    ) -> None:
        if n_participants < 1:
            raise ValueError(f"n_participants must be at least 1, got {n_participants}.")
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be at least 1, got {max_new_tokens}.")
        if personas is not None and len(personas) == 0:
            raise ValueError("personas is empty. Leave it out, or give at least one.")
        for key, value in parameters.items():
            if callable(getattr(type(self), key, None)):
                raise ValueError(f"A parameter can't be called '{key}', since that would replace a method.")
            setattr(self, key, value)
        self.parameters = parameters
        self.n_participants = n_participants
        self.personas = None if personas is None else tuple(personas)
        self.max_new_tokens = max_new_tokens
        self.settings = Settings(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            instructions=instructions,
            examples=tuple(examples),
            seed=seed
        )

    def trials(self, rng: np.random.Generator) -> Iterable[Trial]:
        """Gives one participant's trials.

        Args:
            rng: Seeded for this participant, for anything random in the trials.

        Returns:
            The trials, in any order unless the class sets `ordered`.

        Raises:
            NotImplementedError: If the paradigm defines neither `trials` nor `step`.
        """
        raise NotImplementedError(f"{type(self).__name__} needs a trials(self, rng) method, or its own step().")

    def sequences(self) -> list[list[Trial]]:
        """Gives each participant's trials in the order they see them.

        Returns:
            One list per participant, which depends only on the seed.

        Raises:
            ValueError: If a participant has no trials, a condition label would collide
                with a result column, or a label or item mixes types.
        """
        cached = self.__dict__.get("_sequences")
        if cached is None:
            cached = [self._participant_trials(participant) for participant in range(self.n_participants)]
            _check_columns(cached)
            self._check_response_mode(cached)
            self._sequences = cached
        return cached

    def step(self, index: int, previous: pd.DataFrame | None) -> Step | None:
        """Builds the next batch of queries.

        Args:
            index: The step to build.
            previous: The results table of the step before it, if any.

        Returns:
            The step, or None when the experiment is over.

        Raises:
            ValueError: If a later step has no previous results, or a policy with
                feedback meets a trial with no answer.
        """
        sequences = self.sequences()
        if self.context.keep == 0:
            places = [
                (participant, position, [])
                for participant, row in enumerate(sequences)
                for position in range(len(row))
            ]
            start = index * _ISOLATED_STEP_SIZE
            chosen = places[start : start + _ISOLATED_STEP_SIZE]
        else:
            if index > 0 and previous is None:
                raise ValueError(f"Step {index} needs the previous step's results.")
            chosen = [
                (participant, index, self._history(participant, index, previous))
                for participant, row in enumerate(sequences)
                if index < len(row)
            ]
        if not chosen:
            return None
        return self._build_step(index, chosen)

    def config(self) -> dict[str, Any]:
        """Describes the paradigm in JSON-safe form.

        Returns:
            Everything that changes the results, which together identify a run. The
            digest of the trials identifies them even when a parameter that made them
            can't be written as JSON.
        """
        return {
            "name": self.name,
            "parameters": {key: _json_safe(value) for key, value in self.parameters.items()},
            "trials": self._digest(),
            "n_participants": self.n_participants,
            "personas": None if self.personas is None else list(self.personas),
            "context": {
                "keep": self.context.keep,
                "feedback": self.context.feedback
            },
            "ordered": self.ordered,
            "response_mode": self.response_mode,
            "max_new_tokens": self.max_new_tokens,
            "participants_identical": self._participants_identical(),
            **self.settings.config()
        }

    def score(self, results: Results) -> pd.DataFrame:
        """Reports this paradigm's effects.

        Args:
            results: The results of a run of this paradigm.

        Returns:
            One row per effect, with the primary effect first. See `Results.contrast`
            for the columns.

        Raises:
            MissingAbilityError: If the paradigm declares no effects.
        """
        if not self.effects:
            raise MissingAbilityError(f"Paradigm '{self.name}' declares no effects. Set `effects`, or write score().")
        tables = [
            results.contrast(
                effect.measure,
                effect.by,
                levels=effect.levels,
                against=effect.against,
                where=effect.where or None,
                **effect.options
            ).assign(effect=effect.name)
            for effect in self.effects
        ]
        combined = pd.concat(tables, ignore_index=True)
        return combined[["effect", *(column for column in combined.columns if column != "effect")]]

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
            The personas, the instructions, each trial with each option, and the
            examples.
        """
        lines = [*(self.personas or ())]
        if self.settings.instructions is not None:
            lines.append(self.settings.instructions)
        for row in self.sequences():
            for trial in row:
                if trial.stimulus.text is None:
                    continue
                lines.append(self.template.render_query(trial.stimulus))
                lines.extend(
                    self.template.render_observation(Observation(trial.stimulus, option))
                    for option in trial.options
                )
        for example in self.settings.examples:
            if example.stimulus.text is not None:
                lines.append(self.template.render_observation(example))
        return lines

    def hypothesis_space(self) -> HypothesisSpace:
        """Gives the hypotheses a Bayesian learner considers on this paradigm.

        Returns:
            The hypothesis space.

        Raises:
            MissingAbilityError: If the paradigm doesn't define one.
        """
        raise MissingAbilityError(f"Paradigm '{self.name}' has no hypothesis space.")

    @property
    def group_by(self) -> tuple[str, ...]:
        """The trials' condition labels, which metrics are reported by."""
        if type(self).trials is Paradigm.trials:
            return ()
        return tuple(_condition_keys(self.sequences()))

    @property
    def modalities(self) -> frozenset[str]:
        """The stimulus fields every trial has."""
        explicit = self.__dict__.get("_modalities")
        if explicit is not None:
            return explicit
        trials = [trial for row in self.sequences() for trial in row]
        return frozenset.intersection(*(trial.stimulus.modalities for trial in trials))

    @modalities.setter
    def modalities(self, value: frozenset[str]) -> None:
        self._modalities = value

    def _participant_trials(self, participant: int) -> list[Trial]:
        rng = np.random.default_rng([self.settings.seed, _TRIAL_STREAM, participant])
        trials = list(self.trials(rng))
        if not trials:
            raise ValueError(f"Paradigm '{self.name}' made no trials.")
        trials = [
            trial if trial.item is not None else replace(trial, item=position)
            for position, trial in enumerate(trials)
        ]
        if self.ordered:
            return trials
        shuffled = [trials[position] for position in rng.permutation(len(trials)).tolist()]
        return sorted(shuffled, key=lambda trial: trial.block)

    def _check_response_mode(self, sequences: Sequence[Sequence[Trial]]) -> None:
        trials = [trial for row in sequences for trial in row]
        free = sum(trial.free for trial in trials)
        if self.response_mode == "generate" and free < len(trials):
            raise ValueError(f"Paradigm '{self.name}' has response_mode 'generate', so its trials can't have options.")
        if self.response_mode == "choice" and free > 0:
            raise ValueError(
                f"Paradigm '{self.name}' has response_mode 'choice', so its trials need options. "
                "Set response_mode = 'generate' for free responses."
            )

    def _history(
        self,
        participant: int,
        index: int,
        previous: pd.DataFrame | None
    ) -> list[str]:
        if index == 0:
            return []
        row = self.sequences()[participant]
        if row[index].block != row[index - 1].block:
            return []
        last = previous[previous["participant"] == participant].iloc[0]
        label = last["answer"] if self.context.feedback else last["response"]
        if pd.isna(label):
            raise ValueError(
                f"Trial {index - 1} of participant {participant} has no answer, which a context "
                "with feedback must show."
            )
        history = [*last["history"], label]
        return history if self.context.keep is None else history[-self.context.keep :]

    def _build_step(self, index: int, chosen: Sequence[tuple[int, int, list[str]]]) -> Step:
        sequences = self.sequences()
        conditions = _condition_keys(sequences)
        columns = {
            "participant": [],
            "trial": [],
            "block": [],
            "history": [],
            **{key: [] for key in conditions}
        }
        queries = []
        for participant, position, history in chosen:
            row = sequences[participant]
            trial = row[position]
            past = row[position - len(history) : position]
            queries.append(
                Query(
                    trial=trial,
                    context=(
                        *self.settings.examples,
                        *(Observation(seen.stimulus, label) for seen, label in zip(past, history, strict=True))
                    ),
                    instructions=self._instructions(participant),
                    template=self.template,
                    group=participant,
                    response_seed=(
                        self.settings.seed,
                        _RESPONSE_STREAM,
                        participant,
                        position
                    ),
                    group_seed=(self.settings.seed, _GROUP_STREAM, participant),
                    init_seed=(self.settings.seed, _INIT_STREAM, participant)
                )
            )
            columns["participant"].append(participant)
            columns["trial"].append(position)
            columns["block"].append(trial.block)
            columns["history"].append(history)
            for key in conditions:
                columns[key].append(trial.condition.get(key))
        return Step(index=index, queries=tuple(queries), columns=columns)

    def _instructions(self, participant: int) -> str | None:
        parts = []
        if self.personas is not None:
            parts.append(self.personas[participant % len(self.personas)])
        if self.settings.instructions is not None:
            parts.append(self.settings.instructions)
        return self.template.separator.join(parts) if parts else None

    def _digest(self) -> str:
        digest = hashlib.sha256()
        for row in self.sequences():
            digest.update(json.dumps([_describe_trial(trial) for trial in row], sort_keys=True).encode())
        return digest.hexdigest()

    def _participants_identical(self) -> bool:
        if self.n_participants < 2 or self.context.keep != 0 or self.personas is not None:
            return False
        multisets = [
            Counter(json.dumps(_describe_trial(trial), sort_keys=True) for trial in row)
            for row in self.sequences()
        ]
        return all(multiset == multisets[0] for multiset in multisets)


def _condition_keys(sequences: Sequence[Sequence[Trial]]) -> list[str]:
    # a dict keeps first-seen order, which is stable across runs
    keys = {}
    for row in sequences:
        for trial in row:
            keys.update(dict.fromkeys(trial.condition))
    return list(keys)


def _check_columns(sequences: Sequence[Sequence[Trial]]) -> None:
    keys = _condition_keys(sequences)
    colliding = sorted(_RESERVED_COLUMNS & set(keys))
    if colliding:
        raise ValueError(f"The condition labels {colliding} would replace result columns. Rename them.")
    trials = [trial for row in sequences for trial in row]
    for key in keys:
        kinds = {type(trial.condition[key]).__name__ for trial in trials if trial.condition.get(key) is not None}
        if len(kinds) > 1:
            raise ValueError(f"The condition label '{key}' mixes the types {sorted(kinds)}, which a table can't store.")
    kinds = {type(trial.item).__name__ for trial in trials}
    if len(kinds) > 1:
        raise ValueError(
            f"Trial items mix the types {sorted(kinds)}, which a table can't store. Set every item or none."
        )


def _describe_trial(trial: Trial) -> dict[str, Any]:
    return {
        **_describe_stimulus(trial.stimulus),
        "options": list(trial.options),
        "answer": trial.answer,
        "condition": {key: _json_safe(value) for key, value in trial.condition.items()},
        "item": trial.item,
        "block": trial.block
    }


def _describe_stimulus(stimulus: Stimulus) -> dict[str, Any]:
    described = {"text": stimulus.text}
    for field in ("image", "audio", "features"):
        value = getattr(stimulus, field)
        if value is not None:
            described[field] = hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
    return described


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return f"<{type(value).__name__}>"
    return value
