"""Paradigms about learning and cultural transmission."""

from __future__ import annotations

import functools
import itertools
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch

from cognarium.context import ContextPolicy
from cognarium.paradigms.base import (
    BUILT_IN_STIMULI,
    Effect,
    HypothesisSpace,
    Paradigm,
    Step
)
from cognarium.responses import PromptTemplate, Query, text_only
from cognarium.settings import Settings
from cognarium.stimuli import Observation, Stimulus, Trial

if TYPE_CHECKING:
    from cognarium.tasks.mapping import Mapping
    from cognarium.tasks.shepard import ShepardCategories

# separate seed streams, so adding draws to one never shifts another
_TRANSMIT_STREAM = 0
_RESPONSE_STREAM = 1
_GROUP_STREAM = 2
_INIT_STREAM = 3
# the trial engine draws each participant's trials from stream 4
# cg.tasks.Mapping draws each chain's random start from stream 6
_ORDER_STREAM = 7

# Reber (1967): each state maps a letter to the next state, and None ends the string
_REBER = {
    0: {
        "T": 1,
        "P": 2
    },
    1: {
        "S": 1,
        "X": 3
    },
    2: {
        "T": 2,
        "V": 4
    },
    3: {
        "X": 2,
        "S": None
    },
    4: {
        "P": 3,
        "V": None
    }
}
_LETTERS = ("T", "P", "S", "X", "V")
_LONGEST = 8
_GRAMMAR_TESTS = ("judgment", "probability")
_GRAMMATICAL = "grammatical"
_UNGRAMMATICAL = "ungrammatical"


class IteratedLearning(Paradigm):
    """Each generation learns from the last generation's responses.

    Each chain starts from the task's first labelling for it. Each generation sees
    a random subset of the previous generation's labels, labels every item, and
    passes those labels on. A trial's answer is its chain's first label for the
    item, so `accuracy` measures agreement with where the chain started.

    Maps onto iterated learning experiments, where chains of people drift toward
    what they find easiest to learn, such as structured languages (Kirby, Cornish
    and Smith, 2008) and positive linear functions (Kalish, Griffiths and
    Lewandowsky, 2007). For a Bayesian learner that samples from its posterior, the
    chains converge to its prior (Griffiths and Kalish, 2007). What it can't
    capture: every generation is the same model with a fresh context, so the chains
    show one model's in-context biases rather than a population of different people.

    Args:
        task: The task being transmitted, such as `cg.tasks.ShepardCategories` or
            `cg.tasks.Mapping`.
        generations: How many learners follow each other in a chain.
        chains: How many independent chains to run.
        n_prev_examples: How many items each generation sees labelled by the
            previous one, drawn without repeats. Showing every item lets a subject
            that reads its context simply copy it.
        noise: How likely each transmitted label is to flip to another option.
        memory: How many of its own latest answers in the same generation a chain
            still sees, after the transmitted examples. With 0, every item is named
            independently and a generation is one step. Otherwise each chain names
            the items one at a time, in a new random order every generation, so a
            generation takes one step per item. Imel and Zaslavsky (2026) used 10,
            which kept chains from collapsing onto a single label.
        instructions: Text placed before each prompt.
        examples: Demonstrations shown every generation, before the transmitted
            data. Observations or `(stimulus, label)` pairs. They can bias a
            chain, and results record that they were used.
        temperature: Divides log probabilities before a response is sampled.
        top_p: Keeps the smallest set of options covering this much probability.
        top_k: Keeps only this many of the most likely options.
        seed: Seeds everything random in the experiment.

    Raises:
        ValueError: If a count or probability is out of range, `memory` is
            negative, or an example's label isn't one of the task's options.
    """

    name = "iterated_learning"
    requires_learning = True
    group_by = ("generation",)

    def __init__(
        self,
        task: ShepardCategories | Mapping,
        generations: int = 10,
        chains: int = 50,
        n_prev_examples: int = 4,
        noise: float = 0.0,
        *,
        memory: int = 0,
        instructions: str | None = None,
        examples: Sequence[Any] = (),
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int | None = None,
        seed: int = 0
    ) -> None:
        n_items = len(task.stimuli)
        if memory < 0:
            raise ValueError(f"memory must not be negative, got {memory}.")
        if generations < 1 or chains < 1:
            raise ValueError("generations and chains must be at least 1.")
        if not 1 <= n_prev_examples <= n_items:
            raise ValueError(f"n_prev_examples must be between 1 and {n_items}, got {n_prev_examples}.")
        if not 0 <= noise < 1:
            raise ValueError(f"noise must be in [0, 1), got {noise}.")
        self.task = task
        self.generations = generations
        self.chains = chains
        self.n_prev_examples = n_prev_examples
        self.noise = noise
        self.memory = memory
        self.modalities = task.modalities
        self.template = task.template
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
        """Builds the next step: one generation, or with memory one item per chain.

        Args:
            index: The step to build.
            previous: The results table of the step before it.

        Returns:
            The step's queries, or None after the last generation.

        Raises:
            ValueError: If a later step has no previous results, or they miss a chain.
        """
        if self.memory > 0:
            return self._item_step(index, previous)
        if index >= self.generations:
            return None
        parents = self._parent_labels(index, previous)
        starts = self._starts()
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
            context = (*self.settings.examples, *self._shown(transmitted))
            for item in range(len(self.task.stimuli)):
                queries.append(self._query(
                    index,
                    chain,
                    item,
                    context,
                    starts
                ))
                columns["chain"].append(chain)
                columns["generation"].append(index)
                columns["parent_label"].append(parents[chain][item])
                columns["seen"].append(item in transmitted)
                columns["transmitted_label"].append(transmitted.get(item))
        return Step(index=index, queries=tuple(queries), columns=columns)

    def _item_step(self, index: int, previous: pd.DataFrame | None) -> Step | None:
        n_items = len(self.task.stimuli)
        generation, position = divmod(index, n_items)
        if generation >= self.generations:
            return None
        parents, named = self._carried(generation, position, previous)
        starts = self._starts()
        queries = []
        columns = {
            "chain": [],
            "generation": [],
            "position": [],
            "parent_label": [],
            "seen": [],
            "transmitted_label": [],
            "parents": [],
            "named": []
        }
        for chain in range(self.chains):
            order = self._order(generation, chain)
            item = order[position]
            transmitted = self._transmit(generation, chain, parents[chain])
            recent = order[:position][-self.memory:]
            context = (
                *self.settings.examples,
                *self._shown(transmitted),
                *(Observation(self.task.stimuli[earlier], named[chain][earlier]) for earlier in recent)
            )
            queries.append(self._query(
                generation,
                chain,
                item,
                context,
                starts
            ))
            columns["chain"].append(chain)
            columns["generation"].append(generation)
            columns["position"].append(position)
            columns["parent_label"].append(parents[chain][item])
            columns["seen"].append(item in transmitted)
            columns["transmitted_label"].append(transmitted.get(item))
            columns["parents"].append(list(parents[chain]))
            columns["named"].append(list(named[chain]))
        return Step(index=index, queries=tuple(queries), columns=columns)

    def _carried(
        self,
        generation: int,
        position: int,
        previous: pd.DataFrame | None
    ) -> tuple[list[list[str]], list[list[str | None]]]:
        """Reads the previous generation's labelling and this generation's answers so far."""
        n_items = len(self.task.stimuli)
        empty = [[None] * n_items for _ in range(self.chains)]
        if generation == 0 and position == 0:
            return [list(start) for start in self._starts()], empty
        if previous is None:
            raise ValueError(f"Step {generation * n_items + position} needs the previous step's results.")
        rows = previous.set_index("chain")
        parents, named = [], []
        for chain in range(self.chains):
            if chain not in rows.index:
                raise ValueError(
                    f"The step before generation {generation}, item {position} has no row for chain {chain}."
                )
            row = rows.loc[chain]
            labels = list(row["named"])
            labels[int(row["item"])] = row["response"]
            parents.append(list(row["parents"]))
            named.append(labels)
        if position == 0:
            missing = next(
                ((chain, labels.index(None)) for chain, labels in enumerate(named) if None in labels),
                None
            )
            if missing is not None:
                chain, item = missing
                raise ValueError(f"Generation {generation - 1} has no label from chain {chain}, item {item}.")
            return named, empty
        return parents, named

    def _order(self, generation: int, chain: int) -> list[int]:
        rng = np.random.default_rng([
            self.settings.seed,
            _ORDER_STREAM,
            generation,
            chain
        ])
        return rng.permutation(len(self.task.stimuli)).tolist()

    def _shown(self, transmitted: dict[int, str]) -> tuple[Observation, ...]:
        return tuple(Observation(self.task.stimuli[item], label) for item, label in transmitted.items())

    def _query(
        self,
        generation: int,
        chain: int,
        item: int,
        context: tuple[Observation, ...],
        starts: list[tuple[str, ...]]
    ) -> Query:
        seed = self.settings.seed
        return Query(
            trial=Trial(
                self.task.stimuli[item],
                options=self.task.options,
                answer=starts[chain][item],
                item=item
            ),
            context=context,
            instructions=self.settings.instructions,
            template=self.task.template,
            group=chain,
            response_seed=(
                seed,
                _RESPONSE_STREAM,
                generation,
                chain,
                item
            ),
            group_seed=(
                seed,
                _GROUP_STREAM,
                generation,
                chain
            ),
            init_seed=(seed, _INIT_STREAM, chain)
        )

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
            "n_prev_examples": self.n_prev_examples,
            "noise": self.noise,
            "memory": self.memory,
            "groups": list(self.task.groups),
            **self.settings.config()
        }

    def annotate(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Adds the group each chain's labelling falls into.

        Args:
            frame: The results table of one step.

        Returns:
            The table with a `group` column. With memory, a chain's labelling is
            complete only on its last item of a generation, so only that row gets a
            group and the others get None.
        """
        if self.memory > 0:
            last = len(self.task.stimuli) - 1
            groups = []
            for _, row in frame.iterrows():
                labels = list(row["named"])
                labels[int(row["item"])] = row["response"]
                groups.append(self.task.group_of(labels) if row["position"] == last else None)
            return frame.assign(group=groups)
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
                lines.append(text_only(template.observation_parts(Observation(stimulus, option))))
        for example in self.settings.examples:
            if example.stimulus.text is not None:
                lines.append(text_only(template.observation_parts(example)))
        return lines

    def hypothesis_space(self) -> HypothesisSpace:
        """Gives the hypotheses a Bayesian learner considers on the task.

        Returns:
            The task's hypothesis space.
        """
        return self.task.hypothesis_space()

    def _starts(self) -> list[tuple[str, ...]]:
        return [self.task.start(chain, self.settings.seed) for chain in range(self.chains)]

    def _parent_labels(self, index: int, previous: pd.DataFrame | None) -> np.ndarray:
        n_items = len(self.task.stimuli)
        if index == 0:
            return np.array(self._starts(), dtype=object)
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
        subset = rng.choice(len(labels), size=self.n_prev_examples, replace=False)
        flips = rng.random(self.n_prev_examples) < self.noise
        transmitted = {}
        for item, flip in zip(subset.tolist(), flips.tolist(), strict=True):
            label = labels[item]
            if flip:
                others = [option for option in options if option != label]
                label = others[rng.integers(len(others))]
            transmitted[item] = label
        return transmitted


class _StudyThenTest(Paradigm):
    """A learning paradigm in which each block is studied before it is tested.

    Every query in a block shares one context, which holds the examples and then
    whatever `study` gives for that block, with its correct labels. A subject that
    learns by weights is therefore trained once per block and participant.
    """

    requires_learning = True
    stimuli_label = BUILT_IN_STIMULI
    context = ContextPolicy(keep=None, feedback=True)

    def study(self, participant: int, block: int) -> tuple[Observation, ...]:
        """Gives what a participant has studied before a block is tested.

        Args:
            participant: The participant.
            block: The block about to be tested.

        Returns:
            The labelled observations, in the order they were shown.

        Raises:
            NotImplementedError: If the paradigm doesn't define it.
        """
        raise NotImplementedError(f"{type(self).__name__} needs a study(participant, block) method.")

    def step(self, index: int, previous: pd.DataFrame | None) -> Step | None:
        """Builds one block's tests for every participant.

        Args:
            index: The position of the block to build, in block order.
            previous: Ignored, since what is studied never depends on responses.

        Returns:
            The block's queries, or None after the last block.
        """
        sequences = self.sequences()
        blocks = sorted({trial.block for row in sequences for trial in row})
        if index >= len(blocks):
            return None
        block = blocks[index]
        contexts = {
            participant: (*self.settings.examples, *self.study(participant, block))
            for participant in range(len(sequences))
        }
        chosen = [
            (participant, position, [])
            for participant, row in enumerate(sequences)
            for position, trial in enumerate(row)
            if trial.block == block
        ]
        return self._build_step(index, chosen, contexts)

    def texts(self) -> list[str]:
        """Lists the text a subject could see, for building a vocabulary.

        Returns:
            What every paradigm lists, followed by everything any participant studies.
        """
        lines = super().texts()
        blocks = sorted({trial.block for row in self.sequences() for trial in row})
        for participant in range(self.n_participants):
            for block in blocks:
                lines.extend(
                    text_only(self.template.observation_parts(observation))
                    for observation in self.study(participant, block)
                    if observation.stimulus.text is not None
                )
        return lines


class ArtificialGrammar(_StudyThenTest):
    """Study strings from a finite-state grammar, then judge new ones.

    The grammar is Reber's (1967), over the letters T, P, S, X and V, with strings of
    3 to 8 letters. Each participant studies their own strings, then is tested on
    new grammatical strings and as many ungrammatical ones. Each ungrammatical
    string changes one letter of a tested grammatical string so that it leaves the
    grammar, and of the changes possible, the one whose chunk strength is closest to
    the original's is used, so that grammaticality and familiarity come apart as in
    Knowlton and Squire (1996). A string's chunk strength is how often, on average,
    its letter pairs and triples occurred in the participant's study strings.
    Letters are written apart, as in `Letters: T X S`, so a subject that reads words
    sees letters. A string's features count its letter pairs, including its first
    and last letters, and strings that would share features are never used.

    There are two tests. With `test="judgment"`, the subject studies each string
    labelled grammatical, then judges each test string grammatical or ungrammatical,
    and each test string is marked `high` or `low` in chunk strength, splitting its
    status in half. With `test="probability"`, the subject studies the bare strings,
    then each trial compares a grammatical string with its ungrammatical change by
    the probability of each as a whole string, with no judgment asked. Only a
    language model gives these probabilities, so this test needs a subject that
    reads text. Results gain a `chunk_strength` column, and under the probability
    test a `chunk_difference` column, the grammatical string's chunk strength minus
    its change's.

    Maps onto artificial grammar learning, where people trained on grammatical strings
    later judge new ones above chance without being able to say the rules, and
    endorse familiar chunks whatever their grammaticality (Knowlton and Squire, 1996).
    The probability test is the model's familiarity with each string, which a
    judgment can misreport (Hu and Levy, 2023). What it can't capture: people aren't
    told the strings follow rules, while under the judgment test every study string
    carries the same label, and the measures are probabilities rather than
    endorsement rates. The strings come from the grammar, not human-normed.

    Args:
        n_train: How many strings each participant studies.
        n_test: How many grammatical strings are tested, and as many ungrammatical ones.
        noise: How often a Bayesian learner assumes a label it sees is wrong. The
            labels themselves never change.
        test: `"judgment"` or `"probability"`.
        **settings: What every paradigm takes, such as `n_participants`.

    Raises:
        ValueError: If a count is below 1, there aren't enough grammatical strings for
            both counts, `noise` is outside [0, 1), or `test` is unknown.
    """

    single_label_training = True
    template = PromptTemplate(stimulus="Letters: {text}", response="Judgment:")

    def __init__(
        self,
        n_train: int = 16,
        n_test: int = 8,
        noise: float = 0.2,
        test: str = "judgment",
        **settings: Any
    ) -> None:
        available = len(_grammar_pool()[0])
        if n_train < 1 or n_test < 1:
            raise ValueError(f"n_train and n_test must be at least 1, got {n_train} and {n_test}.")
        if n_train + n_test > available:
            raise ValueError(
                f"The grammar gives {available} usable strings, fewer than n_train + n_test = {n_train + n_test}."
            )
        if not 0 <= noise < 1:
            raise ValueError(f"noise must be in [0, 1), got {noise}.")
        if test not in _GRAMMAR_TESTS:
            raise ValueError(f"Unknown test '{test}'. Choose from {list(_GRAMMAR_TESTS)}.")
        super().__init__(
            n_train=n_train,
            n_test=n_test,
            noise=noise,
            test=test,
            **settings
        )
        if test == "probability":
            self.template = PromptTemplate(stimulus="Letters:", response="", complete=True)

    @property
    def effects(self) -> tuple[Effect, ...]:
        """The effects this test scores.

        Under the judgment test, how far grammaticality and chunk strength each raise
        the probability of answering grammatical. Under the probability test, the
        probability of the grammatical string of each pair, against chance.
        """
        if self.test == "probability":
            return (Effect("preference", "answer_probability", against=0.5),)
        return (
            Effect(
                "grammaticality",
                "option_probability",
                by="status",
                levels=(_GRAMMATICAL, _UNGRAMMATICAL),
                options={"option": _GRAMMATICAL}
            ),
            Effect(
                "chunk_strength",
                "option_probability",
                by="chunk",
                levels=("high", "low"),
                options={"option": _GRAMMATICAL}
            )
        )

    def trials(self, rng: np.random.Generator) -> Iterable[Trial]:
        """Tests new grammatical strings against their one-letter ungrammatical changes.

        Args:
            rng: Seeded for this participant. Its first draws choose what is studied.

        Yields:
            Under the judgment test, one trial per tested string, with its `status`
            and `chunk`. Under the probability test, one trial per grammatical string,
            whose options are it and its change, in random order.
        """
        studied, tested, changed = self._material(rng)
        if self.test == "probability":
            for text, variant in zip(tested, changed, strict=True):
                pair = [text, variant] if rng.random() < 0.5 else [variant, text]
                yield Trial(
                    Stimulus(text=""),
                    options=tuple(" ".join(string) for string in pair),
                    answer=" ".join(text),
                    item=text
                )
            return
        counts = _chunk_counts(studied)
        for strings, status in [(tested, _GRAMMATICAL), (changed, _UNGRAMMATICAL)]:
            ranked = sorted(strings, key=lambda text: (_chunk_strength(text, counts), text))
            low = set(ranked[: len(ranked) // 2])
            for text in strings:
                yield Trial(
                    _letters(text),
                    options=(_GRAMMATICAL, _UNGRAMMATICAL),
                    answer=status,
                    item=text,
                    condition={"status": status, "chunk": "low" if text in low else "high"}
                )

    def study(self, participant: int, block: int) -> tuple[Observation, ...]:
        """Gives the participant's study strings.

        Args:
            participant: The participant.
            block: The block about to be tested. There is only one.

        Returns:
            The study strings, in the order they are shown: each labelled grammatical
            under the judgment test, and bare under the probability test.
        """
        studied, _, _ = self._material(self._participant_rng(participant))
        if self.test == "probability":
            return tuple(Observation(Stimulus(text=""), " ".join(text)) for text in studied)
        return tuple(Observation(_letters(text), _GRAMMATICAL) for text in studied)

    def annotate(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Adds each test string's chunk strength.

        Args:
            frame: The results table of the test.

        Returns:
            The table with `chunk_strength`, and under the probability test
            `chunk_difference`.
        """
        counts = {
            participant: _chunk_counts(self._material(self._participant_rng(participant))[0])
            for participant in frame["participant"].unique()
        }
        frame = frame.copy()
        frame["chunk_strength"] = [
            _chunk_strength(item, counts[participant])
            for item, participant in zip(frame["item"], frame["participant"], strict=True)
        ]
        if self.test == "probability":
            variants = [
                next(option for option in options if option != answer).replace(" ", "")
                for options, answer in zip(frame["options"], frame["answer"], strict=True)
            ]
            frame["chunk_difference"] = frame["chunk_strength"] - [
                _chunk_strength(variant, counts[participant])
                for variant, participant in zip(variants, frame["participant"], strict=True)
            ]
        return frame

    def hypothesis_space(self) -> HypothesisSpace:
        """Gives the grammar under every relabelling of its letters.

        Returns:
            120 hypotheses, one per permutation of the five letters, over every string
            any participant sees. The unpermuted grammar is in group `true` and the
            others in `other`.
        """
        strings = sorted({
            text
            for participant in range(self.n_participants)
            for part in self._material(self._participant_rng(participant))
            for text in part
        })
        labels = []
        for permutation in itertools.permutations(_LETTERS):
            back = dict(zip(permutation, _LETTERS, strict=True))
            labels.append([0 if _accepts("".join(back[letter] for letter in text)) else 1 for text in strings])
        return HypothesisSpace(
            features=torch.stack([_bigram_counts(text) for text in strings]),
            hypotheses=torch.tensor(labels, dtype=torch.long),
            options=(_GRAMMATICAL, _UNGRAMMATICAL),
            groups=("true", *["other"] * (len(labels) - 1))
        )

    def _material(self, rng: np.random.Generator) -> tuple[list[str], list[str], list[str]]:
        # drawn first from the participant's stream, so trials and study agree
        grammatical, variants = _grammar_pool()
        order = rng.permutation(len(grammatical)).tolist()
        studied = [grammatical[i] for i in order[: self.n_train]]
        tested = [grammatical[i] for i in order[self.n_train : self.n_train + self.n_test]]
        counts = _chunk_counts(studied)
        everything = sorted({variant for text in tested for variant in variants[text]})
        changed = []
        for text in tested:
            free = [variant for variant in variants[text] if variant not in changed]
            free = free or [variant for variant in everything if variant not in changed]
            # the change nearest the original in chunk strength, so the two differ in grammaticality most
            target = _chunk_strength(text, counts)
            gaps = [abs(_chunk_strength(variant, counts) - target) for variant in free]
            closest = [variant for variant, gap in zip(free, gaps, strict=True) if gap == min(gaps)]
            changed.append(closest[int(rng.integers(len(closest)))])
        return studied, tested, changed


def _chunk_counts(strings: Sequence[str]) -> Counter[str]:
    return Counter(chunk for text in strings for chunk in _chunks(text))


def _chunk_strength(text: str, counts: Counter[str]) -> float:
    chunks = _chunks(text)
    return sum(counts[chunk] for chunk in chunks) / len(chunks)


def _chunks(text: str) -> list[str]:
    # the letter pairs and triples, as Knowlton and Squire (1996) count them
    return [text[start : start + size] for size in (2, 3) for start in range(len(text) - size + 1)]


def _accepts(text: str) -> bool:
    state = 0
    for letter in text:
        if state is None or letter not in _REBER[state]:
            return False
        state = _REBER[state][letter]
    return state is None


def _bigram_counts(text: str) -> torch.Tensor:
    firsts = ("^", *_LETTERS)
    seconds = (*_LETTERS, "$")
    counts = torch.zeros(len(firsts), len(seconds))
    padded = f"^{text}$"
    for first, second in itertools.pairwise(padded):
        counts[firsts.index(first), seconds.index(second)] += 1
    return counts.flatten()


def _letters(text: str) -> Stimulus:
    return Stimulus(text=" ".join(text), features=_bigram_counts(text))


@functools.cache
def _grammar_pool() -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    # every grammatical string, and its one-letter changes that leave the grammar
    grammatical = set()
    pending = [(0, "")]
    while pending:
        state, prefix = pending.pop()
        for letter, target in _REBER[state].items():
            text = prefix + letter
            if len(text) > _LONGEST:
                continue
            if target is None:
                grammatical.add(text)
            else:
                pending.append((target, text))
    variants = {
        text: {
            text[:position] + letter + text[position + 1 :]
            for position in range(len(text))
            for letter in _LETTERS
            if letter != text[position] and not _accepts(text[:position] + letter + text[position + 1 :])
        }
        for text in grammatical
    }
    # strings with equal letter-pair counts would look the same to a learner that reads features
    seen = Counter(
        tuple(_bigram_counts(text).tolist())
        for text in grammatical | {variant for changed in variants.values() for variant in changed}
    )

    def unique(text: str) -> bool:
        return seen[tuple(_bigram_counts(text).tolist())] == 1

    kept = {text: tuple(sorted(filter(unique, variants[text]))) for text in grammatical if unique(text)}
    usable = tuple(sorted(text for text, changed in kept.items() if changed))
    return usable, {text: kept[text] for text in usable}
