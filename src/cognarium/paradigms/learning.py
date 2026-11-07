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
from cognarium.responses import PromptTemplate, Query
from cognarium.settings import Settings
from cognarium.stimuli import Observation, Stimulus, Trial

if TYPE_CHECKING:
    from cognarium.tasks.shepard import ShepardCategories

# separate seed streams, so adding draws to one never shifts another
_TRANSMIT_STREAM = 0
_RESPONSE_STREAM = 1
_GROUP_STREAM = 2
_INIT_STREAM = 3
# the trial engine draws each participant's trials from stream 4
_LANGUAGE_STREAM = 5

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
_GRAMMATICAL = "grammatical"
_UNGRAMMATICAL = "ungrammatical"
_SYLLABLE_ONSETS = ("b", "d", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v")
_SYLLABLE_VOWELS = ("a", "e", "i", "o", "u")
_FAMILIAR = "familiar"
_UNFAMILIAR = "unfamiliar"
_ALTERNATIVE_PARTITIONS = 63
_STREAM_ATTEMPTS = 1000


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
                    self.template.render_observation(observation)
                    for observation in self.study(participant, block)
                    if observation.stimulus.text is not None
                )
        return lines


class CategoryLearning(_StudyThenTest):
    """Learn a category structure from corrective feedback, block by block.

    Every block shows each object once, in a new order, and asks for its category.
    Before a block, the subject sees every earlier block's objects with their correct
    labels. Each participant is a separate learner with their own orders.

    Maps onto supervised category learning with the Shepard, Hovland and Jenkins
    (1961) structures, where people learn type I fastest and type VI slowest. What it
    can't capture: a person gets feedback after each response and forgets, while the
    model reads its whole history at once, so there is no forgetting and no effect of
    its own errors. The objects are the eight abstract ones of the task, not
    human-normed.

    Args:
        task: The category structure to learn. Its labelling is the correct answer.
        blocks: How many times every object is tested.
        **settings: What every paradigm takes, such as `n_participants`.

    Raises:
        ValueError: If there are fewer than two blocks.
    """

    group_by = ("block",)

    def __init__(
        self,
        task: ShepardCategories,
        blocks: int = 6,
        **settings: Any
    ) -> None:
        if blocks < 2:
            raise ValueError(f"blocks must be at least 2, so the first and last can be compared, got {blocks}.")
        self.task = task
        super().__init__(blocks=blocks, **settings)
        self.template = task.template

    @property
    def effects(self) -> tuple[Effect, ...]:
        """The probability of the correct label in the last block against the first."""
        return (
            Effect(
                "learning",
                "answer_probability",
                by="block",
                levels=(self.blocks - 1, 0)
            ),
        )

    def trials(self, rng: np.random.Generator) -> Iterable[Trial]:
        """Tests every object once per block, against its correct label.

        Args:
            rng: Unused, since each block's order is shuffled afterwards.

        Yields:
            One trial per object and block.
        """
        for block in range(self.blocks):
            for item, stimulus in enumerate(self.task.stimuli):
                yield Trial(
                    stimulus,
                    options=self.task.options,
                    answer=self.task.seed_labels[item],
                    item=item,
                    block=block
                )

    def study(self, participant: int, block: int) -> tuple[Observation, ...]:
        """Gives every earlier block's objects with their correct labels.

        Args:
            participant: The participant.
            block: The block about to be tested.

        Returns:
            The labelled objects, in the order the participant was tested on them.
        """
        return tuple(
            Observation(trial.stimulus, trial.answer)
            for trial in self.sequences()[participant]
            if trial.block < block
        )

    def config(self) -> dict[str, Any]:
        """Describes the paradigm in JSON-safe form.

        Returns:
            What every paradigm reports, and the task.
        """
        return {**super().config(), "task": self.task.config()}

    def hypothesis_space(self) -> HypothesisSpace:
        """Gives the hypotheses a Bayesian learner considers on the task.

        Returns:
            The task's hypothesis space.
        """
        return self.task.hypothesis_space()


class ArtificialGrammar(_StudyThenTest):
    """Study strings from a finite-state grammar, then judge new ones.

    The grammar is Reber's (1967), over the letters T, P, S, X and V, with strings of
    3 to 8 letters. Each participant studies their own strings, each labelled
    grammatical, then judges new grammatical strings and as many ungrammatical ones,
    each made by changing one letter of a tested grammatical string so that it leaves
    the grammar. Letters are written apart, as in `Letters: T X S`, so a subject that
    reads words sees letters. A string's features count its letter pairs, including
    its first and last letters, and strings that would share features are never used.

    Maps onto artificial grammar learning, where people trained on grammatical strings
    later judge new ones above chance without being able to say the rules. What it
    can't capture: people aren't told the strings follow rules, while here every study
    string carries the same label, and the measure is a judgment probability rather
    than the endorsement rate. The strings come from the grammar, not human-normed.

    Args:
        n_train: How many strings each participant studies.
        n_test: How many grammatical strings are tested, and as many ungrammatical ones.
        noise: How often a Bayesian learner assumes a label it sees is wrong. The
            labels themselves never change.
        **settings: What every paradigm takes, such as `n_participants`.

    Raises:
        ValueError: If a count is below 1, there aren't enough grammatical strings for
            both counts, or `noise` is outside [0, 1).
    """

    single_label_training = True
    template = PromptTemplate(stimulus="Letters: {text}", response="Judgment:")
    effects = (
        Effect(
            "grammaticality",
            "option_probability",
            by="status",
            levels=(_GRAMMATICAL, _UNGRAMMATICAL),
            options={"option": _GRAMMATICAL}
        ),
    )

    def __init__(
        self,
        n_train: int = 16,
        n_test: int = 8,
        noise: float = 0.2,
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
        super().__init__(
            n_train=n_train,
            n_test=n_test,
            noise=noise,
            **settings
        )

    def trials(self, rng: np.random.Generator) -> Iterable[Trial]:
        """Tests new grammatical strings and their one-letter ungrammatical variants.

        Args:
            rng: Seeded for this participant. Its first draws choose what is studied.

        Yields:
            One trial per tested string, with its `status`.
        """
        _, tested, changed = self._material(rng)
        for strings, status in [(tested, _GRAMMATICAL), (changed, _UNGRAMMATICAL)]:
            for text in strings:
                yield Trial(
                    _letters(text),
                    options=(_GRAMMATICAL, _UNGRAMMATICAL),
                    answer=status,
                    item=text,
                    condition={"status": status}
                )

    def study(self, participant: int, block: int) -> tuple[Observation, ...]:
        """Gives the participant's study strings, each labelled grammatical.

        Args:
            participant: The participant.
            block: The block about to be tested. There is only one.

        Returns:
            The study strings, in the order they are shown.
        """
        studied, _, _ = self._material(self._participant_rng(participant))
        return tuple(Observation(_letters(text), _GRAMMATICAL) for text in studied)

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
        everything = sorted({variant for text in tested for variant in variants[text]})
        changed = []
        for text in tested:
            free = [variant for variant in variants[text] if variant not in changed]
            free = free or [variant for variant in everything if variant not in changed]
            changed.append(free[int(rng.integers(len(free)))])
        return studied, tested, changed


class StatisticalLearning(_StudyThenTest):
    """Find the words in a stream of syllables, then judge words and part-words.

    The language has nonsense words of three consonant-vowel syllables, each syllable
    used in one word only, drawn from the seed. Each participant hears their own
    stream, in which every word occurs equally often and never twice in a row. The
    stream is shown as every run of three syllables in it, each labelled familiar.
    Tests are the words and as many part-words, which join the end of one word to the
    start of the word that followed it in the stream, so they occurred too, but less
    often. A learner who found the words calls words familiar and part-words not.

    Maps onto statistical word segmentation (Saffran, Aslin and Newport, 1996), where
    syllables inside a word predict each other better than syllables across a word
    boundary. What it can't capture: the stream is shown as overlapping windows of
    text rather than continuous speech, because every subject type must read the same
    trials, and each window is judged rather than two items compared. The syllables
    are generated, not human-normed.

    Args:
        n_words: How many words the language has.
        stream_words: How many words the stream has. A multiple of `n_words`.
        noise: How often a Bayesian learner assumes a label it sees is wrong. The
            labels themselves never change.
        **settings: What every paradigm takes, such as `n_participants`.

    Raises:
        ValueError: If there are fewer than three words or more than the syllables
            allow, the stream isn't at least three rounds of every word, or `noise` is
            outside [0, 1).
    """

    single_label_training = True
    template = PromptTemplate(stimulus="Sounds: {text}", response="Judgment:")
    effects = (
        Effect(
            "segmentation",
            "option_probability",
            by="test_type",
            levels=("word", "part_word"),
            options={"option": _FAMILIAR}
        ),
    )

    def __init__(
        self,
        n_words: int = 4,
        stream_words: int = 24,
        noise: float = 0.3,
        **settings: Any
    ) -> None:
        most = len(_SYLLABLE_ONSETS) * len(_SYLLABLE_VOWELS) // 3
        if not 3 <= n_words <= most:
            raise ValueError(
                f"n_words must be between 3 and {most}, so part-words can be rarer than words, got {n_words}."
            )
        if stream_words % n_words != 0 or stream_words < 3 * n_words:
            raise ValueError(
                f"stream_words must be a multiple of n_words and at least {3 * n_words}, so every word "
                f"occurs equally often and is followed by more than one other, got {stream_words}."
            )
        if not 0 <= noise < 1:
            raise ValueError(f"noise must be in [0, 1), got {noise}.")
        super().__init__(
            n_words=n_words,
            stream_words=stream_words,
            noise=noise,
            **settings
        )

    @property
    def words(self) -> tuple[tuple[str, ...], ...]:
        """The language's words, each a tuple of three syllables."""
        cached = self.__dict__.get("_words")
        if cached is None:
            rng = np.random.default_rng([self.settings.seed, _LANGUAGE_STREAM])
            chosen = rng.choice(len(_SYLLABLES), size=3 * self.n_words, replace=False).tolist()
            syllables = [_SYLLABLES[i] for i in chosen]
            cached = tuple(tuple(syllables[start : start + 3]) for start in range(0, len(syllables), 3))
            self._words = cached
        return cached

    def stream(self, participant: int) -> tuple[str, ...]:
        """Gives the syllables a participant hears, in order.

        Args:
            participant: The participant.

        Returns:
            Every syllable of the stream.
        """
        return self._material(self._participant_rng(participant))[0]

    def trials(self, rng: np.random.Generator) -> Iterable[Trial]:
        """Tests every word and one part-word that ends each word.

        Args:
            rng: Seeded for this participant. Its first draws make the stream.

        Yields:
            One trial per word and part-word, with its `test_type`.
        """
        _, part_words = self._material(rng)
        options = (_FAMILIAR, _UNFAMILIAR)
        for syllables, test_type, answer in [
            *((word, "word", _FAMILIAR) for word in self.words),
            *((part, "part_word", _UNFAMILIAR) for part in part_words)
        ]:
            yield Trial(
                self._sounds(syllables),
                options=options,
                answer=answer,
                item=" ".join(syllables),
                condition={"test_type": test_type}
            )

    def study(self, participant: int, block: int) -> tuple[Observation, ...]:
        """Gives every run of three syllables in the stream, each labelled familiar.

        Args:
            participant: The participant.
            block: The block about to be tested. There is only one.

        Returns:
            The windows, in stream order.
        """
        return tuple(Observation(self._sounds(window), _FAMILIAR) for window in _windows(self.stream(participant)))

    def hypothesis_space(self) -> HypothesisSpace:
        """Gives the true words and other ways of grouping the syllables into words.

        Returns:
            The true word set, in group `true`, and 63 seeded partitions of the same
            syllables, in group `other`. A run of three syllables is familiar under a
            hypothesis when it is one of its words.
        """
        windows = sorted({
            window
            for participant in range(self.n_participants)
            for window in _windows(self.stream(participant))
        } | set(self.words))
        partitions = [set(self.words)]
        rng = np.random.default_rng([self.settings.seed, _LANGUAGE_STREAM, 1])
        syllables = [syllable for word in self.words for syllable in word]
        while len(partitions) <= _ALTERNATIVE_PARTITIONS:
            shuffled = [syllables[i] for i in rng.permutation(len(syllables)).tolist()]
            partition = {tuple(shuffled[start : start + 3]) for start in range(0, len(shuffled), 3)}
            if partition not in partitions:
                partitions.append(partition)
        return HypothesisSpace(
            features=torch.stack([self._sounds(window).features for window in windows]),
            hypotheses=torch.tensor(
                [[0 if window in partition else 1 for window in windows] for partition in partitions],
                dtype=torch.long
            ),
            options=(_FAMILIAR, _UNFAMILIAR),
            groups=("true", *["other"] * _ALTERNATIVE_PARTITIONS)
        )

    def _sounds(self, syllables: Sequence[str]) -> Stimulus:
        inventory = sorted(syllable for word in self.words for syllable in word)
        features = torch.zeros(3, len(inventory))
        for position, syllable in enumerate(syllables):
            features[position, inventory.index(syllable)] = 1.0
        return Stimulus(text=" ".join(syllables), features=features.flatten())

    def _material(self, rng: np.random.Generator) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
        # drawn first from the participant's stream, so trials and the stream agree
        rounds = self.stream_words // self.n_words
        for _ in range(_STREAM_ATTEMPTS):
            order = []
            for _ in range(rounds):
                round_order = rng.permutation(self.n_words).tolist()
                while order and round_order[0] == order[-1]:
                    round_order = rng.permutation(self.n_words).tolist()
                order.extend(round_order)
            pairs = list(itertools.pairwise(order))
            followers = [Counter(after for before, after in pairs if before == word) for word in range(self.n_words)]
            if all(len(counts) >= 2 for counts in followers):
                break
        else:
            raise ValueError(
                "Couldn't build a stream in which every word is followed by more than one other. "
                "Use more stream_words."
            )
        part_words = []
        for word, counts in enumerate(followers):
            follower = min(counts, key=lambda after: (counts[after], after))
            joined = (*self.words[word], *self.words[follower])
            part_words.append(joined[1:4] if word % 2 == 0 else joined[2:5])
        stream = tuple(syllable for word in order for syllable in self.words[word])
        return stream, part_words


_SYLLABLES = tuple(onset + vowel for onset in _SYLLABLE_ONSETS for vowel in _SYLLABLE_VOWELS)


def _windows(syllables: Sequence[str]) -> list[tuple[str, ...]]:
    return [tuple(syllables[start : start + 3]) for start in range(len(syllables) - 2)]


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
