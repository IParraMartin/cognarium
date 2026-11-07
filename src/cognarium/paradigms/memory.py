"""Paradigms about memory: updating, span, scanning, recall and false memory."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from cognarium import context
from cognarium.metrics.statistics import contrast_table
from cognarium.paradigms.base import (
    BUILT_IN_STIMULI,
    Effect,
    Paradigm,
    load_items
)
from cognarium.stimuli import Stimulus, Trial

if TYPE_CHECKING:
    from cognarium.results import Results

_CONSONANTS = ("B", "F", "H", "K", "L", "M", "Q", "R")
_SCANNED = ("B", "C", "D", "F", "G", "H", "J", "K", "L", "M", "N", "P")
_DIGITS = tuple("0123456789")
_THIRDS = ("early", "middle", "late")
_ROLES = ("studied", "lure", "unrelated")
_SEPARATORS = re.compile(r"[,;\n]+")
_PUNCTUATION = re.compile(r"[^\w\s'-]")


class NBack(Paradigm):
    """Say whether each letter matches the one n letters back.

    Maps onto the n-back task: holding more letters in mind should make matches harder to
    judge, so performance should fall as n grows. What it can't capture: every earlier
    letter stays in the context, so the model never has to hold anything, and this
    measures picking out a position rather than a limited store. Surprisal stands in for
    response time. Each level of n is its own block in a random order, and the first n
    letters of a block can't match, so they are left out of the effect. Sequences are
    generated, not human-normed. d′ comes from `results.metrics("d_prime",
    positive="yes")`, by load.

    Args:
        levels: The values of n, one block each.
        sequence_length: How many letters each block shows.
        target_rate: The share of letters, after the first n, that match.
        alphabet: The letters shown.
        **settings: What every paradigm takes.

    Raises:
        ValueError: If a level is below 1, or a block would have no match.
    """

    stimuli_label = BUILT_IN_STIMULI
    context = context.cumulative
    ordered = True
    group_by = ("load",)

    def __init__(
        self,
        *,
        levels: Sequence[int] = (1, 2),
        sequence_length: int = 20,
        target_rate: float = 0.33,
        alphabet: Sequence[str] = _CONSONANTS,
        **settings: Any
    ) -> None:
        levels = tuple(sorted(set(levels)))
        if not levels or levels[0] < 1:
            raise ValueError(f"Each level of n must be at least 1, got {list(levels)}.")
        if round((sequence_length - levels[-1]) * target_rate) < 1:
            raise ValueError(f"sequence_length={sequence_length} leaves no matches at n={levels[-1]}. Make it longer.")
        if len(set(alphabet)) < 3:
            raise ValueError("The alphabet needs at least three letters.")
        super().__init__(
            levels=levels,
            sequence_length=sequence_length,
            target_rate=target_rate,
            alphabet=tuple(alphabet),
            **settings
        )

    @property
    def effects(self) -> tuple[Effect, ...]:
        """The cost of the highest load against the lowest, after each block's first letters."""
        return (
            Effect(
                "load_cost",
                "answer_surprisal",
                by="load",
                levels=(self.levels[-1], self.levels[0]),
                where={"scored": True}
            ),
            Effect(
                "errors",
                "answer_margin",
                by="load",
                levels=(self.levels[0], self.levels[-1]),
                where={"scored": True}
            )
        )

    def trials(self, rng: np.random.Generator) -> list[Trial]:
        """Draws one block per level, in a random order.

        Args:
            rng: Seeded for this participant.

        Returns:
            The trials, with the conditions `load`, `trial_type`, which is `target`,
            `lure` or `filler`, `letter`, `position` and `scored`. The item is the position
            in the block, so loads pair up by position.
        """
        trials = []
        for block, level in enumerate(rng.permutation(self.levels).tolist()):
            trials.extend(self._block(level, block, rng))
        return trials

    def _block(
        self,
        n: int,
        block: int,
        rng: np.random.Generator
    ) -> list[Trial]:
        length = self.sequence_length
        n_targets = round((length - n) * self.target_rate)
        targets = set((n + rng.choice(length - n, size=n_targets, replace=False)).tolist())
        letters: list[str] = []
        for position in range(length):
            if position in targets:
                letters.append(letters[position - n])
                continue
            # a letter that isn't a target must differ from the one n back
            banned = letters[position - n] if position >= n else None
            letters.append(str(rng.choice([letter for letter in self.alphabet if letter != banned])))
        trials = []
        for position, letter in enumerate(letters):
            nearby = [other for other in (position - n + 1, position - n - 1) if 0 <= other < position]
            if position in targets:
                kind = "target"
            else:
                kind = "lure" if any(letters[other] == letter for other in nearby) else "filler"
            trials.append(
                Trial(
                    Stimulus(text=f"{n}-back. Letter: {letter} Does it match the letter {n} back?"),
                    options=("yes", "no"),
                    answer="yes" if position in targets else "no",
                    condition={
                        "load": n,
                        "trial_type": kind,
                        "letter": letter,
                        "position": position,
                        "scored": position >= n
                    },
                    item=position,
                    block=block
                )
            )
        return trials


class Span(Paradigm):
    """Repeat a list of digits in order.

    Maps onto the digit span task: longer lists should be repeated correctly less often.
    What it can't capture: the list stays in the prompt while the model answers, so this
    measures copying a sequence under interference rather than memory decay. A response
    counts as correct only with every digit in order. Lists are generated, not
    human-normed.

    Args:
        lengths: The list lengths, each at most 10.
        n_per_length: How many lists of each length each participant sees.
        **settings: What every paradigm takes. `max_new_tokens` defaults to twice the
            longest list plus four.

    Raises:
        ValueError: If a length is below 1 or above 10, or `n_per_length` is below 2.
    """

    stimuli_label = BUILT_IN_STIMULI
    response_mode = "generate"
    group_by = ("length",)

    def __init__(
        self,
        *,
        lengths: Sequence[int] = (3, 5, 7),
        n_per_length: int = 8,
        **settings: Any
    ) -> None:
        lengths = tuple(sorted(set(lengths)))
        if not lengths or lengths[0] < 1 or lengths[-1] > len(_DIGITS):
            raise ValueError(f"Each length must be between 1 and 10, so no digit repeats, got {list(lengths)}.")
        if len(lengths) < 2 or n_per_length < 2:
            raise ValueError("Give at least two lengths and two lists per length, so the effect has an interval.")
        settings.setdefault("max_new_tokens", 2 * lengths[-1] + 4)
        super().__init__(lengths=lengths, n_per_length=n_per_length, **settings)

    @property
    def effects(self) -> tuple[Effect, ...]:
        """How much more often the shortest lists are repeated correctly than the longest."""
        return (Effect(
            "length_cost",
            "correct",
            by="length",
            levels=(self.lengths[0], self.lengths[-1])
        ),)

    def trials(self, rng: np.random.Generator) -> list[Trial]:
        """Draws `n_per_length` lists of each length, with no digit twice in a list.

        Args:
            rng: Seeded for this participant.

        Returns:
            The trials, whose answer is the digits joined by spaces, with the condition
            `length`.
        """
        trials = []
        for length in self.lengths:
            for copy in range(self.n_per_length):
                digits = rng.choice(_DIGITS, size=length, replace=False).tolist()
                trials.append(
                    Trial(
                        Stimulus(text=f"Remember these digits: {' '.join(digits)}. Repeat them in order."),
                        answer=" ".join(digits),
                        condition={"length": length},
                        item=f"{length}-{copy}"
                    )
                )
        return trials

    def annotate(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Reads the digits out of each response.

        Args:
            frame: The results table of one step.

        Returns:
            The table with `parsed`, the digits in the order written, and `correct`, which
            is 1 when they equal the list and 0 otherwise.
        """
        parsed = [re.findall("[0-9]", str(text)) for text in frame["response"]]
        correct = [float(digits == answer.split()) for digits, answer in zip(parsed, frame["answer"], strict=True)]
        return frame.assign(parsed=parsed, correct=correct)

    def memory_span(self, results: Results) -> int:
        """Estimates the span as the longest length repeated correctly at least half the time.

        Args:
            results: The results of a run of this paradigm.

        Returns:
            The span, or 0 if no length reaches half.
        """
        rates = results.to_pandas().groupby("length")["correct"].mean()
        passed = [int(length) for length, rate in rates.items() if rate >= 0.5]
        return max(passed, default=0)


class Sternberg(Paradigm):
    """Say whether a probe letter was in a memory set.

    Maps onto the Sternberg memory scanning task: larger sets should make the judgment
    harder, which in people shows as slower responses. What it can't capture: the set stays
    in the prompt, and there is no response time, so surprisal stands in for scanning time.
    Half the probes come from the set. Sets are generated, not human-normed. d′ comes from
    `results.metrics("d_prime", positive="yes")`, by set size.

    Args:
        set_sizes: How many letters each set holds.
        n_per_size: How many trials of each set size each participant sees.
        **settings: What every paradigm takes.

    Raises:
        ValueError: If a set size is below 1 or leaves no letter to be an absent probe, or
            `n_per_size` is below 4.
    """

    stimuli_label = BUILT_IN_STIMULI
    group_by = ("set_size",)

    def __init__(
        self,
        *,
        set_sizes: Sequence[int] = (2, 6),
        n_per_size: int = 16,
        **settings: Any
    ) -> None:
        set_sizes = tuple(sorted(set(set_sizes)))
        if len(set_sizes) < 2 or set_sizes[0] < 1 or set_sizes[-1] >= len(_SCANNED):
            raise ValueError(f"Give two or more set sizes between 1 and {len(_SCANNED) - 1}, got {list(set_sizes)}.")
        if n_per_size < 4:
            raise ValueError(f"n_per_size must be at least 4, got {n_per_size}.")
        super().__init__(set_sizes=set_sizes, n_per_size=n_per_size, **settings)

    @property
    def effects(self) -> tuple[Effect, ...]:
        """The cost of the largest set against the smallest."""
        return (
            Effect(
                "set_size_cost",
                "answer_surprisal",
                by="set_size",
                levels=(self.set_sizes[-1], self.set_sizes[0])
            ),
            Effect(
                "errors",
                "answer_margin",
                by="set_size",
                levels=(self.set_sizes[0], self.set_sizes[-1])
            )
        )

    def trials(self, rng: np.random.Generator) -> list[Trial]:
        """Draws the sets and probes, with every other probe taken from its set.

        Args:
            rng: Seeded for this participant.

        Returns:
            The trials, with the conditions `set_size` and `probe`, which is `present` or
            `absent`.
        """
        trials = []
        for size in self.set_sizes:
            for copy in range(self.n_per_size):
                members = rng.choice(_SCANNED, size=size, replace=False).tolist()
                present = copy % 2 == 0
                pool = members if present else [letter for letter in _SCANNED if letter not in members]
                probe = str(rng.choice(pool))
                trials.append(
                    Trial(
                        Stimulus(text=f"Memory set: {' '.join(members)}. Probe: {probe}. Was the probe in the set?"),
                        options=("yes", "no"),
                        answer="yes" if present else "no",
                        condition={"set_size": size, "probe": "present" if present else "absent"},
                        item=f"{size}-{copy}"
                    )
                )
        return trials


class FreeRecall(Paradigm):
    """Study a list of words, then write every word you remember.

    Maps onto immediate free recall: words at the end of a list should be recalled more
    often than those in the middle, which is the recency effect, and words at the start
    often are too, which is primacy. What it can't capture: the list stays in the prompt,
    so recall here is retrieval from context rather than memory, and there's no delay or
    distractor task. Each participant's lists share no words. The word pool was written for
    cognarium and is not human-normed.

    Args:
        list_length: How many words each list holds, at least 3.
        n_lists: How many lists each participant studies.
        **settings: What every paradigm takes, such as `stimuli` with the column `word`
            for your own pool. `max_new_tokens` defaults to 64.

    Raises:
        ValueError: If a list is shorter than 3, there are fewer than two lists, or the
            pool doesn't hold enough words.
    """

    stimuli_label = BUILT_IN_STIMULI
    response_mode = "generate"
    item_columns = ("word",)
    group_by = ()
    effects = (
        Effect(
            "recency",
            "recall",
            by="third",
            levels=("late", "middle")
        ),
        Effect(
            "primacy",
            "recall",
            by="third",
            levels=("early", "middle")
        )
    )

    def __init__(
        self,
        *,
        list_length: int = 12,
        n_lists: int = 8,
        **settings: Any
    ) -> None:
        if list_length < 3 or n_lists < 2:
            raise ValueError("Free recall needs lists of at least 3 words, and at least two lists.")
        settings.setdefault("max_new_tokens", 64)
        super().__init__(list_length=list_length, n_lists=n_lists, **settings)
        pool = len({row["word"] for row in self.item_rows(np.random.default_rng(0))})
        if pool < list_length * n_lists:
            raise ValueError(
                f"{n_lists} lists of {list_length} words need {list_length * n_lists} different words, "
                f"but the pool has {pool}."
            )

    def default_items(self, rng: np.random.Generator) -> list[dict[str, Any]]:
        """Gives the built-in pool of common concrete nouns.

        Args:
            rng: Unused, since the pool is fixed.

        Returns:
            Rows with `word`.
        """
        return load_items("free_recall")

    def trials(self, rng: np.random.Generator) -> list[Trial]:
        """Draws each participant's lists from the pool, with no word in two lists.

        Args:
            rng: Seeded for this participant.

        Returns:
            The trials, with the conditions `list` and `studied`, the list in study order.
        """
        pool = sorted({row["word"] for row in self.item_rows(rng)})
        drawn = rng.choice(pool, size=self.list_length * self.n_lists, replace=False).tolist()
        trials = []
        for index in range(self.n_lists):
            words = drawn[index * self.list_length : (index + 1) * self.list_length]
            text = f"Study these words: {', '.join(words)}. Now write every word you remember, separated by commas."
            trials.append(Trial(Stimulus(text=text), condition={"list": index, "studied": words}, item=index))
        return trials

    def annotate(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Matches each response against its studied list.

        Args:
            frame: The results table of one step.

        Returns:
            The table with `recalled`, the studied words written, in the order first
            written, `n_recalled`, `n_intrusions`, which counts written words that weren't
            studied, `n_repeats`, and the share recalled from the `early`, `middle` and
            `late` thirds of the list.
        """
        parsed = [_recall(str(text), list(studied)) for text, studied in zip(frame["response"], frame["studied"], strict=True)]
        columns = {
            "recalled": [row["recalled"] for row in parsed],
            "n_recalled": [len(row["recalled"]) for row in parsed],
            "n_intrusions": [row["intrusions"] for row in parsed],
            "n_repeats": [row["repeats"] for row in parsed]
        }
        for third in _THIRDS:
            columns[third] = [row[third] for row in parsed]
        return frame.assign(**columns)

    def score(self, results: Results) -> pd.DataFrame:
        """Compares recall across the thirds of each list.

        Args:
            results: The results of a run of this paradigm.

        Returns:
            One row per effect, with the columns of `Results.contrast`, where each list is a
            group and `measure` is `recall`.
        """
        frame = results.to_pandas()
        long = pd.DataFrame([
            {
                "item": f"{participant}-{item}",
                "third": third,
                "recall": row[third]
            }
            for participant, item, (_, row) in zip(
                frame["participant"],
                frame["item"],
                frame.iterrows(),
                strict=True
            )
            for third in _THIRDS
        ])
        tables = [
            contrast_table(
                long,
                "recall",
                by="third",
                levels=effect.levels
            ).assign(effect=effect.name, measure="recall", proxy_for=None)
            for effect in self.effects
        ]
        combined = pd.concat(tables, ignore_index=True)
        return combined[["effect", *(column for column in combined.columns if column != "effect")]]


class DRM(Paradigm):
    """Study lists of related words, then judge whether test words were on the list.

    Maps onto the Deese-Roediger-McDermott false memory task: an unstudied word that every
    list word relates to, the lure, should be called old more often than an unrelated new
    word. What it can't capture: the list stays in the prompt, so a false alarm reflects
    association in the text rather than a memory that was never formed. No word lists ship
    with cognarium, because no open source of them has been verified. Pass your own
    through `stimuli` with the columns `list`, `word` and `role`, where role is `studied`,
    `lure` or `unrelated`. cognarium doesn't check them, so the stimuli are not
    human-normed unless your lists are. Every word is tested once, after its list.

    Args:
        **settings: What every paradigm takes, including the required `stimuli`.

    Raises:
        ValueError: If `stimuli` is missing, a role is unknown, or a list has no studied
            words.
    """

    item_columns = ("list", "word", "role")
    group_by = ()
    effects = (
        Effect(
            "false_memory",
            "option_probability",
            by="role",
            levels=("lure", "unrelated"),
            options={"option": "old"}
        ),
        Effect(
            "true_memory",
            "option_probability",
            by="role",
            levels=("studied", "unrelated"),
            options={"option": "old"}
        )
    )

    def __init__(self, **settings: Any) -> None:
        if settings.get("stimuli") is None:
            raise ValueError(
                "DRM needs word lists through stimuli=, with the columns list, word and role. No open "
                "source of the lists has been verified, so none ship with cognarium."
            )
        super().__init__(**settings)
        for position, row in enumerate(self.rows):
            if row["role"] not in _ROLES:
                raise ValueError(f"Row {position} has the role '{row['role']}', which isn't one of {list(_ROLES)}.")
        for name in dict.fromkeys(row["list"] for row in self.rows):
            if not self._studied(name):
                raise ValueError(f"The list '{name}' has no studied words.")

    def trials(self, rng: np.random.Generator) -> list[Trial]:
        """Tests every word after its list.

        Args:
            rng: Unused, since the rows set every trial.

        Returns:
            The trials, with the conditions `role` and `list`.
        """
        trials = []
        for row in self.rows:
            studied = self._studied(row["list"])
            text = f"Study these words: {', '.join(studied)}. Was this word on the list? Word: {row['word']}"
            trials.append(
                Trial(
                    Stimulus(text=text),
                    options=("old", "new"),
                    answer="old" if row["role"] == "studied" else "new",
                    condition={"role": row["role"], "list": str(row["list"])},
                    item=f"{row['list']}-{row['word']}"
                )
            )
        return trials

    def _studied(self, name: Any) -> list[str]:
        return [row["word"] for row in self.rows if row["list"] == name and row["role"] == "studied"]


def _recall(text: str, studied: list[str]) -> dict[str, Any]:
    known = set(studied)
    written = []
    for token in _SEPARATORS.split(text):
        cleaned = _PUNCTUATION.sub("", token).strip().lower()
        # words run together with spaces are split, unless together they form a studied word
        written.extend([cleaned] if cleaned in known or " " not in cleaned else cleaned.split())
    written = [word for word in written if word]
    recalled = list(dict.fromkeys(word for word in written if word in known))
    thirds = np.array_split(np.arange(len(studied)), len(_THIRDS))
    shares = {
        third: float(np.mean([studied[position] in recalled for position in positions]))
        for third, positions in zip(_THIRDS, thirds, strict=True)
    }
    return {
        "recalled": recalled,
        "intrusions": sum(word not in known for word in written),
        "repeats": sum(word in known for word in written) - len(recalled),
        **shares
    }
