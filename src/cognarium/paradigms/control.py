"""Paradigms about cognitive control: conflict, inhibition and switching tasks."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from typing import Any

import numpy as np

from cognarium import context
from cognarium.paradigms.base import BUILT_IN_STIMULI, Effect, Paradigm
from cognarium.stimuli import Stimulus, Trial

_COLORS = ("red", "blue", "green", "yellow")
_GO_LETTERS = ("K", "M", "P", "R", "T", "L", "N", "B")
_VOWELS = ("A", "E", "I", "U")
_CONSONANTS = ("G", "K", "M", "R")
_PRESS = "press"
_WITHHOLD = "withhold"


class Stroop(Paradigm):
    """Name the ink colour of a colour word.

    Maps onto the Stroop colour-naming task: the word and its ink either match or
    conflict, and conflict should make the ink harder to name. What it can't capture:
    the model reads a description of the ink rather than seeing a colour, so this
    measures conflict between two described attributes, and surprisal stands in for
    naming time. The built-in items are every pairing of the colours, not human-normed.

    Args:
        colors: The colour names, which are also the options.
        n_repeats: How many times the balanced set of items repeats.
        **settings: What every paradigm takes, such as `stimuli` with the columns `word`
            and `ink`. A word that isn't a colour makes a neutral trial.

    Raises:
        ValueError: If there are fewer than two colours, or a row's ink isn't one of them.
    """

    stimuli_label = BUILT_IN_STIMULI
    item_columns = ("word", "ink")
    group_by = ("congruency",)
    effects = (
        Effect(
            "interference",
            "answer_surprisal",
            by="congruency",
            levels=("incongruent", "congruent")
        ),
        Effect(
            "errors",
            "answer_margin",
            by="congruency",
            levels=("congruent", "incongruent"),
            proxy_for="error rate"
        )
    )

    def __init__(
        self,
        *,
        colors: Sequence[str] = _COLORS,
        n_repeats: int = 1,
        **settings: Any
    ) -> None:
        super().__init__(colors=_check_choices(colors, "colors"), n_repeats=_check_repeats(n_repeats), **settings)
        _check_values(self.rows, "ink", self.colors)

    def default_items(self, rng: np.random.Generator) -> list[dict[str, Any]]:
        """Pairs every colour word with every ink, matching pairs as often as mismatched ones.

        Args:
            rng: Unused, since the items are fixed.

        Returns:
            Rows with `word` and `ink`.
        """
        return _balanced(
            self.colors,
            self.n_repeats,
            "word",
            "ink"
        )

    def trials(self, rng: np.random.Generator) -> list[Trial]:
        """Writes one trial per row.

        Args:
            rng: Passed to `item_rows`.

        Returns:
            The trials, with the conditions `congruency`, `word` and `ink`.
        """
        return [
            Trial(
                Stimulus(text=f"The word {row['word'].upper()} is printed in {row['ink']} ink. What color is the ink?"),
                options=self.colors,
                answer=row["ink"],
                condition={
                    "congruency": _congruency(row["word"], row["ink"], self.colors),
                    "word": row["word"],
                    "ink": row["ink"]
                },
                item=f"{row['word']}-{row['ink']}"
            )
            for row in self.item_rows(rng)
        ]


class Flanker(Paradigm):
    """Name the middle letter of a row of letters.

    Maps onto the Eriksen flanker task: the letters around the target either match it or
    call for another response. What it can't capture: the letters are tokens in a row
    rather than positions in space, so there is no spatial attention, and surprisal stands
    in for response time. The built-in items are every target and flanker pairing, not
    human-normed.

    Args:
        letters: The letters, which are also the options.
        flankers: How many letters sit on each side of the target.
        n_repeats: How many times the balanced set of items repeats.
        **settings: What every paradigm takes, such as `stimuli` with the columns
            `target` and `flanker`.

    Raises:
        ValueError: If there are fewer than two letters, `flankers` is below 1, or a row's
            target isn't one of the letters.
    """

    stimuli_label = BUILT_IN_STIMULI
    item_columns = ("target", "flanker")
    group_by = ("congruency",)
    effects = (
        Effect(
            "interference",
            "answer_surprisal",
            by="congruency",
            levels=("incongruent", "congruent")
        ),
        Effect(
            "errors",
            "answer_margin",
            by="congruency",
            levels=("congruent", "incongruent"),
            proxy_for="error rate"
        )
    )

    def __init__(
        self,
        *,
        letters: Sequence[str] = ("H", "S", "K", "C"),
        flankers: int = 2,
        n_repeats: int = 1,
        **settings: Any
    ) -> None:
        if flankers < 1:
            raise ValueError(f"flankers must be at least 1, got {flankers}.")
        super().__init__(
            letters=_check_choices(letters, "letters"),
            flankers=flankers,
            n_repeats=_check_repeats(n_repeats),
            **settings
        )
        _check_values(self.rows, "target", self.letters)

    def default_items(self, rng: np.random.Generator) -> list[dict[str, Any]]:
        """Pairs every target with every flanker, matching pairs as often as mismatched ones.

        Args:
            rng: Unused, since the items are fixed.

        Returns:
            Rows with `target` and `flanker`.
        """
        return _balanced(
            self.letters,
            self.n_repeats,
            "flanker",
            "target"
        )

    def trials(self, rng: np.random.Generator) -> list[Trial]:
        """Writes one trial per row.

        Args:
            rng: Passed to `item_rows`.

        Returns:
            The trials, with the conditions `congruency`, `target` and `flanker`.
        """
        trials = []
        for row in self.item_rows(rng):
            side = row["flanker"] * self.flankers
            trials.append(
                Trial(
                    Stimulus(text=f"{side}{row['target']}{side} What is the middle letter?"),
                    options=self.letters,
                    answer=row["target"],
                    condition={
                        "congruency": "congruent" if row["flanker"] == row["target"] else "incongruent",
                        "target": row["target"],
                        "flanker": row["flanker"]
                    },
                    item=f"{row['flanker']}-{row['target']}"
                )
            )
        return trials


class Simon(Paradigm):
    """Press the key a colour is mapped to, wherever the shape appears.

    Maps onto the Simon task: the side a stimulus appears on is irrelevant, but responses
    should be harder when it differs from the side of the correct key. What it can't
    capture: the model reads the word "left" or "right" rather than seeing a location, so
    the conflict is between words, and surprisal stands in for response time. Each item
    states its own colour-to-key mapping, and half the items use each mapping. The
    built-in items are generated, not human-normed.

    Args:
        colors: The two colours.
        shapes: The shapes, which only vary the wording.
        n_repeats: How many times the set of items repeats.
        **settings: What every paradigm takes, such as `stimuli` with the columns
            `color`, `side`, `shape` and `left_color`, the colour mapped to the left key.

    Raises:
        ValueError: If there aren't exactly two colours, or a row names a colour or side
            that doesn't exist.
    """

    stimuli_label = BUILT_IN_STIMULI
    item_columns = ("color", "side", "shape", "left_color")
    group_by = ("correspondence",)
    effects = (
        Effect(
            "simon_effect",
            "answer_surprisal",
            by="correspondence",
            levels=("non_corresponding", "corresponding")
        ),
        Effect(
            "errors",
            "answer_margin",
            by="correspondence",
            levels=("corresponding", "non_corresponding"),
            proxy_for="error rate"
        )
    )

    def __init__(
        self,
        *,
        colors: Sequence[str] = ("red", "blue"),
        shapes: Sequence[str] = ("square", "circle", "triangle", "star"),
        n_repeats: int = 1,
        **settings: Any
    ) -> None:
        colors = _check_choices(colors, "colors")
        if len(colors) != 2:
            raise ValueError(f"The Simon task maps two colours onto two keys, got {list(colors)}.")
        super().__init__(
            colors=colors,
            shapes=tuple(shapes),
            n_repeats=_check_repeats(n_repeats),
            **settings
        )
        _check_values(self.rows, "color", self.colors)
        _check_values(self.rows, "left_color", self.colors)
        _check_values(self.rows, "side", ("left", "right"))

    def default_items(self, rng: np.random.Generator) -> list[dict[str, Any]]:
        """Crosses every colour, side, shape and mapping.

        Args:
            rng: Unused, since the items are fixed.

        Returns:
            Rows with `color`, `side`, `shape` and `left_color`.
        """
        combinations = product(
            self.colors,
            ("left", "right"),
            self.shapes,
            self.colors
        )
        rows = [
            {
                "color": color,
                "side": side,
                "shape": shape,
                "left_color": left
            }
            for color, side, shape, left in combinations
        ]
        return rows * self.n_repeats

    def trials(self, rng: np.random.Generator) -> list[Trial]:
        """Writes one trial per row.

        Args:
            rng: Passed to `item_rows`.

        Returns:
            The trials, with the conditions `correspondence`, `color`, `side` and
            `left_color`.
        """
        trials = []
        for row in self.item_rows(rng):
            left = row["left_color"]
            right = next(color for color in self.colors if color != left)
            key = "left" if row["color"] == left else "right"
            text = (
                f"Press left for {left} and right for {right}. A {row['color']} {row['shape']} "
                f"appears on the {row['side']}. Which key?"
            )
            trials.append(
                Trial(
                    Stimulus(text=text),
                    options=("left", "right"),
                    answer=key,
                    condition={
                        "correspondence": "corresponding" if key == row["side"] else "non_corresponding",
                        "color": row["color"],
                        "side": row["side"],
                        "left_color": left
                    },
                    item=f"{row['color']}-{row['side']}-{row['shape']}-{left}"
                )
            )
        return trials


class GoNoGo(Paradigm):
    """Press for every letter except one.

    Maps onto the go/no-go task: frequent go trials build a habit of pressing, which the
    rare no-go letter must override. What it can't capture: the habit can only come from
    the subject's own recent responses in its context, and there's no time pressure, so
    surprisal stands in for the difficulty of withholding. Sequences are generated, not
    human-normed. d′ comes from `results.metrics("d_prime", positive="press")`.

    Args:
        go_ratio: The share of go trials.
        no_go_letter: The letter that means withhold.
        n_trials: How many trials each participant sees.
        letters: The go letters.
        **settings: What every paradigm takes.

    Raises:
        ValueError: If `go_ratio` isn't between 0 and 1, or either kind of trial would
            appear fewer than two times.
    """

    stimuli_label = BUILT_IN_STIMULI
    context = context.window(8)
    ordered = True
    group_by = ()
    effects = (
        Effect(
            "inhibition_cost",
            "answer_surprisal",
            by="trial_type",
            levels=("no_go", "go")
        ),
        Effect(
            "errors",
            "answer_margin",
            by="trial_type",
            levels=("go", "no_go"),
            proxy_for="error rate"
        )
    )

    def __init__(
        self,
        *,
        go_ratio: float = 0.75,
        no_go_letter: str = "X",
        n_trials: int = 40,
        letters: Sequence[str] = _GO_LETTERS,
        **settings: Any
    ) -> None:
        if not 0 < go_ratio < 1:
            raise ValueError(f"go_ratio must be between 0 and 1, got {go_ratio}.")
        n_go = round(n_trials * go_ratio)
        if min(n_go, n_trials - n_go) < 2:
            raise ValueError(f"{n_trials} trials at go_ratio={go_ratio} leave fewer than two of one kind.")
        if no_go_letter in letters:
            raise ValueError(f"The no-go letter '{no_go_letter}' can't also be a go letter.")
        super().__init__(
            go_ratio=go_ratio,
            no_go_letter=no_go_letter,
            n_trials=n_trials,
            letters=tuple(letters),
            **settings
        )

    def trials(self, rng: np.random.Generator) -> list[Trial]:
        """Draws a sequence with the asked share of go trials, in random order.

        Args:
            rng: Seeded for this participant.

        Returns:
            The trials, with the condition `trial_type`.
        """
        n_go = round(self.n_trials * self.go_ratio)
        kinds = rng.permutation(["go"] * n_go + ["no_go"] * (self.n_trials - n_go)).tolist()
        trials = []
        for kind in kinds:
            letter = self.no_go_letter if kind == "no_go" else str(rng.choice(self.letters))
            trials.append(
                Trial(
                    Stimulus(text=f"Press unless the letter is {self.no_go_letter}. Letter: {letter}"),
                    options=(_PRESS, _WITHHOLD),
                    answer=_WITHHOLD if kind == "no_go" else _PRESS,
                    condition={"trial_type": kind}
                )
            )
        return trials


class StopSignal(Paradigm):
    """Press for each letter unless a stop cue follows it.

    Maps onto the stop-signal task: a cue that arrives later should be harder to obey, so
    the chance of pressing despite it should rise with its delay. What it can't capture:
    the delay is a count of tokens before the cue, with no time in it, and the model reads
    the whole trial at once, so no stop-signal reaction time is reported, only how pressing
    changes with delay. Go trials show the same filler with no cue. Sequences are
    generated, not human-normed.

    Args:
        delays: How many filler tokens come before the stop cue.
        stop_ratio: The share of stop trials, split evenly across delays.
        n_trials: How many trials each participant sees.
        letters: The letters shown.
        **settings: What every paradigm takes.

    Raises:
        ValueError: If there are fewer than two delays, or fewer than two stop trials per
            delay.
    """

    stimuli_label = BUILT_IN_STIMULI
    context = context.window(8)
    ordered = True
    group_by = ("trial_type",)

    def __init__(
        self,
        *,
        delays: Sequence[int] = (1, 5),
        stop_ratio: float = 0.25,
        n_trials: int = 40,
        letters: Sequence[str] = _GO_LETTERS,
        **settings: Any
    ) -> None:
        delays = tuple(sorted(set(delays)))
        if len(delays) < 2 or delays[0] < 0:
            raise ValueError(f"Give at least two different delays of 0 or more, got {list(delays)}.")
        n_stop = round(n_trials * stop_ratio)
        if n_stop < 2 * len(delays) or n_trials - n_stop < 2:
            raise ValueError(
                f"{n_trials} trials at stop_ratio={stop_ratio} give {n_stop} stop trials, and each of "
                f"the {len(delays)} delays needs two."
            )
        super().__init__(
            delays=delays,
            stop_ratio=stop_ratio,
            n_trials=n_trials,
            letters=tuple(letters),
            **settings
        )

    @property
    def effects(self) -> tuple[Effect, ...]:
        """Pressing despite a late cue against an early one, then stopping against going."""
        return (
            Effect(
                "late_signal",
                "option_probability",
                by="delay",
                levels=(self.delays[-1], self.delays[0]),
                where={"trial_type": "stop"},
                options={"option": _PRESS}
            ),
            Effect(
                "stopping",
                "answer_margin",
                by="trial_type",
                levels=("go", "stop"),
                proxy_for="error rate"
            )
        )

    def trials(self, rng: np.random.Generator) -> list[Trial]:
        """Draws a sequence with the asked share of stop trials, in random order.

        Args:
            rng: Seeded for this participant.

        Returns:
            The trials, with the conditions `trial_type` and `delay`, which is None on
            go trials.
        """
        n_stop = round(self.n_trials * self.stop_ratio)
        stops = [self.delays[position % len(self.delays)] for position in range(n_stop)]
        plan = [("stop", delay) for delay in stops] + [("go", None)] * (self.n_trials - n_stop)
        trials = []
        for position in rng.permutation(len(plan)).tolist():
            kind, delay = plan[position]
            dots = int(rng.choice(self.delays)) if delay is None else delay
            cue = " STOP" if kind == "stop" else ""
            letter = str(rng.choice(self.letters))
            trials.append(
                Trial(
                    Stimulus(text=f"Press unless you see STOP. Letter: {letter}{' .' * dots}{cue}"),
                    options=(_PRESS, _WITHHOLD),
                    answer=_WITHHOLD if kind == "stop" else _PRESS,
                    condition={"trial_type": kind, "delay": delay}
                )
            )
        return trials


class TaskSwitching(Paradigm):
    """Judge either the letter or the number of a pair, as each trial's cue says.

    Maps onto task switching with explicit cues: answering should be harder right after
    the task changes than when it repeats. What it can't capture: there's no time to
    prepare between cue and stimulus, and the previous tasks are only in context as text,
    so surprisal stands in for the switch cost. The first trial has no previous task and
    is labelled `start`, which leaves it out of the effect. Sequences are generated, not
    human-normed.

    Args:
        n_trials: How many trials each participant sees. Half of the transitions
            switch task.
        **settings: What every paradigm takes.

    Raises:
        ValueError: If `n_trials` leaves fewer than two switches or two repeats.
    """

    stimuli_label = BUILT_IN_STIMULI
    context = context.window(4)
    ordered = True
    group_by = ("transition",)
    effects = (
        Effect(
            "switch_cost",
            "answer_surprisal",
            by="transition",
            levels=("switch", "repeat")
        ),
        Effect(
            "errors",
            "answer_margin",
            by="transition",
            levels=("repeat", "switch"),
            proxy_for="error rate"
        )
    )

    def __init__(self, *, n_trials: int = 40, **settings: Any) -> None:
        if (n_trials - 1) // 2 < 2:
            raise ValueError(f"n_trials={n_trials} leaves fewer than two switches. Use at least 6.")
        super().__init__(n_trials=n_trials, **settings)

    def trials(self, rng: np.random.Generator) -> list[Trial]:
        """Draws a sequence in which half of the transitions switch task.

        Args:
            rng: Seeded for this participant.

        Returns:
            The trials, with the conditions `transition` and `task`.
        """
        changes = self.n_trials - 1
        switches = rng.permutation([True] * (changes // 2) + [False] * (changes - changes // 2)).tolist()
        task = str(rng.choice(["letter", "number"]))
        trials = [self._trial(task, "start", rng)]
        for switch in switches:
            if switch:
                task = "number" if task == "letter" else "letter"
            trials.append(self._trial(task, "switch" if switch else "repeat", rng))
        return trials

    def _trial(
        self,
        task: str,
        transition: str,
        rng: np.random.Generator
    ) -> Trial:
        letter = str(rng.choice(_VOWELS + _CONSONANTS))
        number = int(rng.integers(2, 10))
        if task == "letter":
            question, yes = "Is the letter a vowel?", letter in _VOWELS
        else:
            question, yes = "Is the number even?", number % 2 == 0
        return Trial(
            Stimulus(text=f"Task: {task}. {letter}{number} {question}"),
            options=("yes", "no"),
            answer="yes" if yes else "no",
            condition={"transition": transition, "task": task}
        )


def _check_choices(values: Sequence[str], name: str) -> tuple[str, ...]:
    values = tuple(values)
    if len(set(values)) < 2 or len(set(values)) != len(values):
        raise ValueError(f"Give at least two {name}, all different, got {list(values)}.")
    return values


def _check_repeats(n_repeats: int) -> int:
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be at least 1, got {n_repeats}.")
    return n_repeats


def _check_values(rows: Sequence[dict[str, Any]] | None, column: str, allowed: Sequence[str]) -> None:
    for position, row in enumerate(rows or ()):
        if row[column] not in allowed:
            raise ValueError(f"Row {position} has '{row[column]}' in '{column}', which isn't one of {list(allowed)}.")


def _balanced(
    values: Sequence[str],
    n_repeats: int,
    distractor: str,
    target: str
) -> list[dict[str, Any]]:
    # a matching pair repeats once per mismatching partner, so both conditions are the same size
    rows = []
    for first, second in product(values, values):
        copies = len(values) - 1 if first == second else 1
        rows.extend([{distractor: first, target: second}] * (copies * n_repeats))
    return rows


def _congruency(word: str, ink: str, colors: Sequence[str]) -> str:
    if word == ink:
        return "congruent"
    return "incongruent" if word in colors else "neutral"
