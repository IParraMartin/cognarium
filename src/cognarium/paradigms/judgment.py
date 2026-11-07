"""Paradigms about judgment and reasoning: anchoring, framing, conjunctions and beliefs."""

from __future__ import annotations

from typing import Any

import numpy as np

from cognarium.paradigms.base import (
    BUILT_IN_STIMULI,
    Effect,
    Paradigm,
    load_items
)
from cognarium.responses import PromptTemplate
from cognarium.stimuli import Stimulus, Trial

_ANCHORS = ("high", "low")
_FRAMES = ("gain", "loss")
_DESCRIPTIONS = ("representative", "neutral")
_BELIEFS = ("false", "true")
_SURE = "sure"
_GAMBLE = "gamble"
_ANSWER = PromptTemplate(stimulus="{text}", response="Answer:")


class Anchoring(Paradigm):
    """Compare a quantity with a number, then estimate it.

    Each question is asked once after a low anchor and once after a high one, as in "Is
    the number of keys on a standard piano more or less than 220?", and the estimate is
    a choice among the same five ordered ranges either way.

    Maps onto the anchoring effect (Tversky and Kahneman, 1974): estimates move toward a
    number that was considered first, even an arbitrary one. The measure is the expected
    position among the ranges, which should be higher after the high anchor. What it
    can't capture: ranges replace free estimates, so a shift within a range is lost, and
    the anchor is in the same prompt as the estimate rather than a separate question
    answered first. The built-in questions were written for cognarium and are not
    human-normed.

    Args:
        **settings: What every paradigm takes, such as `stimuli` with the columns
            `question`, a noun phrase, `low_anchor`, `high_anchor` and `bins`, the
            ordered ranges as a list or as text separated by `|`.

    Raises:
        ValueError: If a row has fewer than two different ranges.
    """

    stimuli_label = BUILT_IN_STIMULI
    template = PromptTemplate(stimulus="{text}", response="")
    item_columns = ("question", "low_anchor", "high_anchor", "bins")
    group_by = ("anchor",)
    effects = (
        Effect(
            "anchoring",
            "expected_option",
            by="anchor",
            levels=_ANCHORS
        ),
    )

    def default_items(self, rng: np.random.Generator) -> list[dict[str, Any]]:
        """Gives the built-in estimation questions.

        Args:
            rng: Unused, since the items are fixed.

        Returns:
            Rows with `question`, both anchors and five `bins`.
        """
        return load_items("anchoring")

    def trials(self, rng: np.random.Generator) -> list[Trial]:
        """Asks each question after each anchor, with the same ranges.

        Args:
            rng: Seeded for this participant, for the rows.

        Returns:
            Two trials per row, with the conditions `anchor` and `anchor_value`. The item
            is the question, so its two trials pair up.

        Raises:
            ValueError: If a row has fewer than two different ranges.
        """
        trials = []
        for position, row in enumerate(self.item_rows(rng)):
            question = str(row["question"])
            bins = _bins(row["bins"], position)
            for anchor in _ANCHORS:
                value = float(row[f"{anchor}_anchor"])
                written = str(int(value)) if value.is_integer() else str(value)
                trials.append(
                    Trial(
                        Stimulus(text=f"Is {question} more or less than {written}? Your estimate of {question}:"),
                        options=bins,
                        item=question,
                        condition={"anchor": anchor, "anchor_value": value}
                    )
                )
        return trials


class Framing(Paradigm):
    """Choose between a sure outcome and a gamble, described as gains or as losses.

    Each scenario is told twice with the same outcomes: once as what will be saved and
    once as what will be lost. In both, the sure option equals the gamble's expected
    outcome, and the text says to answer `sure` or `gamble`.

    Maps onto risky-choice framing (Tversky and Kahneman, 1981): people prefer the sure
    option when outcomes are described as gains and the gamble when they are described
    as losses. What it can't capture: nothing is at stake, and both frames of a scenario
    are judged independently rather than by different people. The built-in scenarios
    were written for cognarium and are not human-normed.

    Args:
        **settings: What every paradigm takes, such as `stimuli` with the columns
            `gain_text` and `loss_text`, each of which should say what `sure` and `gamble`
            choose.
    """

    stimuli_label = BUILT_IN_STIMULI
    template = _ANSWER
    item_columns = ("gain_text", "loss_text")
    group_by = ("frame",)
    effects = (
        Effect(
            "framing",
            "option_probability",
            by="frame",
            levels=_FRAMES,
            options={"option": _SURE}
        ),
    )

    def default_items(self, rng: np.random.Generator) -> list[dict[str, Any]]:
        """Gives the built-in scenarios.

        Args:
            rng: Unused, since the items are fixed.

        Returns:
            Rows with both texts, and the numbers behind them.
        """
        return load_items("framing")

    def trials(self, rng: np.random.Generator) -> list[Trial]:
        """Tells each scenario in each frame.

        Args:
            rng: Seeded for this participant, for the rows.

        Returns:
            Two trials per row, with the condition `frame`. The item is the row's
            position, so the two frames pair up.
        """
        return [
            Trial(
                Stimulus(text=str(row[f"{frame}_text"])),
                options=(_SURE, _GAMBLE),
                item=position,
                condition={"frame": frame}
            )
            for position, row in enumerate(self.item_rows(rng))
            for frame in _FRAMES
        ]


class ConjunctionFallacy(Paradigm):
    """Judge whether a single property or that property joined with another is likelier.

    Each vignette is told with a description that fits the added property and with a
    neutral one. The two statements are numbered 1 and 2, the single property takes each
    position equally often, and the answer is always the single property, since a
    conjunction can't be more probable than one of its parts.

    Maps onto the conjunction fallacy (Tversky and Kahneman, 1983): a description that
    fits one property makes people rate that property joined with another as more likely
    than the single property. The primary effect is the drop in choosing the single
    property after a fitting description, and `error_rate` compares that choice with
    chance. What it can't capture: a choice between two statements replaces ranking
    several, and a subject may read "more likely" as "more fitting". The built-in
    vignettes were written for cognarium and are not human-normed.

    Args:
        **settings: What every paradigm takes, such as `stimuli` with the columns
            `representative`, `neutral`, `single` and `conjunction`.
    """

    stimuli_label = BUILT_IN_STIMULI
    template = _ANSWER
    item_columns = ("representative", "neutral", "single", "conjunction")
    group_by = ("description",)
    effects = (
        Effect(
            "fallacy",
            "answer_probability",
            by="description",
            levels=("neutral", "representative")
        ),
        Effect(
            "error_rate",
            "answer_probability",
            against=0.5,
            where={"description": "representative"}
        )
    )

    def default_items(self, rng: np.random.Generator) -> list[dict[str, Any]]:
        """Gives the built-in vignettes.

        Args:
            rng: Unused, since the items are fixed.

        Returns:
            Rows with both descriptions and both statements.
        """
        return load_items("conjunction")

    def trials(self, rng: np.random.Generator) -> list[Trial]:
        """Asks about each vignette after each description.

        Args:
            rng: Seeded for this participant, for the rows.

        Returns:
            Two trials per row, with the condition `description`. The single property
            comes first on alternate rows, in opposite places for the two descriptions.
            The item is the row's position.
        """
        trials = []
        for position, row in enumerate(self.item_rows(rng)):
            single, conjunction = str(row["single"]), str(row["conjunction"])
            for offset, description in enumerate(_DESCRIPTIONS):
                single_first = (position + offset) % 2 == 0
                first, second = (single, conjunction) if single_first else (conjunction, single)
                trials.append(
                    Trial(
                        Stimulus(text=f"{row[description]}\nWhich is more likely?\n1: {first}\n2: {second}"),
                        options=("1", "2"),
                        answer="1" if single_first else "2",
                        item=position,
                        condition={"description": description}
                    )
                )
        return trials


class FalseBelief(Paradigm):
    """Say where someone will look for an object that was moved, seen or unseen.

    Each story is told twice from one template: once where the person is away when the
    object is moved, so they believe it is still in its first place, and once where they
    watch it move. The options are the two places, and each place holds the answer
    equally often.

    Maps onto the change-of-location false-belief task (Wimmer and Perner, 1983): someone
    who didn't see an object moved should look where they left it. The true-belief
    control shows a subject that always names the object's real place, or always its
    first place, for what it is. What it can't capture: passing doesn't show theory of
    mind, because text statistics can produce the same answers, and the story is read
    rather than watched. The built-in stories were written for cognarium and are not
    human-normed.

    Args:
        **settings: What every paradigm takes, such as `stimuli` with the columns
            `name`, `object`, `first_place` and `second_place`.
    """

    stimuli_label = BUILT_IN_STIMULI
    template = _ANSWER
    item_columns = ("name", "object", "first_place", "second_place")
    group_by = ("belief",)
    effects = (
        Effect(
            "belief_cost",
            "answer_surprisal",
            by="belief",
            levels=_BELIEFS
        ),
        Effect(
            "above_chance",
            "answer_probability",
            against=0.5,
            where={"belief": "false"}
        )
    )

    def default_items(self, rng: np.random.Generator) -> list[dict[str, Any]]:
        """Gives the built-in characters, objects and places.

        Args:
            rng: Unused, since the items are fixed.

        Returns:
            Rows with `name`, `object`, `first_place` and `second_place`.
        """
        return load_items("false_belief")

    def trials(self, rng: np.random.Generator) -> list[Trial]:
        """Tells each story with a false and a true belief.

        Args:
            rng: Seeded for this participant, for the rows.

        Returns:
            Two trials per row, with the condition `belief`. The answer comes first on
            alternate rows, in opposite places for the two beliefs. The item is the row's
            position.
        """
        trials = []
        for position, row in enumerate(self.item_rows(rng)):
            name, thing = str(row["name"]), str(row["object"])
            first, second = str(row["first_place"]), str(row["second_place"])
            for offset, belief in enumerate(_BELIEFS):
                if belief == "false":
                    events = (
                        f"{name} puts the {thing} in the {first}, then leaves the room. While {name} is away, "
                        f"someone moves the {thing} to the {second}. {name} comes back."
                    )
                else:
                    events = (
                        f"{name} puts the {thing} in the {first}. While {name} watches, someone moves the "
                        f"{thing} to the {second}. {name} leaves the room and comes back."
                    )
                answer, other = (first, second) if belief == "false" else (second, first)
                trials.append(
                    Trial(
                        Stimulus(text=f"{events} Where will {name} look for the {thing} first?"),
                        options=(answer, other) if (position + offset) % 2 == 0 else (other, answer),
                        answer=answer,
                        item=position,
                        condition={"belief": belief}
                    )
                )
        return trials


def _bins(value: Any, position: int) -> tuple[str, ...]:
    bins = value.split("|") if isinstance(value, str) else list(value)
    bins = tuple(str(label).strip() for label in bins)
    if len(set(bins)) < 2 or len(set(bins)) < len(bins):
        raise ValueError(f"Row {position} needs at least two different bins, in order, got {list(bins)}.")
    return bins

