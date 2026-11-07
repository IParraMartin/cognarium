"""Paradigms about language: priming, word recognition, sentence processing and fluency."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from cognarium.paradigms.base import (
    BUILT_IN_STIMULI,
    Effect,
    Paradigm,
    load_items
)
from cognarium.responses import PromptTemplate
from cognarium.stimuli import Stimulus, Trial

_WORD = "word"
_NONWORD = "nonword"
_RELATEDNESS = ("related", "unrelated", "filler")
_STRUCTURES = ("po", "do")
_AMBIGUITY = ("ambiguous", "unambiguous")
_SEPARATORS = re.compile(r"[,;\n]+")
_PUNCTUATION = re.compile(r"[^\w\s'-]")
# a word is letters only, which may be joined by an apostrophe or a hyphen
_SPELLED = re.compile(r"[^\W\d_]+(?:['-][^\W\d_]+)*")
# the response continues the stimulus, so an option is scored as its next words
_CONTINUATION = PromptTemplate(stimulus="{text}", response="")


class SemanticPriming(Paradigm):
    """Decide whether a target is a word, after a related or an unrelated prime.

    Maps onto semantic priming in lexical decision (Meyer and Schvaneveldt, 1971): a
    word is recognised faster after a related word, as `nurse` after `doctor`. What it
    can't capture: there is no timing, so the surprisal of the correct answer stands in
    for decision time and is labelled a proxy, and the prime is read in the same prompt
    rather than flashed before the target. Each target appears once with a related and
    once with an unrelated prime, so the comparison is within items, and nonword targets
    are fillers so that "word" isn't always right. The built-in pairs were written for
    cognarium and are not human-normed.

    Args:
        **settings: What every paradigm takes, such as `stimuli` with the columns
            `prime`, `target` and `relatedness`, which is `related`, `unrelated` or
            `filler`. Filler targets are nonwords.

    Raises:
        ValueError: If a row's relatedness is none of those three.
    """

    stimuli_label = BUILT_IN_STIMULI
    item_columns = ("prime", "target", "relatedness")
    group_by = ("relatedness",)
    effects = (
        Effect(
            "priming",
            "answer_surprisal",
            by="relatedness",
            levels=("unrelated", "related")
        ),
    )

    def default_items(self, rng: np.random.Generator) -> list[dict[str, Any]]:
        """Gives the built-in prime and target pairs, with nonword fillers.

        Args:
            rng: Unused, since the items are fixed.

        Returns:
            Rows with `prime`, `target` and `relatedness`.
        """
        return load_items("semantic_priming")

    def trials(self, rng: np.random.Generator) -> list[Trial]:
        """Asks whether each target is a word, after its prime.

        Args:
            rng: Seeded for this participant, for the rows.

        Returns:
            One trial per row, with the conditions `relatedness` and `prime`. The item is
            the target, so its related and unrelated trials pair up.

        Raises:
            ValueError: If a row's relatedness is unknown.
        """
        trials = []
        for position, row in enumerate(self.item_rows(rng)):
            relatedness = _one_of(
                row["relatedness"],
                _RELATEDNESS,
                "relatedness",
                position
            )
            prime, target = str(row["prime"]), str(row["target"])
            trials.append(
                Trial(
                    Stimulus(text=f"Prime: {prime}. Target: {target}. Is the target a word?"),
                    options=(_WORD, _NONWORD),
                    answer=_NONWORD if relatedness == "filler" else _WORD,
                    item=target,
                    condition={"relatedness": relatedness, "prime": prime}
                )
            )
        return trials


class StructuralPriming(Paradigm):
    """Read a dative sentence, then choose between two forms of another.

    Maps onto structural priming (Bock, 1986): after a prepositional-object prime, as in
    "gave a book to the girl", people more often describe a new event with a
    prepositional object too, and after a double-object prime, as in "gave the girl a
    book", with a double object. What it can't capture: the subject chooses between two
    given sentences rather than producing one, and prime and target share their verb,
    which in people strengthens priming, so the effect isn't purely structural. The
    prepositional-object form is always the first option. The built-in sentences were
    written for cognarium and are not human-normed.

    Args:
        **settings: What every paradigm takes, such as `stimuli` with the columns
            `prime_po`, `prime_do`, `target_po`, `target_do` and `verb`.
    """

    stimuli_label = BUILT_IN_STIMULI
    template = PromptTemplate(stimulus="Sentence: {text}", response="Sentence:")
    item_columns = ("prime_po", "prime_do", "target_po", "target_do", "verb")
    group_by = ("prime_structure",)
    effects = (
        Effect(
            "priming",
            "option_probability",
            by="prime_structure",
            levels=_STRUCTURES,
            options={"option": 0}
        ),
    )

    def default_items(self, rng: np.random.Generator) -> list[dict[str, Any]]:
        """Gives the built-in dative sentences.

        Args:
            rng: Unused, since the items are fixed.

        Returns:
            Rows with both forms of a prime and a target, and their shared verb.
        """
        return load_items("structural_priming")

    def trials(self, rng: np.random.Generator) -> list[Trial]:
        """Shows each target after a prime of each form.

        Args:
            rng: Seeded for this participant, for the rows.

        Returns:
            Two trials per row, with the conditions `prime_structure` and `verb`. The item
            is the target, so its two trials pair up.
        """
        trials = []
        for row in self.item_rows(rng):
            options = (str(row["target_po"]), str(row["target_do"]))
            for structure in _STRUCTURES:
                trials.append(
                    Trial(
                        Stimulus(text=str(row[f"prime_{structure}"])),
                        options=options,
                        item=options[0],
                        condition={"prime_structure": structure, "verb": str(row["verb"])}
                    )
                )
        return trials


class LexicalDecision(Paradigm):
    """Decide whether a string of letters is a word.

    Maps onto visual lexical decision: people accept frequent words faster than rare
    ones, the word frequency effect, and reject pronounceable nonwords. What it can't
    capture: there is no timing, so the surprisal of the correct answer stands in for
    decision time and is labelled a proxy. Discrimination is available as
    `results.metrics("d_prime", positive="word")`. The built-in words and nonwords were
    written for cognarium, and their frequency labels are our own rough estimate rather
    than corpus counts, so they are not human-normed.

    Args:
        **settings: What every paradigm takes, such as `stimuli` with the columns
            `letters` and `lexicality`, which is `word` or `nonword`, and optionally
            `frequency`, such as `high` or `low`, which nonwords leave empty.

    Raises:
        ValueError: If a row's lexicality is neither `word` nor `nonword`.
    """

    stimuli_label = BUILT_IN_STIMULI
    item_columns = ("letters", "lexicality")
    group_by = ()
    effects = (
        Effect(
            "frequency_effect",
            "answer_surprisal",
            by="frequency",
            levels=("low", "high"),
            where={"lexicality": _WORD}
        ),
    )

    def default_items(self, rng: np.random.Generator) -> list[dict[str, Any]]:
        """Gives the built-in words, of high and low frequency, and nonwords.

        Args:
            rng: Unused, since the items are fixed.

        Returns:
            Rows with `letters`, `lexicality` and `frequency`.
        """
        return load_items("lexical_decision")

    def trials(self, rng: np.random.Generator) -> list[Trial]:
        """Asks whether each string is a word.

        Args:
            rng: Seeded for this participant, for the rows.

        Returns:
            One trial per row, with the conditions `lexicality` and `frequency`, which
            is None where a row gives none.

        Raises:
            ValueError: If a row's lexicality is unknown.
        """
        trials = []
        for position, row in enumerate(self.item_rows(rng)):
            lexicality = _one_of(
                row["lexicality"],
                (_WORD, _NONWORD),
                "lexicality",
                position
            )
            frequency = row.get("frequency")
            letters = str(row["letters"])
            trials.append(
                Trial(
                    Stimulus(text=f"Is this a word? {letters}"),
                    options=(_WORD, _NONWORD),
                    answer=lexicality,
                    item=letters,
                    condition={
                        "lexicality": lexicality,
                        "frequency": None if _missing(frequency) else str(frequency)
                    }
                )
            )
        return trials


class GardenPath(Paradigm):
    """Score the word that resolves a temporarily ambiguous sentence.

    Each item is a sentence beginning in an ambiguous form, as in "The boat sailed down
    the river", and an unambiguous one, "The boat that was sailed down the river". The
    options are the disambiguating word, "sank", and a foil that continues the misleading
    reading. Both are scored as the next word, straight after the beginning.

    Maps onto garden-path effects in reading: the disambiguating word is read slowly
    after an ambiguous beginning, and less so after an unambiguous one. Its surprisal
    stands in for the slowdown, as in Futrell et al. (2019). What it can't capture: there
    is no reading time, and surprisal is renormalised over the disambiguator and one foil,
    so it is relative to that foil. The built-in sentences were written for cognarium and
    are not human-normed.

    Args:
        **settings: What every paradigm takes, such as `stimuli` with the columns
            `ambiguous`, `unambiguous`, `disambiguator` and `foil`.
    """

    stimuli_label = BUILT_IN_STIMULI
    template = _CONTINUATION
    item_columns = ("ambiguous", "unambiguous", "disambiguator", "foil")
    group_by = ("ambiguity",)
    effects = (
        Effect(
            "garden_path",
            "answer_surprisal",
            by="ambiguity",
            levels=_AMBIGUITY
        ),
    )

    def default_items(self, rng: np.random.Generator) -> list[dict[str, Any]]:
        """Gives the built-in sentence beginnings.

        Args:
            rng: Unused, since the items are fixed.

        Returns:
            Rows with both forms of a beginning, the disambiguating word and the foil.
        """
        return load_items("garden_path")

    def trials(self, rng: np.random.Generator) -> list[Trial]:
        """Scores the disambiguating word after each form of each beginning.

        Args:
            rng: Seeded for this participant, for the rows.

        Returns:
            Two trials per row, with the conditions `ambiguity` and `foil`. The item is
            the ambiguous beginning, so the two forms pair up.
        """
        trials = []
        for row in self.item_rows(rng):
            answer, foil = str(row["disambiguator"]), str(row["foil"])
            for ambiguity in _AMBIGUITY:
                trials.append(
                    Trial(
                        Stimulus(text=str(row[ambiguity])),
                        options=(answer, foil),
                        answer=answer,
                        item=str(row["ambiguous"]),
                        condition={"ambiguity": ambiguity, "foil": foil}
                    )
                )
        return trials


class MinimalPairs(Paradigm):
    """Prefer the acceptable sentence of a pair that differs in one place.

    Maps onto acceptability judgments of minimal pairs, as used for language models in
    BLiMP (Warstadt et al., 2020): the acceptable sentence should be the more probable
    one. What it can't capture: this is a preference between two sentences, not a rating
    of one, and summed log probability favours shorter sentences, so the two sentences of
    a pair should be the same length. Each pair is shown in its own random order, and
    results are reported by phenomenon. The built-in pairs were written for cognarium,
    are the same length in words, and are not human-normed.

    Args:
        **settings: What every paradigm takes, such as `stimuli` with the columns `good`,
            `bad` and `phenomenon`.
    """

    stimuli_label = BUILT_IN_STIMULI
    template = _CONTINUATION
    item_columns = ("good", "bad", "phenomenon")
    group_by = ("phenomenon",)
    effects = (
        Effect(
            "preference",
            "answer_probability",
            against=0.5
        ),
    )

    def default_items(self, rng: np.random.Generator) -> list[dict[str, Any]]:
        """Gives the built-in sentence pairs.

        Args:
            rng: Unused, since the items are fixed.

        Returns:
            Rows with `good`, `bad` and `phenomenon`.
        """
        return load_items("minimal_pairs")

    def trials(self, rng: np.random.Generator) -> list[Trial]:
        """Offers both sentences of each pair, in a random order.

        Args:
            rng: Seeded for this participant, for the order of each pair.

        Returns:
            One trial per pair, with the condition `phenomenon`. The answer is the
            acceptable sentence.
        """
        trials = []
        for row in self.item_rows(rng):
            good, bad = str(row["good"]), str(row["bad"])
            trials.append(
                Trial(
                    Stimulus(text="Sentence:"),
                    options=(good, bad) if rng.random() < 0.5 else (bad, good),
                    answer=good,
                    item=good,
                    condition={"phenomenon": str(row["phenomenon"])}
                )
            )
        return trials


class Fluency(Paradigm):
    """Name as many words as you can from a category, or starting with a letter.

    Maps onto verbal fluency, where people usually produce more words for a semantic
    category than for a letter in the same time. What it can't capture: there is no time
    limit, so the token budget stands in for one, and words given for a category aren't
    checked for membership, since cognarium holds no category norms. A response is read
    as a list separated by commas, semicolons or new lines, or by spaces when it has none
    of those. Only items of one or two words made of letters count, so numbers, code and
    sentences are left out. The categories and letters are common choices, not
    human-normed.

    Args:
        semantic: The categories, each named in the plural.
        letters: The letters words must start with.
        **settings: What every paradigm takes. `max_new_tokens` defaults to 64.

    Raises:
        ValueError: If there are no categories or no letters.
    """

    stimuli_label = BUILT_IN_STIMULI
    response_mode = "generate"
    group_by = ("kind",)
    effects = (
        Effect(
            "fluency_gap",
            "n_produced",
            by="kind",
            levels=("semantic", "letter")
        ),
    )

    def __init__(
        self,
        *,
        semantic: Sequence[str] = ("animals", "fruits", "tools"),
        letters: Sequence[str] = ("f", "a", "s"),
        **settings: Any
    ) -> None:
        if not semantic or not letters:
            raise ValueError(
                "Fluency compares the two kinds, so it needs at least one semantic category "
                "and one letter."
            )
        settings.setdefault("max_new_tokens", 64)
        super().__init__(semantic=tuple(semantic), letters=tuple(letters), **settings)

    def trials(self, rng: np.random.Generator) -> list[Trial]:
        """Asks for words of each category and each letter.

        Args:
            rng: Unused, since the cues are fixed.

        Returns:
            One trial per cue, with the conditions `kind` and `cue`.
        """
        cues = [
            *((category, "semantic", f"List as many {category} as you can") for category in self.semantic),
            *((letter, "letter", f"List as many words starting with {letter} as you can") for letter in self.letters)
        ]
        return [
            Trial(
                Stimulus(text=f"{request}, separated by commas:"),
                item=cue,
                condition={"kind": kind, "cue": cue}
            )
            for cue, kind, request in cues
        ]

    def annotate(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Reads the words out of each response.

        Args:
            frame: The results table of one step.

        Returns:
            The table with `produced`, the different words in the order first written,
            `n_produced`, how many there are, `n_valid`, those that start with the letter
            on letter trials and all of them on semantic trials, and `n_repeats`, how many
            words were written again.
        """
        parsed = [_words(str(text)) for text in frame["response"]]
        produced = [list(dict.fromkeys(words)) for words in parsed]
        valid = [
            len(unique) if kind == "semantic" else sum(word.startswith(str(cue).lower()) for word in unique)
            for unique, kind, cue in zip(
                produced,
                frame["kind"],
                frame["cue"],
                strict=True
            )
        ]
        return frame.assign(
            produced=produced,
            n_produced=[len(unique) for unique in produced],
            n_valid=valid,
            n_repeats=[len(words) - len(unique) for words, unique in zip(parsed, produced, strict=True)]
        )


def _words(text: str) -> list[str]:
    chunks = _SEPARATORS.split(text)
    if len(chunks) == 1:
        chunks = text.split()
    words = []
    for chunk in chunks:
        tokens = chunk.lower().split()
        cleaned = [_PUNCTUATION.sub("", token) for token in tokens]
        if 1 <= len(tokens) <= 2 and all(_SPELLED.fullmatch(token) for token in cleaned):
            words.append(" ".join(cleaned))
    return words


def _one_of(
    value: Any,
    allowed: Sequence[str],
    column: str,
    position: int
) -> str:
    if value not in allowed:
        raise ValueError(f"Row {position} has '{value}' in '{column}', which isn't one of {list(allowed)}.")
    return str(value)


def _missing(value: Any) -> bool:
    return value is None or (np.ndim(value) == 0 and pd.isna(value))
