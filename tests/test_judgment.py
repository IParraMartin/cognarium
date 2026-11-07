"""Anchoring, framing, the conjunction fallacy and false belief."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from importlib import resources

import pytest
from helpers import ScriptedSubject

import cognarium as cg
from cognarium.paradigms import judgment
from cognarium.paradigms.base import load_items

ITEM_FILES = ("anchoring", "framing", "conjunction", "false_belief")


def trials_of(paradigm, participant=0):
    return paradigm.sequences()[participant]


def by_item(paradigm):
    grouped = defaultdict(list)
    for trial in trials_of(paradigm):
        grouped[trial.item].append(trial)
    return grouped


def test_anchoring_options_do_not_change_with_the_anchor():
    grouped = by_item(judgment.Anchoring())
    assert len(grouped) >= 24
    for pair in grouped.values():
        assert sorted(trial.condition["anchor"] for trial in pair) == ["high", "low"]
        assert pair[0].options == pair[1].options
        assert len(pair[0].options) == 5


def test_anchoring_prompt_holds_the_condition_anchor():
    rows = {row["question"]: row for row in load_items("anchoring")}
    for trial in trials_of(judgment.Anchoring()):
        row = rows[trial.item]
        anchor = row[f"{trial.condition['anchor']}_anchor"]
        assert trial.condition["anchor_value"] == anchor
        question = row["question"]
        assert trial.stimulus.text == f"Is {question} more or less than {anchor}? Your estimate of {question}:"


def test_anchoring_expected_option_hand_value(tmp_path):
    subject = ScriptedSubject(probabilities=(0.1, 0.2, 0.3, 0.2, 0.2))
    results = cg.run(
        judgment.Anchoring(),
        subject,
        out=tmp_path / "run",
        progress=False
    )
    # 0 * 0.1 + 1 * 0.2 + 2 * 0.3 + 3 * 0.2 + 4 * 0.2
    primary = results.score().iloc[0]
    assert primary["mean_a"] == pytest.approx(2.2)
    assert primary["mean_b"] == pytest.approx(2.2)


def test_anchoring_reads_bins_written_as_text():
    rows = [
        {
            "question": "the number of moons of Mars",
            "low_anchor": 0,
            "high_anchor": 10,
            "bins": "none|one|two|three|more"
        }
    ]
    trial = trials_of(judgment.Anchoring(stimuli=rows))[0]
    assert trial.options == ("none", "one", "two", "three", "more")


def test_framing_outcomes_match_across_frames():
    for row in load_items("framing"):
        numerator, denominator = row["chance"]
        total, sure = row["total"], row["sure"]
        # the sure option equals the gamble's expected outcome, in both frames
        assert sure * denominator == total * numerator
        assert f"{sure} " in row["gain_text"] and f"all {total} " in row["gain_text"]
        assert f"{total - sure} " in row["loss_text"] and f"all {total} " in row["loss_text"]
        assert row["gain_text"].split(" If plan A")[0] == row["loss_text"].split(" If plan A")[0]


def test_framing_scenarios_pair_across_frames():
    grouped = by_item(judgment.Framing())
    assert len(grouped) >= 24
    for pair in grouped.values():
        assert sorted(trial.condition["frame"] for trial in pair) == ["gain", "loss"]
        assert all(trial.options == ("sure", "gamble") for trial in pair)
        assert all("sure" in trial.stimulus.text and "gamble" in trial.stimulus.text for trial in pair)


def test_conjunction_answer_is_always_the_single_property():
    rows = load_items("conjunction")
    for trial in trials_of(judgment.ConjunctionFallacy()):
        row = rows[trial.item]
        statements = re.findall(r"^[12]: (.*)$", trial.stimulus.text, flags=re.MULTILINE)
        assert sorted(statements) == sorted([row["single"], row["conjunction"]])
        assert statements[int(trial.answer) - 1] == row["single"]
        assert trial.stimulus.text.startswith(row[trial.condition["description"]])


def test_conjunction_positions_are_balanced():
    trials = trials_of(judgment.ConjunctionFallacy())
    counts = Counter((trial.condition["description"], trial.answer) for trial in trials)
    assert counts[("representative", "1")] == counts[("representative", "2")]
    assert counts[("neutral", "1")] == counts[("neutral", "2")]


def test_false_belief_answer_is_the_first_place_only_when_the_belief_is_false():
    rows = load_items("false_belief")
    for trial in trials_of(judgment.FalseBelief()):
        row = rows[trial.item]
        assert sorted(trial.options) == sorted([row["first_place"], row["second_place"]])
        expected = row["first_place"] if trial.condition["belief"] == "false" else row["second_place"]
        assert trial.answer == expected


def test_false_belief_story_mentions_who_saw_the_move():
    rows = load_items("false_belief")
    for trial in trials_of(judgment.FalseBelief()):
        name = rows[trial.item]["name"]
        saw = f"While {name} watches" in trial.stimulus.text
        away = f"While {name} is away" in trial.stimulus.text
        assert saw != away
        assert saw == (trial.condition["belief"] == "true")
        assert trial.stimulus.text.endswith(f"Where will {name} look for the {rows[trial.item]['object']} first?")


def test_false_belief_positions_are_balanced():
    counts = Counter(
        (trial.condition["belief"], trial.options.index(trial.answer))
        for trial in trials_of(judgment.FalseBelief())
    )
    assert counts[("false", 0)] == counts[("false", 1)]
    assert counts[("true", 0)] == counts[("true", 1)]


def test_false_belief_stories_vary_names_and_places():
    rows = load_items("false_belief")
    assert len({row["name"] for row in rows}) == len(rows) >= 24
    assert len({(row["first_place"], row["second_place"]) for row in rows}) > len(rows) // 2


def test_items_are_our_own_wording():
    folder = resources.files("cognarium.paradigms") / "items"
    for name in ITEM_FILES:
        text = (folder / f"{name}.json").read_text(encoding="utf-8")
        assert not re.search(r"\b(Linda|Sally|Anne)\b", text), name
        assert "asian disease" not in text.lower(), name
        assert "not human-normed" in text, name


def test_every_judgment_item_file_has_enough_items():
    for name in ITEM_FILES:
        assert len(load_items(name)) >= 24, name
