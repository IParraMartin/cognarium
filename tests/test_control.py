from __future__ import annotations

import pytest
from paradigm_cases import EffectSubject

import cognarium as cg
from cognarium.paradigms.control import (
    Flanker,
    GoNoGo,
    Simon,
    StopSignal,
    Stroop,
    TaskSwitching
)


def trials(paradigm):
    return [trial for row in paradigm.sequences() for trial in row]


def test_stroop_stimuli_say_what_they_claim():
    for trial in trials(Stroop()):
        word, ink = trial.condition["word"], trial.answer
        assert (trial.condition["congruency"] == "congruent") == (word == ink)
        assert f"The word {word.upper()} is printed in {ink} ink" in trial.stimulus.text
        assert trial.options == ("red", "blue", "green", "yellow")


def test_stroop_conditions_are_balanced():
    kinds = [trial.condition["congruency"] for trial in trials(Stroop(n_repeats=2))]
    assert kinds.count("congruent") == kinds.count("incongruent") == 24


def test_stroop_takes_stimuli_rows_including_neutral_words():
    rows = [
        {"word": "red", "ink": "red"},
        {"word": "blue", "ink": "red"},
        {"word": "table", "ink": "blue"}
    ]
    kinds = sorted(trial.condition["congruency"] for trial in trials(Stroop(stimuli=rows)))
    assert kinds == ["congruent", "incongruent", "neutral"]


def test_unknown_or_too_few_colors_fail():
    with pytest.raises(ValueError, match="two colors"):
        Stroop(colors=["red"])
    with pytest.raises(ValueError, match="Row 0.*'purple'"):
        Stroop(stimuli=[{"word": "red", "ink": "purple"}])


def test_flanker_target_is_the_middle_letter():
    for trial in trials(Flanker(flankers=3)):
        letters = trial.stimulus.text.split()[0]
        assert len(letters) == 7 and letters[3] == trial.answer
        assert (trial.condition["congruency"] == "congruent") == (set(letters) == {trial.answer})


def test_flanker_conditions_are_balanced():
    kinds = [trial.condition["congruency"] for trial in trials(Flanker())]
    assert kinds.count("congruent") == kinds.count("incongruent")


def test_simon_answer_follows_the_rule_not_the_side():
    for trial in trials(Simon()):
        text = trial.stimulus.text
        left = text.split("Press left for ")[1].split(" ")[0]
        color = trial.condition["color"]
        assert trial.answer == ("left" if color == left else "right")
        assert f"appears on the {trial.condition['side']}" in text


def test_simon_correspondence_is_labelled_correctly():
    seen = set()
    for trial in trials(Simon()):
        corresponding = trial.answer == trial.condition["side"]
        assert trial.condition["correspondence"] == ("corresponding" if corresponding else "non_corresponding")
        seen.add(trial.condition["correspondence"])
    assert seen == {"corresponding", "non_corresponding"}


def test_simon_mappings_are_counterbalanced():
    lefts = [trial.condition["left_color"] for trial in trials(Simon())]
    assert lefts.count("red") == lefts.count("blue")


def test_go_no_go_keeps_its_ratio_and_answers():
    sequence = GoNoGo(n_trials=40).sequences()[0]
    kinds = [trial.condition["trial_type"] for trial in sequence]
    assert kinds.count("go") == 30
    for trial in sequence:
        is_no_go = "Letter: X" in trial.stimulus.text
        assert (trial.condition["trial_type"] == "no_go") == is_no_go
        assert trial.answer == ("withhold" if is_no_go else "press")


def test_go_no_go_ratio_must_leave_both_kinds():
    with pytest.raises(ValueError, match="go_ratio"):
        GoNoGo(go_ratio=1.0)
    with pytest.raises(ValueError, match="two"):
        GoNoGo(n_trials=4, go_ratio=0.9)


def test_go_no_go_d_prime_comes_from_metrics(tmp_path):
    paradigm = GoNoGo(n_trials=16)
    results = cg.run(
        paradigm,
        EffectSubject(paradigm.effects[0]),
        out=tmp_path / "run",
        progress=False
    )
    table = results.metrics("d_prime", positive="press")
    assert len(table) == 1 and table["value"].item() > 0


def test_stop_signal_delay_changes_the_text_and_no_ssrt_is_reported(tmp_path):
    paradigm = StopSignal(n_trials=24)
    stops = [trial for trial in trials(paradigm) if trial.condition["trial_type"] == "stop"]
    for trial in stops:
        assert trial.stimulus.text.endswith(" ." * trial.condition["delay"] + " STOP")
        assert trial.answer == "withhold"
    assert sorted({trial.condition["delay"] for trial in stops}) == [1, 5]
    go = [trial for trial in trials(paradigm) if trial.condition["trial_type"] == "go"]
    assert all("STOP" not in trial.stimulus.text.split("Letter:")[1] for trial in go)
    results = cg.run(
        paradigm,
        EffectSubject(paradigm.effects[0]),
        out=tmp_path / "run",
        progress=False
    )
    table = results.score()
    assert table["effect"].tolist() == ["late_signal", "stopping"]
    assert not any("ssrt" in str(value).lower() for value in [*table.columns, *table["effect"]])


def test_stop_signal_delays_are_balanced():
    stops = [trial for trial in trials(StopSignal(n_trials=40)) if trial.condition["trial_type"] == "stop"]
    delays = [trial.condition["delay"] for trial in stops]
    assert delays.count(1) == delays.count(5) == 5


def test_stop_signal_needs_enough_stops():
    with pytest.raises(ValueError, match="stop trials"):
        StopSignal(n_trials=8)


def test_task_switching_labels_transitions_from_the_previous_task():
    sequence = TaskSwitching(n_trials=30).sequences()[0]
    assert sequence[0].condition["transition"] == "start"
    for before, after in zip(sequence, sequence[1:], strict=False):
        same = before.condition["task"] == after.condition["task"]
        assert after.condition["transition"] == ("repeat" if same else "switch")
    transitions = [trial.condition["transition"] for trial in sequence[1:]]
    assert transitions.count("switch") == 14


def test_task_switching_answers_follow_the_current_task():
    for trial in TaskSwitching(n_trials=30).sequences()[0]:
        pair = trial.stimulus.text.split()[2]
        letter, number = pair[0], int(pair[1])
        if trial.condition["task"] == "letter":
            assert "vowel" in trial.stimulus.text
            assert trial.answer == ("yes" if letter in "AEIU" else "no")
        else:
            assert "even" in trial.stimulus.text
            assert trial.answer == ("yes" if number % 2 == 0 else "no")


def test_task_switching_leaves_the_first_trial_out_of_the_effect():
    assert TaskSwitching().effects[0].levels == ("switch", "repeat")


def test_paradigms_without_item_columns_refuse_stimuli():
    with pytest.raises(ValueError, match="item_columns"):
        GoNoGo(stimuli=[{"letter": "K"}])
