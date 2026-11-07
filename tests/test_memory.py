from __future__ import annotations

import pandas as pd
import pytest
from paradigm_cases import DRM_ROWS, EffectSubject

import cognarium as cg
from cognarium.paradigms.memory import (
    DRM,
    FreeRecall,
    NBack,
    Span,
    Sternberg
)


def trials(paradigm):
    return [trial for row in paradigm.sequences() for trial in row]


def run_quietly(paradigm, subject, folder):
    return cg.run(
        paradigm,
        subject,
        out=folder,
        progress=False
    )


def writing(text):
    """A subject that writes the same text for every trial."""
    return EffectSubject(None, text_for=lambda trial, effect: text)


def test_n_back_answers_match_the_sequence():
    for sequence in NBack(levels=(1, 2), sequence_length=30).sequences():
        sequence = [trial for trial in sequence if trial.condition["load"] == 2]
        letters = [trial.condition["letter"] for trial in sequence]
        for position, trial in enumerate(sequence):
            expected = position >= 2 and letters[position] == letters[position - 2]
            assert (trial.answer == "yes") == expected
            assert f"Letter: {letters[position]}" in trial.stimulus.text


def test_n_back_target_rate_is_close_to_asked():
    sequence = NBack(levels=(1, 2), sequence_length=32, target_rate=0.25).sequences()[0]
    sequence = [trial for trial in sequence if trial.condition["load"] == 2]
    assert [trial.condition["trial_type"] for trial in sequence].count("target") == 8


def test_n_back_lures_match_at_another_distance():
    for sequence in NBack(levels=(1, 2), sequence_length=40, seed=5).sequences():
        sequence = [trial for trial in sequence if trial.condition["load"] == 2]
        letters = [trial.condition["letter"] for trial in sequence]
        for position, trial in enumerate(sequence):
            near = [position - 1, position - 3]
            lure = any(0 <= other and letters[other] == letters[position] for other in near)
            if trial.condition["trial_type"] == "lure":
                assert lure and trial.answer == "no"
            if trial.condition["trial_type"] == "filler":
                assert not lure


def test_n_back_leaves_out_the_first_letters_of_a_block():
    for trial in NBack(levels=(1, 3), sequence_length=10).sequences()[0]:
        assert trial.condition["scored"] == (trial.condition["position"] >= trial.condition["load"])


def test_each_n_back_level_is_its_own_block(tmp_path):
    paradigm = NBack(levels=(1, 2), sequence_length=6)
    blocks = {}
    for trial in paradigm.sequences()[0]:
        blocks.setdefault(trial.block, set()).add(trial.condition["load"])
    assert sorted(len(loads) for loads in blocks.values()) == [1, 1]
    frame = run_quietly(paradigm, EffectSubject(None), tmp_path / "run").to_pandas()
    starts = frame[frame["position"] == 0]
    assert starts["history"].map(len).tolist() == [0, 0]


def test_n_back_block_order_is_counterbalanced():
    rows = NBack(levels=(1, 2), sequence_length=6, n_participants=8).sequences()
    orders = {tuple(trial.condition["load"] for trial in row[::6]) for row in rows}
    assert orders == {(1, 2), (2, 1)}


def test_n_back_d_prime_comes_from_metrics(tmp_path):
    paradigm = NBack(levels=(1, 2), sequence_length=12)
    results = run_quietly(paradigm, EffectSubject(paradigm.effects[0]), tmp_path / "run")
    assert results.metrics("d_prime", positive="yes")["load"].tolist() == [1, 2]


def test_n_back_needs_two_levels_to_compare():
    with pytest.raises(ValueError, match="two different levels"):
        NBack(levels=(2,))
    with pytest.raises(ValueError, match="two different levels"):
        NBack(levels=(2, 2))


def test_n_back_refuses_a_sequence_too_short():
    with pytest.raises(ValueError, match="sequence_length"):
        NBack(levels=(1, 3), sequence_length=4)


def test_span_parses_digits_in_order():
    paradigm = Span()
    frame = pd.DataFrame({
        "response": ["4, 7 and 1", "1 7 4", "", "٣ 4"],
        "answer": ["4 7 1"] * 4
    })
    parsed = paradigm.annotate(frame)
    assert parsed["parsed"].tolist() == [
        ["4", "7", "1"],
        ["1", "7", "4"],
        [],
        ["4"]
    ]
    assert parsed["correct"].tolist() == [1.0, 0.0, 0.0, 0.0]


def test_span_lists_have_the_lengths_they_claim():
    for trial in trials(Span(lengths=(3, 6))):
        digits = trial.answer.split()
        assert len(digits) == trial.condition["length"]
        assert f"Remember these digits: {' '.join(digits)}." in trial.stimulus.text


def test_span_budget_fits_the_longest_list():
    assert Span(lengths=(3, 9)).max_new_tokens == 22
    assert Span(lengths=(3, 9), max_new_tokens=5).max_new_tokens == 5


def test_span_counts_garbled_output_as_wrong(tmp_path):
    frame = run_quietly(Span(lengths=(2, 3), n_per_length=2), writing("no digits here"), tmp_path / "run").to_pandas()
    assert frame["correct"].tolist() == [0.0] * 4


def test_span_estimate_hand_value(tmp_path):
    paradigm = Span(lengths=(2, 4, 6), n_per_length=2)
    subject = EffectSubject(None, text_for=lambda trial, effect: trial.answer if trial.condition["length"] < 6 else "")
    results = run_quietly(paradigm, subject, tmp_path / "run")
    assert paradigm.memory_span(results) == 4
    paradigm = Span(lengths=(2, 4), n_per_length=2)
    assert paradigm.memory_span(run_quietly(paradigm, writing(""), tmp_path / "none")) == 0


def test_sternberg_probe_presence_matches_the_answer():
    for trial in trials(Sternberg()):
        members = trial.stimulus.text.split("Memory set: ")[1].split(".")[0].split()
        probe = trial.stimulus.text.split("Probe: ")[1][0]
        assert (probe in members) == (trial.answer == "yes")
        assert trial.condition["probe"] == ("present" if probe in members else "absent")


def test_sternberg_set_sizes_are_what_they_say():
    paradigm = Sternberg(set_sizes=(2, 5), n_per_size=6)
    sizes = []
    for trial in trials(paradigm):
        members = trial.stimulus.text.split("Memory set: ")[1].split(".")[0].split()
        assert len(members) == len(set(members)) == trial.condition["set_size"]
        sizes.append(trial.condition["set_size"])
    assert sizes.count(2) == sizes.count(5) == 6


def test_sternberg_probes_are_half_present():
    probes = [trial.condition["probe"] for trial in trials(Sternberg(n_per_size=8))]
    assert probes.count("present") == probes.count("absent")


def test_free_recall_lists_do_not_share_words_within_a_participant():
    for row in FreeRecall(n_lists=8, list_length=12, n_participants=2).sequences():
        words = [word for trial in row for word in trial.condition["studied"]]
        assert len(words) == len(set(words)) == 96


def test_free_recall_needs_enough_words():
    with pytest.raises(ValueError, match="words"):
        FreeRecall(n_lists=20, list_length=12)


def test_free_recall_counts_recalled_intrusions_and_repeats():
    paradigm = FreeRecall(list_length=6, n_lists=2)
    studied = ["apple", "boat", "cup", "door", "egg", "fig"]
    frame = pd.DataFrame({"response": ["Fig, boat; apple\nfig, zebra, Cup!"], "studied": [studied]})
    row = paradigm.annotate(frame).iloc[0]
    assert row["recalled"] == ["fig", "boat", "apple", "cup"]
    assert (row["n_recalled"], row["n_intrusions"], row["n_repeats"]) == (4, 1, 1)
    assert (row["early"], row["middle"], row["late"]) == (1.0, 0.5, 0.5)


def test_free_recall_handles_empty_and_non_ascii_output(tmp_path):
    for text in ["", "ünïcødé, 日本語", ",,,\n\n"]:
        frame = run_quietly(FreeRecall(list_length=6, n_lists=2), writing(text), tmp_path / str(hash(text))).to_pandas()
        assert frame["n_recalled"].tolist() == [0, 0]
        assert frame[["early", "middle", "late"]].to_numpy().sum() == 0


def test_free_recall_serial_positions_hand_value(tmp_path):
    def recall_last(trial, effect):
        return ", ".join(trial.condition["studied"][-2:])

    paradigm = FreeRecall(list_length=6, n_lists=3)
    results = run_quietly(paradigm, EffectSubject(None, text_for=recall_last), tmp_path / "run")
    table = results.score()
    assert table["effect"].tolist() == ["recency", "primacy"]
    assert table.iloc[0]["difference"] == pytest.approx(1.0)
    assert table.iloc[1]["difference"] == pytest.approx(0.0)
    assert table["measure"].tolist() == ["recall", "recall"]


def test_drm_requires_lists():
    with pytest.raises(ValueError, match="stimuli"):
        DRM()


def test_drm_studies_each_list_then_tests_every_word():
    for trial in trials(DRM(stimuli=DRM_ROWS)):
        role = trial.condition["role"]
        studied = trial.stimulus.text.split("Study these words: ")[1].split(".")[0].split(", ")
        probe = trial.stimulus.text.split("Word: ")[1]
        assert (probe in studied) == (role == "studied")
        assert trial.answer == ("old" if role == "studied" else "new")


def test_drm_scores_a_lure_as_new_by_answer(tmp_path):
    paradigm = DRM(stimuli=DRM_ROWS)
    results = run_quietly(paradigm, EffectSubject(paradigm.effects[0]), tmp_path / "run")
    lures = results.to_pandas().query("role == 'lure'")
    assert set(lures["answer"]) == {"new"}
    assert results.score().iloc[0]["detected"]


def test_drm_refuses_an_unknown_role():
    with pytest.raises(ValueError, match="'critical'"):
        DRM(stimuli=[*DRM_ROWS, {"list": "x", "word": "y", "role": "critical"}])


def test_drm_refuses_a_list_with_nothing_to_study():
    with pytest.raises(ValueError, match="'empty'"):
        DRM(stimuli=[*DRM_ROWS, {"list": "empty", "word": "y", "role": "lure"}])
