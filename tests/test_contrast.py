from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from helpers import ScriptedSubject
from paradigm_cases import EffectSubject

import cognarium as cg
from cognarium.metrics.measures import ROW_MEASURES
from cognarium.metrics.statistics import contrast_table

PAIRS = [
    ("doctor", "nurse", "related"),
    ("bread", "butter", "related"),
    ("cup", "nurse", "unrelated"),
    ("sky", "butter", "unrelated")
]


class Priming(cg.Paradigm):
    effects = (
        cg.Effect(
            "priming",
            "answer_surprisal",
            by="relatedness",
            levels=("unrelated", "related")
        ),
        cg.Effect("correctness", "answer_probability", against=0.5)
    )

    def trials(self, rng):
        for prime, target, relatedness in self.pairs:
            yield cg.Trial(
                cg.Stimulus(text=f"{prime} {target}"),
                options=["word", "nonword"],
                answer="word",
                condition={"relatedness": relatedness},
                item=target
            )


def scored(values_a, values_b, *, paired=True):
    """Builds a table with one row per group and level."""
    offset = 0 if paired else len(values_a)
    rows = [{"item": index, "level": "a", "value": value} for index, value in enumerate(values_a)]
    rows += [{"item": index + offset, "level": "b", "value": value} for index, value in enumerate(values_b)]
    return pd.DataFrame(rows)


def run_quietly(paradigm, subject, folder):
    return cg.run(
        paradigm,
        subject,
        out=folder,
        progress=False
    )


def probabilities_frame():
    """Two rows with known probabilities, written out as a run would store them."""
    log_probs = [
        np.log([0.25, 0.75]).tolist(),
        np.log([0.5, 0.25, 0.25]).tolist()
    ]
    return pd.DataFrame({
        "options": [["a", "b"], ["a", "b", "c"]],
        "answer": ["b", "c"],
        "response": ["a", "c"],
        "log_probs": log_probs,
        "shaped_log_probs": log_probs
    })


def test_contrast_matches_a_hand_value_when_paired():
    table = contrast_table(
        scored([1.0, 2.0, 3.0], [0.0, 1.0, 1.0]),
        "value",
        by="level",
        levels=("a", "b")
    )
    assert table["difference"].item() == pytest.approx(4 / 3)
    assert table["mean_a"].item() == pytest.approx(2.0)
    assert table["mean_b"].item() == pytest.approx(2 / 3)
    # the differences are 1, 1 and 2, so a bootstrap of pairs can't leave [1, 2]
    assert 1.0 <= table["ci_low"].item() <= table["ci_high"].item() <= 2.0
    assert table["detected"].item()


def test_contrast_is_not_detected_when_levels_match():
    table = contrast_table(
        scored([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
        "value",
        by="level",
        levels=("a", "b")
    )
    assert table["difference"].item() == 0
    assert table["ci_low"].item() == 0 and table["ci_high"].item() == 0
    assert not table["detected"].item()


def test_rounding_alone_is_not_detected():
    table = contrast_table(
        scored([1.0, 1.0], [1.0 + 1e-15, 1.0 + 1e-15]),
        "value",
        by="level",
        levels=("b", "a")
    )
    assert not table["detected"].item()


def test_contrast_pairs_only_when_groups_overlap():
    values_a, values_b = [1.0, 2.0, 3.0, 9.0], [0.0, 1.0, 2.0, 8.0]
    paired = contrast_table(
        scored(values_a, values_b),
        "value",
        by="level",
        levels=("a", "b")
    )
    apart = contrast_table(
        scored(values_a, values_b, paired=False),
        "value",
        by="level",
        levels=("a", "b")
    )
    # every paired difference is 1, so only the unpaired interval sees the spread within levels
    assert paired["ci_low"].item() == pytest.approx(1.0) and paired["ci_high"].item() == pytest.approx(1.0)
    assert apart["ci_high"].item() - apart["ci_low"].item() > 1.0
    assert apart["difference"].item() == pytest.approx(1.0)


def test_contrast_defaults_to_the_later_level_minus_the_earlier():
    frame = scored([2.0, 2.0], [1.0, 1.0]).replace({"level": {"a": True, "b": False}})
    table = contrast_table(frame, "value", by="level")
    assert (table["level_a"].item(), table["level_b"].item()) == (True, False)
    assert table["difference"].item() == pytest.approx(1.0)


def test_contrast_against_a_value():
    frame = pd.DataFrame({"item": [0, 1, 2, 3], "value": [0.7, 0.8, 0.9, 0.8]})
    table = contrast_table(frame, "value", against=0.5)
    assert table["level_b"].item() == "chance"
    assert table["difference"].item() == pytest.approx(0.3)
    assert table["mean_b"].item() == 0.5
    assert table["detected"].item()
    other = contrast_table(frame, "value", against=0.25)
    assert other["level_b"].item() == "0.25"


def test_contrast_against_a_value_can_use_one_level():
    frame = pd.DataFrame({
        "item": [0, 1, 2, 3],
        "kind": ["x", "x", "y", "y"],
        "value": [0.9, 0.9, 0.1, 0.1]
    })
    table = contrast_table(
        frame,
        "value",
        by="kind",
        levels=("x",),
        against=0.5
    )
    assert table["level_a"].item() == "x" and table["difference"].item() == pytest.approx(0.4)


def test_contrast_interval_widens_with_item_noise():
    quiet = contrast_table(
        scored([1.0, 1.1, 0.9, 1.0], [0.0, 0.0, 0.0, 0.0]),
        "value",
        by="level",
        levels=("a", "b")
    )
    noisy = contrast_table(
        scored([1.0, 3.0, -1.0, 1.0], [0.0, 0.0, 0.0, 0.0]),
        "value",
        by="level",
        levels=("a", "b")
    )
    assert noisy["ci_high"].item() - noisy["ci_low"].item() > quiet["ci_high"].item() - quiet["ci_low"].item()
    assert quiet["detected"].item() and not noisy["detected"].item()


def test_contrast_averages_rows_within_a_group():
    frame = pd.DataFrame({
        "item": [0, 0, 1, 0, 1],
        "level": ["a", "a", "a", "b", "b"],
        "value": [0.0, 2.0, 3.0, 0.0, 0.0]
    })
    table = contrast_table(
        frame,
        "value",
        by="level",
        levels=("a", "b")
    )
    assert table["mean_a"].item() == pytest.approx(2.0)


def test_contrast_gives_a_p_value_from_shuffling():
    many = contrast_table(
        scored(list(range(1, 13)), [0.0] * 12),
        "value",
        by="level",
        levels=("a", "b")
    )
    none = contrast_table(
        scored([1.0, 2.0], [1.0, 2.0]),
        "value",
        by="level",
        levels=("a", "b")
    )
    assert many["p_value"].item() < 0.01
    assert none["p_value"].item() == 1.0


def test_contrast_is_seeded():
    frame = scored([1.0, 3.1, 2.3, 5.7, 0.4, 7.9, 2.2], [0.0, 1.3, 0.5, 0.1, 0.9, 1.7, 0.6])
    first = contrast_table(
        frame,
        "value",
        by="level",
        levels=("a", "b"),
        seed=1
    )
    second = contrast_table(
        frame,
        "value",
        by="level",
        levels=("a", "b"),
        seed=1
    )
    third = contrast_table(
        frame,
        "value",
        by="level",
        levels=("a", "b"),
        seed=2
    )
    pd.testing.assert_frame_equal(first, second)
    assert first["ci_low"].item() != third["ci_low"].item()


def test_contrast_refuses_too_few_groups():
    with pytest.raises(ValueError, match="two groups"):
        contrast_table(
            scored([1.0], [0.0, 1.0]),
            "value",
            by="level",
            levels=("a", "b")
        )


def test_contrast_refuses_more_than_two_levels_without_choosing():
    frame = pd.concat([scored([1.0, 2.0], [0.0, 1.0]), pd.DataFrame({"item": [0, 1], "level": "c", "value": 1.0})])
    with pytest.raises(ValueError, match="levels"):
        contrast_table(frame, "value", by="level")
    assert contrast_table(
        frame,
        "value",
        by="level",
        levels=("a", "c")
    )["difference"].item() == pytest.approx(0.5)


def test_contrast_refuses_an_unknown_level_and_non_finite_values():
    frame = scored([1.0, 2.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="'z'"):
        contrast_table(
            frame,
            "value",
            by="level",
            levels=("a", "z")
        )
    with pytest.raises(ValueError, match="finite"):
        contrast_table(
            scored([1.0, math.inf], [0.0, 1.0]),
            "value",
            by="level",
            levels=("a", "b")
        )
    with pytest.raises(ValueError, match="finite"):
        contrast_table(
            scored([1.0, math.nan], [0.0, 1.0]),
            "value",
            by="level",
            levels=("a", "b")
        )


def test_contrast_refuses_a_missing_column_or_no_comparison():
    frame = scored([1.0, 2.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="'nothing'"):
        contrast_table(frame, "nothing", by="level")
    with pytest.raises(ValueError, match="`by` or `against`"):
        contrast_table(frame, "value")


def test_contrast_needs_one_level_when_comparing_with_a_value():
    with pytest.raises(ValueError, match="one level"):
        contrast_table(
            scored([1.0, 2.0], [0.0, 1.0]),
            "value",
            by="level",
            against=0.5
        )


def test_every_row_measure_matches_a_hand_value():
    frame = probabilities_frame()
    expected = {
        "surprisal": [2.0, 2.0],
        "answer_surprisal": [-math.log2(0.75), 2.0],
        "entropy": [-(0.25 * math.log2(0.25) + 0.75 * math.log2(0.75)), 1.5],
        "accuracy": [0.0, 1.0],
        "answer_probability": [0.75, 0.25],
        "answer_margin": [math.log(3.0), math.log(0.5)],
        "expected_option": [0.75, 0.75]
    }
    for name, values in expected.items():
        assert ROW_MEASURES[name](frame).tolist() == pytest.approx(values), name


def test_row_measures_read_the_raw_distribution_when_asked():
    frame = probabilities_frame().assign(shaped_log_probs=[[0.0, -math.inf], [0.0, -math.inf, -math.inf]])
    assert ROW_MEASURES["answer_probability"](frame).tolist() == pytest.approx([0.0, 0.0])
    assert ROW_MEASURES["answer_probability"](frame, raw=True).tolist() == pytest.approx([0.75, 0.25])


def test_option_probability_by_name_and_by_position():
    frame = probabilities_frame()
    assert ROW_MEASURES["option_probability"](frame, option="a").tolist() == pytest.approx([0.25, 0.5])
    assert ROW_MEASURES["option_probability"](frame, option=1).tolist() == pytest.approx([0.75, 0.25])
    with pytest.raises(ValueError, match="'z'"):
        ROW_MEASURES["option_probability"](frame, option="z")
    with pytest.raises(ValueError, match="position 2"):
        ROW_MEASURES["option_probability"](frame, option=2)


def test_answer_measures_refuse_rows_with_no_answer():
    frame = probabilities_frame().assign(answer=[None, "c"])
    for name in ("answer_surprisal", "accuracy", "answer_probability", "answer_margin"):
        with pytest.raises(cg.MissingAbilityError, match=name):
            ROW_MEASURES[name](frame)


def test_results_contrast_labels_proxies(tmp_path):
    results = run_quietly(Priming(pairs=PAIRS), ScriptedSubject(), tmp_path / "run")
    proxy = results.contrast("answer_surprisal", by="relatedness", levels=("unrelated", "related"))
    plain = results.contrast("accuracy", by="relatedness", levels=("unrelated", "related"))
    assert proxy["proxy_for"].item() == "reaction time"
    assert plain["proxy_for"].item() is None
    assert proxy["measure"].item() == "answer_surprisal"


def test_results_contrast_reads_a_numeric_column(tmp_path):
    results = run_quietly(Priming(pairs=PAIRS), ScriptedSubject(), tmp_path / "run")
    table = results.contrast("trial", against=0.0)
    assert table["measure"].item() == "trial" and table["proxy_for"].item() is None


def test_results_contrast_refuses_an_unknown_measure(tmp_path):
    results = run_quietly(Priming(pairs=PAIRS), ScriptedSubject(), tmp_path / "run")
    with pytest.raises(ValueError, match="numeric column"):
        results.contrast("nothing", by="relatedness")
    with pytest.raises(ValueError, match="numeric column"):
        results.contrast("response", by="relatedness")


def test_results_contrast_filters_rows(tmp_path):
    results = run_quietly(Priming(pairs=PAIRS), EffectSubject(Priming.effects[1]), tmp_path / "run")
    table = results.contrast("answer_probability", against=0.5, where={"relatedness": "related"})
    assert table["detected"].item()
    with pytest.raises(ValueError, match="'nothing'"):
        results.contrast("answer_probability", against=0.5, where={"nothing": 1})
    with pytest.raises(ValueError, match="no rows"):
        results.contrast("answer_probability", against=0.5, where={"relatedness": "other"})


def test_a_measure_that_a_top_p_made_infinite_suggests_raw(tmp_path):
    paradigm = Priming(pairs=PAIRS, top_k=1)
    results = run_quietly(paradigm, ScriptedSubject(probabilities=(0.2, 0.8)), tmp_path / "run")
    with pytest.raises(ValueError, match="raw=True"):
        results.contrast("answer_surprisal", by="relatedness", levels=("unrelated", "related"))
    table = results.contrast(
        "answer_surprisal",
        by="relatedness",
        levels=("unrelated", "related"),
        raw=True
    )
    assert table["difference"].item() == 0


def test_results_score_uses_the_declared_effects(tmp_path):
    results = run_quietly(Priming(pairs=PAIRS), EffectSubject(Priming.effects[0]), tmp_path / "run")
    table = results.score()
    assert table["effect"].tolist() == ["priming", "correctness"]
    assert table.columns[0] == "effect"
    assert table.iloc[0]["detected"] and table.iloc[0]["difference"] > 0


def test_a_paradigm_can_write_its_own_score(tmp_path):
    class Custom(Priming):
        def score(self, results):
            return results.contrast("surprisal", by="relatedness", levels=("unrelated", "related"))

    results = run_quietly(Custom(pairs=PAIRS), ScriptedSubject(), tmp_path / "run")
    assert results.score()["measure"].item() == "surprisal"


def test_a_paradigm_with_no_effects_says_so(tmp_path):
    class Plain(cg.Paradigm):
        def trials(self, rng):
            yield cg.Trial(cg.Stimulus(text="a"), options=["a", "b"])

    results = run_quietly(Plain(), ScriptedSubject(), tmp_path / "run")
    with pytest.raises(cg.MissingAbilityError, match="plain"):
        results.score()


def test_reopened_results_need_the_paradigm_to_score(tmp_path):
    paradigm = Priming(pairs=PAIRS)
    run_quietly(paradigm, ScriptedSubject(), tmp_path / "run")
    with pytest.raises(cg.MissingAbilityError, match="paradigm="):
        cg.Results.load(tmp_path / "run").score()
    assert len(cg.Results.load(tmp_path / "run", paradigm=paradigm).score()) == 2


def test_participant_interval_is_refused_for_identical_participants(tmp_path):
    with pytest.warns(cg.IdenticalParticipantsWarning):
        results = run_quietly(Priming(pairs=PAIRS, n_participants=3), ScriptedSubject(), tmp_path / "run")
    with pytest.raises(ValueError, match="identical"):
        results.contrast(
            "answer_surprisal",
            by="relatedness",
            levels=("unrelated", "related"),
            ci_by="participant"
        )
    assert len(results.contrast("answer_surprisal", by="relatedness", levels=("unrelated", "related"))) == 1


def test_identical_participants_warn_before_any_model_call(tmp_path):
    subject = ScriptedSubject()
    with pytest.warns(cg.IdenticalParticipantsWarning), pytest.raises(RuntimeError):
        run_quietly(Priming(pairs=PAIRS, n_participants=3), _failing(subject), tmp_path / "run")
    assert subject.calls == 0


def _failing(subject):
    subject.fail_after = 0
    return subject


def test_one_participant_does_not_warn(tmp_path, recwarn):
    run_quietly(Priming(pairs=PAIRS), ScriptedSubject(), tmp_path / "run")
    assert not [item for item in recwarn if item.category is cg.IdenticalParticipantsWarning]


def test_participant_interval_works_when_participants_differ(tmp_path, recwarn):
    class Remembered(Priming):
        context = cg.context.cumulative

    results = run_quietly(Remembered(pairs=PAIRS, n_participants=3), ScriptedSubject(), tmp_path / "run")
    table = results.contrast(
        "answer_surprisal",
        by="relatedness",
        levels=("unrelated", "related"),
        ci_by="participant"
    )
    assert table["ci_low"].item() <= table["difference"].item() <= table["ci_high"].item()
    assert not [item for item in recwarn if item.category is cg.IdenticalParticipantsWarning]


def test_effect_subject_gives_the_declared_direction(tmp_path):
    for measure, options in [
        ("answer_surprisal", {}),
        ("answer_probability", {}),
        ("answer_margin", {}),
        ("option_probability", {"option": "word"}),
        ("option_probability", {"option": 1}),
        ("expected_option", {})
    ]:
        effect = cg.Effect(
            "effect",
            measure,
            by="relatedness",
            levels=("unrelated", "related"),
            options=options
        )

        class Declared(Priming):
            effects = (effect,)

        folder = tmp_path / f"{measure}-{options.get('option')}"
        table = run_quietly(Declared(pairs=PAIRS), EffectSubject(effect), folder).score()
        assert table.iloc[0]["detected"] and table.iloc[0]["difference"] > 0, measure
        flat = run_quietly(Declared(pairs=PAIRS), EffectSubject(None), f"{folder}-none").score()
        assert not flat.iloc[0]["detected"], measure


def test_a_level_with_no_rows_is_refused(tmp_path):
    class Mislabelled(Priming):
        def trials(self, rng):
            for trial in super().trials(rng):
                yield cg.Trial(
                    trial.stimulus,
                    options=trial.options,
                    answer=trial.answer,
                    condition={"relatedness": "related"},
                    item=trial.item
                )

    subject = EffectSubject(Priming.effects[0])
    with pytest.raises(ValueError, match="'unrelated'"):
        run_quietly(Mislabelled(pairs=PAIRS), subject, tmp_path / "run").score()


def test_effects_and_warning_are_exposed():
    assert cg.Effect("e", "accuracy").by is None
    assert issubclass(cg.IdenticalParticipantsWarning, UserWarning)
