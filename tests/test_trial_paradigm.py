from __future__ import annotations

import pandas as pd
import pytest
import torch

import cognarium as cg
from helpers import ScriptedSubject

PAIRS = [("doctor", "nurse", True), ("bread", "nurse", False)]


class Priming(cg.Paradigm):
    context = cg.context.isolated

    def trials(self, rng):
        for prime, target, related in self.pairs:
            yield cg.Trial(
                stimulus=cg.Stimulus(text=f"{prime} {target}"),
                options=["word", "nonword"],
                answer="word",
                condition={"related": related}
            )


class Letters(cg.Paradigm):
    """A numbered list of trials, whose answers differ from what the subject says."""

    ordered = True

    def trials(self, rng):
        for position in range(self.n_trials):
            yield cg.Trial(
                cg.Stimulus(text=f"letter {position}"),
                options=["yes", "no"],
                answer="no",
                block=position // self.block_size
            )


class Remembering(Letters):
    context = cg.context.cumulative


class Recent(Letters):
    context = cg.context.window(2)


class Corrected(Letters):
    context = cg.context.window(3, feedback=True)


class Watching(ScriptedSubject):
    """A scripted subject that keeps every query it was asked."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seen = []

    def respond(self, queries, *, capture_logits=False):
        self.seen.extend(queries)
        return super().respond(queries, capture_logits=capture_logits)

    def contexts(self):
        return [[(item.stimulus.text, item.label) for item in query.context] for query in self.seen]


def run_quietly(
    paradigm,
    subject,
    folder,
    **options
):
    return cg.run(
        paradigm,
        subject,
        out=folder,
        progress=False,
        **options
    )


def test_the_spec_paradigm_runs_with_no_init(tmp_path):
    results = run_quietly(Priming(pairs=PAIRS), ScriptedSubject(), tmp_path / "run")
    frame = results.to_pandas()
    assert frame["related"].tolist() in ([True, False], [False, True])
    assert Priming.name == "priming"
    assert Priming(pairs=PAIRS).pairs == PAIRS


def test_class_names_become_snake_case_names():
    class GoNoGo(Priming):
        pass

    class DRM(Priming):
        pass

    class NBack(Priming):
        pass

    assert (GoNoGo.name, DRM.name, NBack.name) == ("go_no_go", "drm", "n_back")


def test_conditions_become_the_default_grouping():
    assert Priming(pairs=PAIRS).group_by == ("related",)


def test_a_declared_grouping_wins():
    class Grouped(Priming):
        group_by = ("other",)

    assert Grouped(pairs=PAIRS).group_by == ("other",)


def test_modalities_come_from_the_trials():
    assert Priming(pairs=PAIRS).modalities == {"text"}


def test_isolated_trials_see_no_past(tmp_path):
    subject = Watching()
    example = ("a demonstration", "word")
    paradigm = Priming(pairs=PAIRS * 3, examples=[example])
    run_quietly(paradigm, subject, tmp_path / "run")
    assert subject.contexts() == [[("a demonstration", "word")]] * 6


def test_a_large_isolated_paradigm_is_chunked(tmp_path):
    subject = Watching()
    paradigm = Letters(n_trials=600, block_size=600)
    frame = run_quietly(paradigm, subject, tmp_path / "run").to_pandas()
    assert frame.groupby("step").size().tolist() == [256, 256, 88]


def test_cumulative_context_holds_the_subjects_own_responses(tmp_path):
    subject = Watching(choices=0)
    paradigm = Remembering(n_trials=3, block_size=3)
    frame = run_quietly(paradigm, subject, tmp_path / "run").to_pandas()
    assert subject.contexts() == [
        [],
        [("letter 0", "yes")],
        [("letter 0", "yes"), ("letter 1", "yes")]
    ]
    assert [list(row) for row in frame["history"]] == [[], ["yes"], ["yes", "yes"]]


def test_window_keeps_only_the_last_k(tmp_path):
    subject = Watching(choices=0)
    run_quietly(Recent(n_trials=4, block_size=4), subject, tmp_path / "run")
    assert subject.contexts()[3] == [("letter 1", "yes"), ("letter 2", "yes")]


def test_feedback_shows_the_answer_not_the_response(tmp_path):
    subject = Watching(choices=0)
    run_quietly(Corrected(n_trials=2, block_size=2), subject, tmp_path / "run")
    assert subject.contexts()[1] == [("letter 0", "no")]


def test_feedback_without_an_answer_fails(tmp_path):
    class Unanswered(cg.Paradigm):
        context = cg.context.window(1, feedback=True)

        def trials(self, rng):
            for position in range(2):
                yield cg.Trial(cg.Stimulus(text=f"t{position}"), options=["a", "b"])

    with pytest.raises(ValueError, match="answer"):
        run_quietly(Unanswered(), ScriptedSubject(), tmp_path / "run")


def test_a_new_block_clears_the_context(tmp_path):
    subject = Watching(choices=0)
    run_quietly(Remembering(n_trials=4, block_size=2), subject, tmp_path / "run")
    assert subject.contexts() == [
        [],
        [("letter 0", "yes")],
        [],
        [("letter 2", "yes")]
    ]


def test_window_needs_at_least_one_trial():
    with pytest.raises(ValueError, match="at least 1"):
        cg.context.window(0)


def test_examples_come_before_past_trials(tmp_path):
    subject = Watching(choices=0)
    paradigm = Remembering(n_trials=2, block_size=2, examples=[("shown first", "no")])
    run_quietly(paradigm, subject, tmp_path / "run")
    assert subject.contexts()[1] == [("shown first", "no"), ("letter 0", "yes")]


def test_participants_get_their_own_order_unless_ordered():
    shuffled = Priming(pairs=[(str(i), "x", True) for i in range(20)], n_participants=2).sequences()
    first, second = ([trial.stimulus.text for trial in row] for row in shuffled)
    assert first != second and sorted(first) == sorted(second)
    fixed = Letters(n_trials=5, block_size=5, n_participants=2).sequences()
    assert [trial.item for trial in fixed[0]] == [0, 1, 2, 3, 4]
    assert [trial.item for trial in fixed[1]] == [0, 1, 2, 3, 4]


def test_shuffling_keeps_blocks_in_order():
    class Blocked(Letters):
        ordered = False

    for row in Blocked(n_trials=30, block_size=10, seed=3).sequences():
        assert [trial.block for trial in row] == sorted(trial.block for trial in row)


def test_items_line_up_across_participants():
    rows = Priming(pairs=[(str(i), "x", True) for i in range(10)], n_participants=2).sequences()
    by_item = [{trial.item: trial.stimulus.text for trial in row} for row in rows]
    assert by_item[0] == by_item[1]


def test_sequences_depend_only_on_the_seed():
    build = lambda seed: Priming(pairs=[(str(i), "x", True) for i in range(10)], seed=seed)
    text = lambda paradigm: [trial.stimulus.text for trial in paradigm.sequences()[0]]
    assert text(build(1)) == text(build(1))
    assert text(build(1)) != text(build(2))


def test_participant_columns_and_groups_reach_the_table(tmp_path):
    subject = Watching()
    paradigm = Priming(pairs=PAIRS, n_participants=3)
    with pytest.warns(cg.IdenticalParticipantsWarning):
        frame = run_quietly(paradigm, subject, tmp_path / "run").to_pandas()
    assert sorted(frame["participant"].tolist()) == [0, 0, 1, 1, 2, 2]
    assert set(frame["trial"]) == {0, 1}
    assert [query.group for query in subject.seen] == frame["participant"].tolist()


def test_contextual_participants_run_in_parallel(tmp_path):
    class Uneven(cg.Paradigm):
        context = cg.context.cumulative
        ordered = True

        def trials(self, rng):
            # participants end at different points when their random lengths differ
            for position in range(int(rng.integers(2, 6))):
                yield cg.Trial(cg.Stimulus(text=f"t{position}"), options=["a", "b"])

    paradigm = Uneven(n_participants=6)
    lengths = [len(row) for row in paradigm.sequences()]
    frame = run_quietly(paradigm, ScriptedSubject(), tmp_path / "run").to_pandas()
    assert len(set(lengths)) > 1
    assert frame.groupby("step").size().tolist()[0] == 6
    assert frame.groupby("participant").size().tolist() == lengths


def test_personas_cycle_and_come_before_instructions(tmp_path):
    subject = Watching()
    paradigm = Letters(
        n_trials=1,
        block_size=1,
        n_participants=3,
        personas=["You are A.", "You are B."],
        instructions="Answer."
    )
    run_quietly(paradigm, subject, tmp_path / "run")
    assert [query.instructions for query in subject.seen] == [
        "You are A.\nAnswer.",
        "You are B.\nAnswer.",
        "You are A.\nAnswer."
    ]


def test_personas_need_a_subject_that_reads(tmp_path):
    class Both(cg.Paradigm):
        def trials(self, rng):
            yield cg.Trial(cg.Stimulus(text="a", features=torch.zeros(2)), options=["a", "b"])

    paradigm = Both(personas=["You are A."])
    with pytest.raises(cg.MissingAbilityError, match="persona"):
        run_quietly(paradigm, ScriptedSubject(modalities=("features",)), tmp_path / "run")


def test_participants_are_flagged_identical_only_when_they_are():
    identical = lambda **options: Priming(pairs=PAIRS, **options).config()["participants_identical"]
    assert identical(n_participants=2)
    assert not identical(n_participants=1)
    assert not identical(n_participants=2, personas=["You are A."])

    class Remembered(Priming):
        context = cg.context.cumulative

    assert not Remembered(pairs=PAIRS, n_participants=2).config()["participants_identical"]

    class Random(cg.Paradigm):
        def trials(self, rng):
            yield cg.Trial(cg.Stimulus(text=str(rng.integers(1000))), options=["a", "b"])

    assert not Random(n_participants=2).config()["participants_identical"]


@pytest.mark.parametrize("key", ["step", "history", "participant", "response", "metric", "state_shard"])
def test_condition_key_colliding_with_a_column_fails(key):
    class Colliding(cg.Paradigm):
        def trials(self, rng):
            yield cg.Trial(cg.Stimulus(text="a"), options=["a", "b"], condition={key: 1})

    with pytest.raises(ValueError, match=key):
        Colliding().sequences()


def test_mixed_condition_types_fail_clearly():
    class Mixed(cg.Paradigm):
        def trials(self, rng):
            yield cg.Trial(cg.Stimulus(text="a"), options=["a", "b"], condition={"size": 1})
            yield cg.Trial(cg.Stimulus(text="b"), options=["a", "b"], condition={"size": "large"})

    with pytest.raises(ValueError, match="size"):
        Mixed().sequences()


def test_mixed_item_types_fail_clearly():
    class Mixed(cg.Paradigm):
        def trials(self, rng):
            yield cg.Trial(cg.Stimulus(text="a"), options=["a", "b"], item="first")
            yield cg.Trial(cg.Stimulus(text="b"), options=["a", "b"])

    with pytest.raises(ValueError, match="item"):
        Mixed().sequences()


def test_changing_the_trials_changes_the_run_key(tmp_path):
    key = lambda pairs, name: run_quietly(Priming(pairs=pairs), ScriptedSubject(), tmp_path / name).manifest["key"]
    assert key(PAIRS, "a") == key(PAIRS, "b")
    assert key(PAIRS, "a") != key(PAIRS[:1], "c")


def test_unserialisable_parameters_do_not_break_identity(tmp_path):
    key = lambda name: run_quietly(
        Priming(pairs=PAIRS, scorer=lambda value: value),
        ScriptedSubject(),
        tmp_path / name
    ).manifest["key"]
    assert key("a") == key("b")


def test_parameters_may_not_replace_methods():
    with pytest.raises(ValueError, match="trials"):
        Priming(trials=[])


def test_a_policy_can_be_given_per_instance():
    assert Priming(pairs=PAIRS, context=cg.context.window(2)).config()["context"] == {"keep": 2, "feedback": False}


@pytest.mark.parametrize("options", [{"n_participants": 0}, {"max_new_tokens": 0}, {"personas": []}])
def test_bad_settings_fail(options):
    with pytest.raises(ValueError):
        Priming(pairs=PAIRS, **options)


def test_a_paradigm_with_no_trials_names_itself():
    class Empty(cg.Paradigm):
        pass

    with pytest.raises(NotImplementedError, match="Empty"):
        Empty().sequences()


def test_a_paradigm_whose_trials_are_empty_fails():
    class Nothing(cg.Paradigm):
        def trials(self, rng):
            return []

    with pytest.raises(ValueError, match="no trials"):
        Nothing().sequences()


def test_texts_list_what_a_subject_could_read():
    lines = Priming(pairs=PAIRS, instructions="Answer.", personas=["You are A."]).texts()
    assert "Answer." in lines and "You are A." in lines
    assert any("doctor nurse" in line and "nonword" in line for line in lines)


def test_contextual_run_resumes_exactly_after_a_kill(tmp_path):
    whole = run_quietly(
        Remembering(n_trials=6, block_size=6, n_participants=2),
        Watching(probabilities=(0.6, 0.4)),
        tmp_path / "whole"
    ).to_pandas()
    paradigm = Remembering(n_trials=6, block_size=6, n_participants=2)
    with pytest.raises(RuntimeError, match="scripted failure"):
        run_quietly(paradigm, Watching(probabilities=(0.6, 0.4), fail_after=3), tmp_path / "killed")
    resumed = Watching(probabilities=(0.6, 0.4))
    frame = run_quietly(paradigm, resumed, tmp_path / "killed").to_pandas()
    assert len(resumed.seen) == 6
    columns = ["participant", "trial", "response"]
    pd.testing.assert_frame_equal(frame[columns], whole[columns])
    assert [list(row) for row in frame["history"]] == [list(row) for row in whole["history"]]
    # the context a resumed step sees is what the first run would have shown
    last = resumed.contexts()[-1]
    assert [label for _, label in last] == list(whole["history"].iloc[-1])


def test_context_too_long_for_the_model_suggests_a_window(tiny_model_and_tokenizer, tmp_path):
    model, tokenizer = tiny_model_and_tokenizer

    class Long(cg.Paradigm):
        context = cg.context.cumulative
        ordered = True

        def trials(self, rng):
            for _ in range(60):
                yield cg.Trial(cg.Stimulus(text="small red circle"), options=["A", "B"])

    subject = cg.load(
        model,
        tokenizer=tokenizer,
        device="cpu",
        dtype="float32"
    )
    with pytest.raises(ValueError, match="window"):
        run_quietly(Long(), subject, tmp_path / "run")
    assert len(list((tmp_path / "run" / "steps").glob("*.parquet"))) > 1


def test_iterated_learning_keeps_its_run_key(tmp_path):
    paradigm = cg.paradigms.IteratedLearning(cg.tasks.ShepardCategories(type="VI"), generations=1, chains=1)
    results = run_quietly(paradigm, ScriptedSubject(), tmp_path / "run")
    assert results.manifest["key"] == "54882cbf18b2"


def test_exposes_context():
    assert cg.context.window(3).keep == 3 and cg.context.isolated.keep == 0 and cg.context.cumulative.keep is None
