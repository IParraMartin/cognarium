from __future__ import annotations

import json

import pytest
import torch

import cognarium as cg
from cognarium.errors import MissingAbilityError

FEATURES = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
LABELS = ("wug", "dax")


def run_quietly(paradigm, subject, folder):
    return cg.run(
        paradigm,
        subject,
        out=folder,
        progress=False
    )


def iterated(task, **settings):
    defaults = {
        "generations": 2,
        "chains": 3,
        "n_prev_examples": 2
    }
    return cg.paradigms.IteratedLearning(task=task, **{**defaults, **settings})


def test_examples_and_items_read_as_features_then_label():
    task = cg.tasks.Mapping(FEATURES, LABELS, start=["wug", "wug", "dax", "dax"])
    query = iterated(task, instructions="Name each one.").step(0, None).queries[0]
    lines = query.prompt().splitlines()
    assert lines[0] == "Name each one."
    shown = {"[0, 0]": "wug", "[0, 1]": "wug", "[1, 0]": "dax", "[1, 1]": "dax"}
    for line in lines[1:3]:
        features, label = line.removeprefix("Features: ").split(" -> Label: ")
        assert shown[features] == label
    assert lines[3] == "Features: [0, 0] -> Label:"


def test_a_single_value_per_item_is_written_bare():
    task = cg.tasks.Mapping([3, 7.5], LABELS)
    assert [stimulus.text for stimulus in task.stimuli] == ["3", "7.5"]


def test_words_are_written_as_given_and_reach_only_readers():
    task = cg.tasks.Mapping([["female", "older"], ["male", "younger"]], LABELS)
    assert task.stimuli[0].text == "[female, older]"
    assert task.stimuli[0].features is None
    assert task.modalities == frozenset({"text"})


def test_numeric_features_also_reach_learners_that_read_features():
    task = cg.tasks.Mapping(FEATURES, LABELS)
    assert torch.equal(task.stimuli[3].features, torch.tensor([1.0, 1.0]))
    assert task.modalities == frozenset({"text", "features"})


def test_without_a_start_each_chain_draws_its_own_random_naming():
    task = cg.tasks.Mapping([[value] for value in range(12)], LABELS)
    starts = [task.start(chain, seed=0) for chain in range(3)]
    assert len(set(starts)) == 3
    assert all(set(start) <= set(LABELS) and len(start) == 12 for start in starts)
    assert task.start(1, seed=0) == starts[1]
    assert task.start(1, seed=1) != starts[1]


def test_a_given_start_is_every_chains_first_naming():
    start = ["wug", "dax", "dax", "wug"]
    task = cg.tasks.Mapping(FEATURES, LABELS, start=start)
    step = iterated(task).step(0, None)
    assert all(task.start(chain, seed=5) == tuple(start) for chain in range(3))
    assert [query.trial.answer for query in step.queries[:4]] == start


def test_accuracy_compares_each_chain_with_its_own_start():
    task = cg.tasks.Mapping([[value] for value in range(12)], LABELS)
    step = iterated(task).step(0, None)
    for query, chain in zip(step.queries, step.columns["chain"], strict=True):
        assert query.trial.answer == task.start(chain, seed=0)[query.trial.item]


def test_a_template_changes_the_wording():
    template = cg.PromptTemplate(stimulus="x = {text} -> y =", response="")
    task = cg.tasks.Mapping([1, 2], ["1", "2"], template=template)
    assert iterated(task, n_prev_examples=1).step(0, None).queries[0].prompt().endswith("x = 1 -> y =")


def test_groups_count_the_labels_used_unless_a_rule_is_given():
    task = cg.tasks.Mapping(FEATURES, ("a", "b", "c"))
    assert task.groups == ("1 label", "2 labels", "3 labels")
    assert task.group_of(["a", "a", "b", "a"]) == "2 labels"
    ruled = cg.tasks.Mapping(FEATURES, LABELS, group_by=lambda labels: "same" if len(set(labels)) == 1 else "split")
    assert ruled.group_of(["wug"] * 4) == "same"
    assert ruled.groups == ()


def test_from_function_rounds_the_start_to_the_scale():
    task = cg.tasks.Mapping.from_function(
        [1, 2, 3, 4],
        lambda x: 10 - 2.2 * x,
        scale=range(0, 11)
    )
    assert task.options == tuple(str(value) for value in range(11))
    assert task.start(0, seed=0) == ("8", "6", "3", "1")
    assert [stimulus.text for stimulus in task.stimuli] == ["1", "2", "3", "4"]


@pytest.mark.parametrize(
    "bad",
    [
        {"labels": ["wug"]},
        {"labels": ["wug", "wug"]},
        {"start": ["wug", "dax"]},
        {"start": ["wug", "dax", "dax", "zib"]},
        {"features": []},
        {"features": [[0, 1], [0]]},
        {"sigma": 0},
        {"points": [[0, 0], [0, 1], [1, 0]]},
        {"points": [[0, 0], [0, 1], [1, 0], [1]]},
        {"points": [[0, 0], [0, 1], [1, 0], [1, float("nan")]]},
        {"prior": [1, 1, 1]},
        {"prior": [1, -1, 1, 1]}
    ]
)
def test_invalid_settings_raise(bad):
    settings = {"features": FEATURES, "labels": LABELS, **bad}
    with pytest.raises(ValueError):
        cg.tasks.Mapping(**settings)


def test_bayesian_learners_are_refused_with_a_reason():
    with pytest.raises(MissingAbilityError, match="every possible labelling"):
        iterated(cg.tasks.Mapping(FEATURES, LABELS)).hypothesis_space()


def test_config_is_json_safe_and_tracks_the_items():
    task = cg.tasks.Mapping(FEATURES, LABELS, start=["wug", "dax", "dax", "wug"], prior=[1, 1, 1, 2])
    config = iterated(task).config()
    json.dumps(config)
    assert config["task"]["features"] == FEATURES
    assert config["task"]["labels"] == list(LABELS)
    other = cg.tasks.Mapping([[0, 0], [0, 1], [1, 0], [1, 2]], LABELS)
    assert other.config() != task.config()


def test_points_replace_the_features_in_the_meaning_space_only():
    points = [[0, 0, 0], [0, 0, 10], [0, 10, 0], [10, 0, 0]]
    task = cg.tasks.Mapping(FEATURES, LABELS, points=points)
    assert task.stimuli[1].text == "[0, 1]"
    meanings, sigma, prior = task.meaning_space()
    assert meanings.tolist() == points
    assert sigma == pytest.approx(10.0)
    assert prior.tolist() == [0.25] * 4
    assert task.config()["points"] == points
    words = cg.tasks.Mapping(["red", "green", "blue", "grey"], LABELS, points=points)
    assert words.meaning_space()[0].tolist() == points


def test_an_exemplar_learner_runs_on_numeric_features(tmp_path):
    task = cg.tasks.Mapping(FEATURES, LABELS, start=["wug", "wug", "dax", "dax"])
    results = run_quietly(iterated(task), cg.subjects.ExemplarLearner(), tmp_path / "run")
    frame = results.to_pandas()
    assert len(frame) == 2 * 3 * 4
    assert set(frame["response"]) <= set(LABELS)


def test_a_language_model_runs_it(byte_model_and_tokenizer, tmp_path):
    model, tokenizer = byte_model_and_tokenizer
    subject = cg.load(model, tokenizer=tokenizer, device="cpu")
    results = run_quietly(iterated(cg.tasks.Mapping(FEATURES, LABELS)), subject, tmp_path / "run")
    assert set(results.to_pandas()["group"]) <= {"1 label", "2 labels"}
