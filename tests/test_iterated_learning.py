from __future__ import annotations

import json

import pandas as pd
import pytest

from cognarium.paradigms.learning import IteratedLearning
from cognarium.stimuli import Observation
from cognarium.tasks.shepard import ShepardCategories


def paradigm(**kwargs):
    defaults = {
        "task": ShepardCategories(type="I"),
        "generations": 3,
        "chains": 2
    }
    return IteratedLearning(**{**defaults, **kwargs})


def responses_frame(step, label):
    return pd.DataFrame({
        "chain": step.columns["chain"],
        "item": [query.trial.item for query in step.queries],
        "response": [label] * len(step.queries)
    })


def test_first_generation_has_one_query_per_chain_and_item():
    step = paradigm().step(0, None)
    assert len(step.queries) == 16
    assert all(len(query.context) == 4 for query in step.queries)
    assert set(step.columns["generation"]) == {0}
    assert sum(step.columns["seen"]) == 8


def test_noise_free_context_matches_seed_labels():
    task = ShepardCategories(type="I")
    step = paradigm(task=task).step(0, None)
    labels = {stimulus.text: label for stimulus, label in zip(task.stimuli, task.seed_labels)}
    for query in step.queries:
        assert all(labels[obs.stimulus.text] == obs.label for obs in query.context)


def test_later_generation_learns_from_previous_responses():
    made = paradigm()
    step = made.step(1, responses_frame(made.step(0, None), "B"))
    assert all(obs.label == "B" for query in step.queries for obs in query.context)
    assert set(step.columns["parent_label"]) == {"B"}


def test_noise_flips_transmitted_labels_at_the_set_rate():
    made = paradigm(chains=2000, noise=0.25)
    step = made.step(0, None)
    pairs = [
        (transmitted, parent)
        for seen, transmitted, parent in zip(
            step.columns["seen"],
            step.columns["transmitted_label"],
            step.columns["parent_label"]
        )
        if seen
    ]
    flipped = sum(transmitted != parent for transmitted, parent in pairs) / len(pairs)
    assert flipped == pytest.approx(0.25, abs=0.02)


def test_examples_come_first_in_every_generation():
    task = ShepardCategories(type="I")
    example = Observation(task.stimuli[0], "B")
    made = paradigm(task=task, examples=[example])
    first = made.step(0, None)
    second = made.step(1, responses_frame(first, "A"))
    assert first.queries[0].context[0] is made.settings.examples[0]
    assert second.queries[0].context[0] is made.settings.examples[0]
    assert len(second.queries[0].context) == 5


def test_same_seed_repeats_and_chains_differ():
    one, two = paradigm(chains=20).step(0, None), paradigm(chains=20).step(0, None)
    assert one.columns["seen"] == two.columns["seen"]
    subsets = {tuple(obs.stimulus.text for obs in q.context) for q in one.queries}
    assert len(subsets) > 1


def test_different_seeds_differ():
    one, two = paradigm(chains=20, seed=1).step(0, None), paradigm(chains=20, seed=2).step(0, None)
    assert one.columns["seen"] != two.columns["seen"]


def test_step_after_last_generation_is_none():
    assert paradigm(generations=1).step(1, pd.DataFrame()) is None


def test_later_step_without_previous_raises():
    with pytest.raises(ValueError, match="previous"):
        paradigm().step(1, None)


def test_annotate_adds_each_chains_group():
    task = ShepardCategories(type="I")
    frame = pd.DataFrame({
        "chain": [0] * 8,
        "item": list(range(8)),
        "response": list(task.seed_labels)
    })
    assert set(paradigm(task=task).annotate(frame)["group"]) == {"I"}


@pytest.mark.parametrize(
    "bad",
    [
        {"chains": 0},
        {"generations": 0},
        {"n_prev_examples": 9},
        {"n_prev_examples": 0},
        {"noise": 1.0}
    ]
)
def test_invalid_settings_raise(bad):
    with pytest.raises(ValueError):
        paradigm(**bad)


def test_each_generation_sees_n_prev_examples_from_the_last_one():
    step = paradigm(n_prev_examples=6).step(0, None)
    assert all(len(query.context) == 6 for query in step.queries)
    assert paradigm(n_prev_examples=6).config()["n_prev_examples"] == 6


def test_example_label_must_be_an_option():
    with pytest.raises(ValueError, match="Z"):
        paradigm(examples=[("red circle", "Z")])


def test_config_is_json_safe():
    config = paradigm(examples=[("red circle", "A")], seed=3).config()
    json.dumps(config)
    assert config["seed"] == 3 and config["task"]["type"] == "I"


def test_texts_cover_every_stimulus_and_option():
    text = " ".join(paradigm(instructions="Sort them.", examples=[("odd thing", "A")]).texts())
    for word in [
        "Sort",
        "them.",
        "odd",
        "thing",
        "small",
        "square",
        "Object:",
        "Category:",
        "A",
        "B"
    ]:
        assert word in text.split()


def test_missing_parent_labels_raise_rather_than_reach_a_prompt():
    made = IteratedLearning(task=ShepardCategories(type="I"), chains=3)
    previous = pd.DataFrame(
        {
            "chain": [0] * 8,
            "item": list(range(8)),
            "response": ["A"] * 8
        }
    )
    with pytest.raises(ValueError, match="chain 1"):
        made.step(1, previous)
