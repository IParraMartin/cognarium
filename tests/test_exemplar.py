from __future__ import annotations

import math

import pytest
import torch

from cognarium.paradigms.learning import IteratedLearning
from cognarium.responses import Query, sample_responses, shape_log_probs
from cognarium.stimuli import Observation, Trial
from cognarium.subjects.exemplar import ExemplarLearner
from cognarium.tasks.shepard import ShepardCategories


def query_for(task, item, context):
    return Query(trial=Trial(task.stimuli[item], options=task.options), context=tuple(context))


def all_items(task):
    return [Observation(s, label) for s, label in zip(task.stimuli, task.seed_labels)]


@pytest.mark.parametrize(
    ("kind", "expected"),
    [("I", 0.660756), ("VI", 0.516617)]
)
def test_matches_hand_computed_values(kind, expected):
    task = ShepardCategories(type=kind)
    context = all_items(task)
    queries = [query_for(task, i, context) for i in range(8)]
    probs = ExemplarLearner().respond(queries).log_probs.exp()
    for i, label in enumerate(task.seed_labels):
        assert probs[i, task.options.index(label)].item() == pytest.approx(expected, abs=1e-5)


def test_c_zero_gives_category_frequencies():
    task = ShepardCategories(type="I")
    context = [Observation(task.stimuli[i], label) for i, label in enumerate("AAAB")]
    probs = ExemplarLearner(c=0.0).respond([query_for(task, 7, context)]).log_probs.exp()
    assert probs[0].tolist() == pytest.approx([0.75, 0.25])


def test_gamma_sharpens():
    task = ShepardCategories(type="I")
    context = [Observation(task.stimuli[i], label) for i, label in enumerate("AAAB")]
    probs = ExemplarLearner(c=0.0, gamma=2.0).respond([query_for(task, 7, context)]).log_probs.exp()
    assert probs[0].tolist() == pytest.approx([0.9, 0.1])


def test_attention_is_normalised_and_used():
    task = ShepardCategories(type="I")
    learner = ExemplarLearner(attention=[10, 0, 0])
    probs = learner.respond([query_for(task, 0, all_items(task))]).log_probs.exp()
    assert probs[0, 0].item() == pytest.approx(4 / (4 + 4 * math.exp(-2.0)), abs=1e-5)


def test_type_one_is_easier_than_type_six():
    scores = {}
    for kind in ("I", "VI"):
        task = ShepardCategories(type=kind)
        context = all_items(task)
        queries = [query_for(task, i, context) for i in range(8)]
        probs = ExemplarLearner().respond(queries).log_probs.exp()
        indices = torch.tensor([task.options.index(label) for label in task.seed_labels])
        scores[kind] = probs[torch.arange(8), indices].mean().item()
    assert scores["I"] > scores["VI"]


def test_fixated_context_gives_zero_probability_not_nan():
    task = ShepardCategories(type="I")
    context = [Observation(task.stimuli[i], "A") for i in range(4)]
    log_probs = ExemplarLearner().respond([query_for(task, 7, context)]).log_probs
    assert log_probs[0, 0] == 0.0 and log_probs[0, 1] == -math.inf
    shaped = shape_log_probs(log_probs, temperature=0.7, top_p=0.9)
    assert sample_responses(shaped, [(0,)]).tolist() == [0]


def test_guess_keeps_every_option_possible():
    task = ShepardCategories(type="I")
    context = [Observation(task.stimuli[i], "A") for i in range(4)]
    probs = ExemplarLearner(guess=0.1).respond([query_for(task, 7, context)]).log_probs.exp()
    assert probs[0, 1].item() == pytest.approx(0.05)


def test_empty_context_is_uniform():
    task = ShepardCategories(type="I")
    probs = ExemplarLearner().respond([query_for(task, 0, [])]).log_probs.exp()
    assert probs[0].tolist() == pytest.approx([0.5, 0.5])


def test_very_large_c_does_not_underflow_to_impossible():
    task = ShepardCategories(type="I")
    context = [Observation(task.stimuli[0], "A"), Observation(task.stimuli[7], "B")]
    probs = ExemplarLearner(c=1e6).respond([query_for(task, 1, context)]).log_probs.exp()
    assert probs[0].tolist() == pytest.approx([1.0, 0.0])


def test_label_outside_the_options_raises():
    task = ShepardCategories(type="I")
    context = [Observation(task.stimuli[0], "Z")]
    with pytest.raises(ValueError, match="Z"):
        ExemplarLearner().respond([query_for(task, 0, context)])


def test_text_only_example_fails_at_bind():
    made = IteratedLearning(task=ShepardCategories(type="I"), examples=[("a thing", "A")])
    with pytest.raises(ValueError, match="features"):
        ExemplarLearner().bind(made, "exemplar")


def test_config_changes_with_the_parameters():
    assert ExemplarLearner(c=1.0).config() != ExemplarLearner(c=2.0).config()


def test_runs_on_every_device(device):
    task = ShepardCategories(type="I")
    scores = ExemplarLearner(device=device).respond([query_for(task, 0, all_items(task))]).scores
    assert scores.device.type == "cpu"
