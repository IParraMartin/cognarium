from __future__ import annotations

from collections import Counter

import pytest

from cognarium.tasks.shepard import SHEPARD_TYPES, ShepardCategories


def test_group_counts_match_the_literature():
    task = ShepardCategories()
    assert Counter(task.hypothesis_space().groups) == {
        "I": 6,
        "II": 6,
        "III": 24,
        "IV": 8,
        "V": 24,
        "VI": 2,
        "unbalanced": 186
    }


@pytest.mark.parametrize("name", SHEPARD_TYPES)
def test_each_type_seeds_its_own_group(name):
    task = ShepardCategories(type=name)
    assert task.group_of(task.seed_labels) == name


def test_rendering_gives_text_and_features():
    task = ShepardCategories()
    assert task.stimuli[0].text == "small red circle"
    assert task.stimuli[7].text == "large blue square"
    assert task.stimuli[1].features.tolist() == [0.0, 0.0, 1.0]


def test_hypothesis_bit_order():
    hypotheses = ShepardCategories().hypothesis_space().hypotheses
    assert hypotheses.shape == (256, 8)
    assert hypotheses[1].tolist() == [1, 0, 0, 0, 0, 0, 0, 0]


def test_unbalanced_labelling_is_named():
    assert ShepardCategories().group_of(["A"] * 8) == "unbalanced"


def test_type_prior_splits_weight_evenly_within_groups():
    task = ShepardCategories()
    prior = task.type_prior({"I": 0.75, "VI": 0.25})
    groups = task.hypothesis_space().groups
    assert prior.sum().item() == pytest.approx(1.0)
    assert prior[groups.index("VI")].item() == pytest.approx(0.125)
    assert prior[groups.index("I")].item() == pytest.approx(0.125)
    assert prior[groups.index("II")].item() == 0.0


def test_unknown_type_and_group_raise():
    with pytest.raises(ValueError, match="VII"):
        ShepardCategories(type="VII")
    with pytest.raises(ValueError, match="nope"):
        ShepardCategories().type_prior({"nope": 1.0})
