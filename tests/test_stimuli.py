from __future__ import annotations

import pytest
import torch

from cognarium.stimuli import Stimulus, Trial, as_observation


def test_stimulus_needs_at_least_one_field():
    with pytest.raises(ValueError):
        Stimulus()


def test_modalities_lists_set_fields():
    assert Stimulus(text="a", features=torch.zeros(3)).modalities == {"text", "features"}


def test_trial_normalises_and_checks_options():
    assert Trial(Stimulus(text="a"), options=["x", "y"]).options == ("x", "y")
    with pytest.raises(ValueError):
        Trial(Stimulus(text="a"), options=["x", "x"])
    with pytest.raises(ValueError):
        Trial(Stimulus(text="a"), options=["x"])
    with pytest.raises(ValueError):
        Trial(Stimulus(text="a"), options=["x", "y"], answer="z")


def test_as_observation_accepts_text_tuples():
    observation = as_observation(("red circle", "A"))
    assert observation.stimulus.text == "red circle" and observation.label == "A"


def test_a_free_trial_has_no_options():
    trial = Trial(Stimulus(text="a"))
    assert trial.free and trial.options == ()
    assert not Trial(Stimulus(text="a"), options=["x", "y"]).free


def test_one_option_still_fails():
    with pytest.raises(ValueError, match="or none"):
        Trial(Stimulus(text="a"), options=["x"])


def test_a_free_answer_needs_no_option():
    assert Trial(Stimulus(text="a"), answer="anything").answer == "anything"


def test_a_trial_starts_in_the_first_block():
    assert Trial(Stimulus(text="a"), options=["x", "y"]).block == 0
