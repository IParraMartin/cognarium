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
