from __future__ import annotations

import math

import pytest
import torch

from cognarium.responses import (
    PromptTemplate,
    Query,
    Responses,
    concat_responses,
    sample_responses,
    shape_log_probs
)
from cognarium.stimuli import Observation, Stimulus, Trial


def test_prompt_renders_exact_format():
    template = PromptTemplate(stimulus="Object: {text}", response="Category:")
    query = Query(
        trial=Trial(Stimulus(text="large blue square"), options=["A", "B"]),
        context=(Observation(Stimulus(text="small red circle"), "A"),),
        instructions="Sort each object.",
        template=template
    )
    assert query.prompt() == (
        "Sort each object.\nObject: small red circle\nCategory: A\n"
        "Object: large blue square\nCategory:"
    )


def test_an_empty_cue_lets_the_response_continue_the_stimulus():
    template = PromptTemplate(stimulus="{text}", response="")
    query = Query(
        trial=Trial(Stimulus(text="The boat sailed down the river"), options=["sank", "slowly"]),
        context=(Observation(Stimulus(text="The dog barked"), "loudly"),),
        template=template
    )
    assert query.prompt() == "The dog barked loudly\nThe boat sailed down the river"


def test_prompt_without_text_raises():
    query = Query(trial=Trial(Stimulus(features=torch.zeros(3)), options=["A", "B"]))
    with pytest.raises(ValueError, match="text"):
        query.prompt()


def test_log_probs_normalise_and_keep_impossible_options():
    scores = torch.log(torch.tensor([[3.0, 1.0], [1.0, 0.0]]))
    probs = Responses(scores=scores).log_probs.exp()
    assert probs.flatten().tolist() == pytest.approx([0.75, 0.25, 1.0, 0.0])


def test_concat_pads_to_the_widest():
    narrow = Responses(scores=torch.zeros((1, 2)))
    wide = Responses(scores=torch.zeros((2, 3)))
    joined = concat_responses([narrow, wide])
    assert joined.scores.shape == (3, 3)
    assert joined.scores[0, 2] == -math.inf and joined.choices is None


def test_all_impossible_row_raises():
    with pytest.raises(ValueError, match="query 0"):
        Responses(scores=torch.full((1, 2), -math.inf)).log_probs


def test_temperature_values():
    base = torch.log(torch.tensor([[0.8, 0.2]], dtype=torch.float64))
    colder = shape_log_probs(base, temperature=0.5).exp()[0].tolist()
    warmer = shape_log_probs(base, temperature=2.0).exp()[0].tolist()
    assert colder == pytest.approx([0.941176, 0.058824], abs=1e-5)
    assert warmer == pytest.approx([2 / 3, 1 / 3], abs=1e-5)


def test_top_p_keeps_smallest_covering_set():
    base = torch.log(torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float64))
    assert shape_log_probs(base, top_p=0.6).exp()[0].tolist() == pytest.approx([0.625, 0.375, 0.0])
    assert shape_log_probs(base, top_p=0.5).exp()[0].tolist() == pytest.approx([1.0, 0.0, 0.0])


def test_top_k_one_is_greedy():
    base = torch.log(torch.tensor([[0.3, 0.7]], dtype=torch.float64))
    assert shape_log_probs(base, top_k=1).exp()[0].tolist() == pytest.approx([0.0, 1.0])


def test_shaping_keeps_impossible_options_finite_elsewhere():
    base = torch.tensor([[0.0, -math.inf]], dtype=torch.float64)
    shaped = shape_log_probs(base, temperature=0.5, top_p=0.9)
    assert shaped[0, 0] == 0.0 and shaped[0, 1] == -math.inf


def test_sampling_depends_only_on_each_seed():
    base = torch.log(torch.tensor([[0.5, 0.5]] * 3, dtype=torch.float64))
    seeds = [(0, 1), (0, 2), (0, 3)]
    together = sample_responses(base, seeds)
    apart = torch.cat([sample_responses(base[i:i + 1], [seeds[i]]) for i in range(3)])
    assert together.tolist() == apart.tolist()


def test_sampling_never_picks_impossible_option():
    base = torch.tensor([[-math.inf, 0.0]] * 200, dtype=torch.float64)
    assert sample_responses(base, [(i,) for i in range(200)]).tolist() == [1] * 200
