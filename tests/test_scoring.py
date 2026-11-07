from __future__ import annotations

import pytest
import torch

from cognarium.responses import Query
from cognarium.stimuli import Stimulus, Trial
from cognarium.subjects.scoring import score_options

CPU = torch.device("cpu")
TOKENS = {
    " a": [2],
    " b": [3],
    " ab": [2, 3]
}


def query(options=("a", "b"), text="x"):
    return Query(trial=Trial(Stimulus(text=text), options=options))


def run_scoring(
    queries,
    forward,
    prompts,
    **kwargs
):
    return score_options(
        queries,
        forward=forward,
        encode_prompt=lambda text: prompts[text],
        encode_option=lambda text: TOKENS[text],
        pad_token_id=0,
        device=CPU,
        **kwargs
    )


def fixed_forward(ids, mask):
    return torch.tensor([1.0, 2.0, 3.0, 4.0]).expand(ids.shape[0], ids.shape[1], -1)


def last_token_forward(ids, mask):
    return torch.nn.functional.one_hot(ids, 4).float() * 5.0


def test_scores_are_log_softmax_at_the_option_tokens():
    scores = run_scoring([query()], fixed_forward, {"x\nResponse:": [1]}).scores
    expected = torch.tensor([1.0, 2.0, 3.0, 4.0]).log_softmax(-1)
    assert scores[0].tolist() == pytest.approx(expected[2:4].tolist())


def test_last_real_token_supplies_the_logits_after_padding():
    prompts = {"x\nResponse:": [1], "y\nResponse:": [1, 2, 3]}
    queries = [query(text="x"), query(text="y")]
    scores = run_scoring(queries, last_token_forward, prompts).scores
    ones = torch.nn.functional.one_hot(torch.tensor(1), 4).float().mul(5.0).log_softmax(-1)
    threes = torch.nn.functional.one_hot(torch.tensor(3), 4).float().mul(5.0).log_softmax(-1)
    assert scores[0].tolist() == pytest.approx(ones[2:4].tolist())
    assert scores[1].tolist() == pytest.approx(threes[2:4].tolist())


def test_multi_token_option_sums_each_tokens_log_probability():
    prompts = {"x\nResponse:": [1]}
    scores = run_scoring([query(options=("a", "ab"))], last_token_forward, prompts).scores
    after_prompt = torch.nn.functional.one_hot(torch.tensor(1), 4).float().mul(5.0).log_softmax(-1)
    after_a = torch.nn.functional.one_hot(torch.tensor(2), 4).float().mul(5.0).log_softmax(-1)
    assert scores[0, 0].item() == pytest.approx(after_prompt[2].item())
    assert scores[0, 1].item() == pytest.approx((after_prompt[2] + after_a[3]).item())


def test_capture_logits_returns_the_last_prompt_position():
    prompts = {"x\nResponse:": [1, 2]}
    logits = run_scoring(
        [query()],
        last_token_forward,
        prompts,
        capture_logits=True
    ).logits
    assert logits[0].tolist() == [0.0, 0.0, 5.0, 0.0]


def test_mixed_option_lengths_keep_query_order():
    prompts = {"x\nResponse:": [1], "y\nResponse:": [2]}
    queries = [query(options=("a", "ab"), text="x"), query(text="y")]
    scores = run_scoring(
        queries,
        last_token_forward,
        prompts,
        batch_size=1
    ).scores
    single = torch.nn.functional.one_hot(torch.tensor(2), 4).float().mul(5.0).log_softmax(-1)
    assert scores[1].tolist() == pytest.approx(single[2:4].tolist())


def test_empty_prompt_raises():
    with pytest.raises(ValueError, match="no tokens"):
        run_scoring([query()], fixed_forward, {"x\nResponse:": []})
