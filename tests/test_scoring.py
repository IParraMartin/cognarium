from __future__ import annotations

import pytest
import torch

from cognarium.responses import PromptTemplate, Query
from cognarium.stimuli import Stimulus, Trial
from cognarium.subjects.scoring import score_options

CPU = torch.device("cpu")
TOKENS = {
    " a": [2],
    " b": [3],
    " ab": [2, 3],
    " ba": [3, 2],
    " a\n": [2, 1],
    " b\n": [3, 1],
    " ab\n": [2, 3, 1]
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


def after(token):
    return torch.nn.functional.one_hot(torch.tensor(token), 4).float().mul(5.0).log_softmax(-1)


def test_multi_token_option_sums_each_tokens_log_probability():
    prompts = {"x\nResponse:": [1]}
    scores = run_scoring([query(options=("a", "ba"))], last_token_forward, prompts).scores
    assert scores[0, 0].item() == pytest.approx(after(1)[2].item())
    assert scores[0, 1].item() == pytest.approx((after(1)[3] + after(3)[2]).item())


def test_an_option_that_starts_another_is_scored_up_to_the_end_of_an_answer():
    prompts = {"x\nResponse:": [1]}
    scores = run_scoring([query(options=("a", "ab"))], last_token_forward, prompts).scores
    # otherwise "a" would always beat "ab", whose score adds a second log probability
    assert scores[0, 0].item() == pytest.approx((after(1)[2] + after(2)[1]).item())
    assert scores[0, 1].item() == pytest.approx((after(1)[2] + after(2)[3] + after(3)[1]).item())


def test_a_complete_template_scores_every_option_up_to_the_end_of_an_answer():
    prompts = {"x\nResponse:": [1]}
    template = PromptTemplate(response="Response:", complete=True)
    complete = Query(trial=Trial(Stimulus(text="x"), options=("a", "b")), template=template)
    scores = run_scoring([complete, query()], last_token_forward, prompts).scores
    assert scores[0, 0].item() == pytest.approx((after(1)[2] + after(2)[1]).item())
    assert scores[0, 1].item() == pytest.approx((after(1)[3] + after(3)[1]).item())
    assert scores[1, 0].item() == pytest.approx(after(1)[2].item())


def test_a_prompt_scorer_replaces_running_each_option_with_its_prompt():
    seen = []

    def score_after_prompt(prompt, options):
        seen.append((prompt, options))
        return torch.tensor([-1.0, -2.0], dtype=torch.float64)

    prompts = {"x\nResponse:": [1]}
    scores = run_scoring(
        [query(options=("a", "ba"))],
        last_token_forward,
        prompts,
        score_after_prompt=score_after_prompt
    ).scores
    assert seen == [([1], [[2], [3, 2]])]
    assert scores[0].tolist() == [-1.0, -2.0]


def test_a_prompt_scorer_can_decline_and_leave_the_query_to_the_pairs():
    prompts = {"x\nResponse:": [1]}
    declined = run_scoring(
        [query(options=("a", "ba"))],
        last_token_forward,
        prompts,
        score_after_prompt=lambda prompt, options: None
    ).scores
    plain = run_scoring([query(options=("a", "ba"))], last_token_forward, prompts).scores
    assert torch.equal(declined, plain)


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
    queries = [query(options=("a", "ba"), text="x"), query(text="y")]
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
