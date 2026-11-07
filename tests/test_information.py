from __future__ import annotations

import math

import pytest
import scipy.stats
import torch

from cognarium.metrics.information import entropy, kl_divergence, surprisal


def test_entropy_matches_scipy():
    p = torch.tensor([0.5, 0.25, 0.25], dtype=torch.float64)
    assert entropy(p.log()).item() == pytest.approx(scipy.stats.entropy(p.numpy(), base=2))
    assert entropy(p.log()).item() == pytest.approx(1.5)


def test_kl_matches_scipy_and_is_never_negative():
    p = torch.tensor([0.7, 0.3], dtype=torch.float64)
    q = torch.tensor([0.4, 0.6], dtype=torch.float64)
    expected = scipy.stats.entropy(p.numpy(), q.numpy(), base=2)
    assert kl_divergence(p.log(), q.log()).item() == pytest.approx(expected)
    assert kl_divergence(p.log(), p.log()).item() == pytest.approx(0.0)
    assert kl_divergence(p.log(), q.log()) >= 0


def test_impossible_outcomes_are_handled():
    certain = torch.tensor([0.0, -math.inf], dtype=torch.float64)
    assert entropy(certain).item() == 0.0
    even = torch.log(torch.tensor([0.5, 0.5], dtype=torch.float64))
    assert kl_divergence(certain, even).item() == pytest.approx(1.0)
    assert kl_divergence(even, certain).item() == math.inf


def test_surprisal_in_bits():
    assert surprisal(torch.log(torch.tensor(0.25))).item() == pytest.approx(2.0)


def test_base_changes_the_unit():
    even = torch.log(torch.tensor([0.5, 0.5], dtype=torch.float64))
    assert entropy(even, base=math.e).item() == pytest.approx(math.log(2))
