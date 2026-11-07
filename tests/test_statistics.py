from __future__ import annotations

import pytest
import scipy.stats
import torch

from cognarium.metrics.statistics import (
    bootstrap_ci,
    cohens_d,
    fdr,
    hedges_g,
    permutation_test
)


def test_effect_size_hand_values():
    x, y = torch.tensor([1.0, 2.0, 3.0]), torch.tensor([2.0, 3.0, 4.0])
    assert cohens_d(x, y) == pytest.approx(-1.0)
    assert hedges_g(x, y) == pytest.approx(-0.8)


def test_zero_variance_raises():
    with pytest.raises(ValueError, match="variance"):
        cohens_d(torch.ones(3), torch.ones(3))


def test_effect_size_needs_two_values_per_group():
    with pytest.raises(ValueError, match="two"):
        cohens_d(torch.ones(1), torch.tensor([1.0, 2.0]))


def test_bootstrap_is_reproducible_and_covers_the_mean():
    values = torch.arange(50.0)
    low, high = bootstrap_ci(values, n_resamples=2000, seed=1)
    assert (low, high) == bootstrap_ci(values, n_resamples=2000, seed=1)
    assert low < values.mean() < high


def test_bootstrap_matches_scipy():
    values = torch.arange(50.0, dtype=torch.float64)
    expected = scipy.stats.bootstrap(
        (values.numpy(),),
        lambda sample, axis: sample.mean(axis=axis),
        n_resamples=20_000,
        method="percentile",
        random_state=0
    ).confidence_interval
    low, high = bootstrap_ci(values, n_resamples=20_000)
    assert low == pytest.approx(expected.low, abs=0.3)
    assert high == pytest.approx(expected.high, abs=0.3)


def test_group_bootstrap_counts_groups_not_rows():
    values = torch.arange(5.0).repeat_interleave(10)
    groups = torch.arange(5).repeat_interleave(10)
    by_row = bootstrap_ci(values, n_resamples=2000)
    by_group = bootstrap_ci(values, groups=groups, n_resamples=2000)
    assert by_group[1] - by_group[0] > by_row[1] - by_row[0]


def test_group_bootstrap_with_unequal_groups_covers_the_mean():
    values = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0])
    groups = torch.tensor([0, 0, 0, 1, 2])
    low, high = bootstrap_ci(values, groups=groups, n_resamples=2000)
    assert low <= values.mean() <= high


def test_custom_statistic():
    values = torch.arange(50.0)
    low, high = bootstrap_ci(values, statistic=torch.median, n_resamples=500)
    assert low < 24.5 < high


def test_custom_statistic_with_groups():
    values = torch.arange(20.0)
    groups = torch.arange(4).repeat_interleave(5)
    low, high = bootstrap_ci(values, groups=groups, statistic=torch.median, n_resamples=500)
    assert low < high


def test_bootstrap_rejects_bad_input():
    with pytest.raises(ValueError, match="finite"):
        bootstrap_ci(torch.tensor([1.0, float("nan")]))
    with pytest.raises(ValueError, match="empty"):
        bootstrap_ci(torch.tensor([]))
    with pytest.raises(ValueError, match="level"):
        bootstrap_ci(torch.tensor([1.0, 2.0]), level=1.5)
    with pytest.raises(ValueError, match="groups"):
        bootstrap_ci(torch.tensor([1.0, 2.0]), groups=torch.tensor([0]))


def test_permutation_test_matches_scipy():
    x, y = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]), torch.tensor([3.0, 4.0, 5.0, 6.0, 9.0])
    expected = scipy.stats.permutation_test(
        (x.numpy(), y.numpy()),
        lambda a, b, axis: a.mean(axis=axis) - b.mean(axis=axis),
        n_resamples=20_000,
        random_state=0
    ).pvalue
    assert permutation_test(x, y, n_resamples=20_000) == pytest.approx(expected, abs=0.02)


def test_identical_groups_are_not_significant():
    x = torch.arange(10.0)
    assert permutation_test(x, x.clone()) > 0.9


def test_paired_permutation_sees_a_consistent_shift():
    x = torch.arange(10.0)
    assert permutation_test(x, x + 1, paired=True) < 0.01


def test_paired_needs_equal_lengths():
    with pytest.raises(ValueError, match="length"):
        permutation_test(torch.arange(3.0), torch.arange(4.0), paired=True)


def test_permutation_p_is_reproducible_and_never_zero():
    x, y = torch.arange(20.0), torch.arange(20.0) + 100
    assert permutation_test(x, y, seed=2) == permutation_test(x, y, seed=2)
    assert permutation_test(x, y) > 0


def test_fdr_matches_scipy():
    p = [0.001, 0.01, 0.03, 0.2, 0.8]
    expected = scipy.stats.false_discovery_control(p).tolist()
    assert fdr(torch.tensor(p, dtype=torch.float64)).tolist() == pytest.approx(expected)


def test_fdr_keeps_the_input_order():
    p = [0.8, 0.001, 0.2, 0.03, 0.01]
    expected = scipy.stats.false_discovery_control(p).tolist()
    assert fdr(torch.tensor(p, dtype=torch.float64)).tolist() == pytest.approx(expected)


def test_fdr_rejects_values_outside_zero_and_one():
    with pytest.raises(ValueError, match="between"):
        fdr(torch.tensor([1.5]))
