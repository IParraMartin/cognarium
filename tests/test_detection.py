from __future__ import annotations

import math

import pytest
import scipy.stats

from cognarium.metrics.detection import a_prime, criterion, d_prime


def test_d_prime_matches_the_z_formula():
    hit, false_alarm = 40.5 / 51, 10.5 / 51
    expected = scipy.stats.norm.ppf(hit) - scipy.stats.norm.ppf(false_alarm)
    assert d_prime(40, 10, 10, 40) == pytest.approx(expected)
    assert criterion(40, 10, 10, 40) == pytest.approx(0.0)


def test_criterion_sign_shows_the_bias():
    assert criterion(45, 5, 25, 25) < 0
    assert criterion(25, 25, 5, 45) > 0


def test_perfect_rates_stay_finite_with_the_correction():
    assert math.isfinite(d_prime(50, 0, 0, 50))
    assert math.isfinite(criterion(50, 0, 0, 50))


@pytest.mark.parametrize("metric", [d_prime, criterion])
def test_perfect_rates_raise_without_the_correction(metric):
    with pytest.raises(ValueError, match="correction"):
        metric(50, 0, 0, 50, correction=None)


def test_a_prime_hand_value():
    assert a_prime(8, 2, 2, 8, correction=None) == pytest.approx(0.875)
    assert a_prime(2, 8, 8, 2, correction=None) == pytest.approx(0.125)
    assert a_prime(10, 0, 0, 10, correction=None) == pytest.approx(1.0)


def test_a_prime_with_no_discrimination_is_a_half():
    assert a_prime(0, 10, 0, 10, correction=None) == pytest.approx(0.5)
    assert a_prime(10, 0, 10, 0, correction=None) == pytest.approx(0.5)


@pytest.mark.parametrize("metric", [d_prime, criterion, a_prime])
def test_empty_signal_or_noise_trials_raise(metric):
    with pytest.raises(ValueError, match="trials"):
        metric(0, 0, 3, 3)
    with pytest.raises(ValueError, match="trials"):
        metric(3, 3, 0, 0)


def test_negative_counts_raise():
    with pytest.raises(ValueError, match="negative"):
        d_prime(-1, 3, 3, 3)


def test_unknown_correction_raises():
    with pytest.raises(ValueError, match="loglinear"):
        d_prime(3, 3, 3, 3, correction="nope")
