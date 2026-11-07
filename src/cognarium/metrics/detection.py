"""Signal detection measures, computed from the four outcome counts."""

from __future__ import annotations

import torch


def d_prime(
    hits: int,
    misses: int,
    false_alarms: int,
    correct_rejections: int,
    *,
    correction: str | None = "loglinear"
) -> float:
    """Computes sensitivity, the separation of signal and noise in z units.

    Args:
        hits: Signal trials answered as signal.
        misses: Signal trials answered as noise.
        false_alarms: Noise trials answered as signal.
        correct_rejections: Noise trials answered as noise.
        correction: `"loglinear"` adds 0.5 to each count and 1 to each total, which
            keeps rates of 0 and 1 finite. None uses the raw rates.

    Returns:
        d'. Zero means no sensitivity.

    Raises:
        ValueError: If a count is negative, signal or noise has no trials, or a
            rate of 0 or 1 is used with no correction.
    """
    hit, false_alarm = _rates(
        hits,
        misses,
        false_alarms,
        correct_rejections,
        correction
    )
    return _z(hit, correction) - _z(false_alarm, correction)


def criterion(
    hits: int,
    misses: int,
    false_alarms: int,
    correct_rejections: int,
    *,
    correction: str | None = "loglinear"
) -> float:
    """Computes response bias.

    Args:
        hits: Signal trials answered as signal.
        misses: Signal trials answered as noise.
        false_alarms: Noise trials answered as signal.
        correct_rejections: Noise trials answered as noise.
        correction: `"loglinear"` adds 0.5 to each count and 1 to each total, which
            keeps rates of 0 and 1 finite. None uses the raw rates.

    Returns:
        c. Negative values lean towards answering signal, positive towards noise.

    Raises:
        ValueError: If a count is negative, signal or noise has no trials, or a
            rate of 0 or 1 is used with no correction.
    """
    hit, false_alarm = _rates(
        hits,
        misses,
        false_alarms,
        correct_rejections,
        correction
    )
    return -(_z(hit, correction) + _z(false_alarm, correction)) / 2


def a_prime(
    hits: int,
    misses: int,
    false_alarms: int,
    correct_rejections: int,
    *,
    correction: str | None = "loglinear"
) -> float:
    """Computes the non-parametric sensitivity of Pollack and Norris (1964).

    Args:
        hits: Signal trials answered as signal.
        misses: Signal trials answered as noise.
        false_alarms: Noise trials answered as signal.
        correct_rejections: Noise trials answered as noise.
        correction: `"loglinear"` adds 0.5 to each count and 1 to each total. None
            uses the raw rates, which A' can handle.

    Returns:
        A', where 0.5 means no sensitivity and 1 means perfect.

    Raises:
        ValueError: If a count is negative, or signal or noise has no trials.
    """
    hit, false_alarm = _rates(
        hits,
        misses,
        false_alarms,
        correct_rejections,
        correction
    )
    difference = hit - false_alarm
    if difference >= 0:
        denominator = 4 * hit * (1 - false_alarm)
        return 0.5 if denominator == 0 else 0.5 + difference * (1 + difference) / denominator
    denominator = 4 * false_alarm * (1 - hit)
    return 0.5 - difference * (difference - 1) / denominator


def _rates(
    hits: int,
    misses: int,
    false_alarms: int,
    correct_rejections: int,
    correction: str | None
) -> tuple[float, float]:
    if correction not in ("loglinear", None):
        raise ValueError(f"Unknown correction '{correction}'. Use 'loglinear' or None.")
    if min(hits, misses, false_alarms, correct_rejections) < 0:
        raise ValueError("Counts must not be negative.")
    if hits + misses == 0 or false_alarms + correct_rejections == 0:
        raise ValueError("Signal detection needs both signal trials and noise trials.")
    shift = 0.5 if correction == "loglinear" else 0.0
    hit = (hits + shift) / (hits + misses + 2 * shift)
    false_alarm = (false_alarms + shift) / (false_alarms + correct_rejections + 2 * shift)
    return hit, false_alarm


def _z(rate: float, correction: str | None) -> float:
    if rate in (0.0, 1.0):
        raise ValueError(f"A rate of {rate} has no z score. Use correction='loglinear', not {correction}.")
    return torch.special.ndtri(torch.tensor(rate, dtype=torch.float64)).item()
