"""Comparing a model with people: item correlations, noise ceilings and effect sizes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

_METHODS = ("pearson", "spearman")


def item_correlation(
    model: Sequence[float],
    human: Sequence[float],
    *,
    method: str = "pearson",
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0
) -> dict[str, float]:
    """Correlates a model's value for each item with people's.

    The interval is a percentile bootstrap over items, so it reflects how much the
    correlation depends on which items happened to be chosen.

    Args:
        model: One value per item.
        human: One value per item, in the same order.
        method: `pearson`, or `spearman` for the correlation of ranks, with ties given
            their average rank.
        n_resamples: How many resamples of the items the interval uses.
        level: The interval's coverage, such as 0.95.
        seed: Seeds the resampling.

    Returns:
        `correlation`, the interval as `ci_low` and `ci_high`, and `n`, the number of
        items.

    Raises:
        ValueError: If the method is unknown, the two differ in length, there are fewer
            than 3 items, a value isn't finite, either side is constant, or `level` isn't
            between 0 and 1.
    """
    x, y = _paired_values(model, human)
    _check_method(method)
    _check_level(level)
    estimate = _correlations(x[None, :], y[None, :], method)[0]
    if np.isnan(estimate):
        raise ValueError("A correlation is undefined when one side is constant. Every value is the same.")
    picks = np.random.default_rng(seed).integers(len(x), size=(n_resamples, len(x)))
    low, high = _interval(_correlations(x[picks], y[picks], method), level)
    return {
        "correlation": float(estimate),
        "ci_low": low,
        "ci_high": high,
        "n": len(x)
    }


def split_half_ceiling(
    ratings: np.ndarray | pd.DataFrame,
    *,
    n_splits: int = 1000,
    method: str = "pearson",
    level: float = 0.95,
    seed: int = 0
) -> dict[str, float]:
    """Estimates how well any model could correlate with people's item means.

    Split `s` shuffles the participants with `numpy.random.default_rng([seed, s])` and
    halves them, the first half taking the smaller share when the count is odd. The
    item means of the two halves are correlated, over items both halves rated, and the
    Spearman-Brown formula `2r / (1 + r)` corrects each correlation to the full sample.
    The ceiling is the mean of the corrected values.

    Args:
        ratings: One row per participant and one column per item. Missing ratings are
            NaN.
        n_splits: How many random splits to average.
        method: `pearson` or `spearman`.
        level: The share of corrected values between `ci_low` and `ci_high`.
        seed: Seeds the splits.

    Returns:
        `ceiling`, and `ci_low` and `ci_high`, which bound the middle `level` of the
        corrected values across splits. They show how much the ceiling depends on the
        split, and aren't a confidence interval for it.

    Raises:
        ValueError: If there are fewer than two participants or 3 items, the method is
            unknown, `level` isn't between 0 and 1, or no split has two halves that
            vary over at least 3 shared items.
    """
    values = np.asarray(ratings, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 3:
        raise ValueError(
            "ratings need one row per participant and one column per item, with at least two "
            f"participants and 3 items, got shape {values.shape}."
        )
    if np.isinf(values).any():
        raise ValueError("Ratings must be finite or NaN.")
    _check_method(method)
    _check_level(level)
    corrected = []
    half = len(values) // 2
    for split in range(n_splits):
        order = np.random.default_rng([seed, split]).permutation(len(values))
        first, second = _means(values[order[:half]]), _means(values[order[half:]])
        rated = ~np.isnan(first) & ~np.isnan(second)
        if rated.sum() < 3:
            continue
        r = _correlations(first[None, rated], second[None, rated], method)[0]
        if not np.isnan(r):
            corrected.append(2 * r / (1 + r) if r > -1 else -1.0)
    if not corrected:
        raise ValueError(
            "No split gave two halves that vary over at least 3 items both rated, so the ratings are "
            "constant or too sparse for a ceiling."
        )
    low, high = _interval(np.array(corrected), level)
    return {
        "ceiling": float(np.mean(corrected)),
        "ci_low": low,
        "ci_high": high
    }


def effect_comparison(
    model_a: Sequence[float],
    model_b: Sequence[float],
    human_a: Sequence[float],
    human_b: Sequence[float],
    *,
    paired: bool = False,
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0
) -> dict[str, float]:
    """Compares the size of the same effect in a model and in people.

    Each effect is standardised on its own data: Cohen's d with a pooled standard
    deviation, or with `paired`, the mean difference over the standard deviation of the
    differences. A model's values and people's are on different scales, such as
    surprisal and milliseconds, and standardising puts them on one. The interval
    resamples the model's data and people's data separately, each within itself.

    Args:
        model_a: The model's values in the first condition.
        model_b: The model's values in the second condition.
        human_a: People's values in the first condition.
        human_b: People's values in the second condition.
        paired: Whether `a[i]` and `b[i]` belong together, on each side.
        n_resamples: How many resamples the interval uses.
        level: The interval's coverage, such as 0.95.
        seed: Seeds the resampling.

    Returns:
        `model_d`, `human_d`, `difference`, which is `model_d` minus `human_d`, and its
        interval as `ci_low` and `ci_high`.

    Raises:
        ValueError: If a condition has fewer than two values or a value isn't finite,
            paired conditions differ in length, a side has no variance, or `level` isn't
            between 0 and 1.
    """
    _check_level(level)
    model = [_finite(values, name) for values, name in [(model_a, "model_a"), (model_b, "model_b")]]
    human = [_finite(values, name) for values, name in [(human_a, "human_a"), (human_b, "human_b")]]
    for side in (model, human):
        if paired and len(side[0]) != len(side[1]):
            raise ValueError(f"Paired conditions must have the same length, got {len(side[0])} and {len(side[1])}.")
    model_d = _standardised(*(values[None, :] for values in model), paired)[0]
    human_d = _standardised(*(values[None, :] for values in human), paired)[0]
    if np.isnan(model_d) or np.isnan(human_d):
        raise ValueError("Effect sizes are undefined when there is no variance, and one side has none.")
    rng = np.random.default_rng(seed)
    resampled = [_standardised(*_resample(
        side,
        paired,
        n_resamples,
        rng
    ), paired) for side in (model, human)]
    low, high = _interval(resampled[0] - resampled[1], level)
    return {
        "model_d": float(model_d),
        "human_d": float(human_d),
        "difference": float(model_d - human_d),
        "ci_low": low,
        "ci_high": high
    }


def _paired_values(model: Sequence[float], human: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.asarray(model, dtype=np.float64).ravel(), np.asarray(human, dtype=np.float64).ravel()
    if len(x) != len(y):
        raise ValueError(f"model and human need the same number of items, got {len(x)} and {len(y)}.")
    if len(x) < 3:
        raise ValueError(f"A correlation needs at least 3 items, got {len(x)}.")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("Values must be finite to correlate them.")
    return x, y


def _finite(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).ravel()
    if len(array) < 2:
        raise ValueError(f"{name} needs at least two values, got {len(array)}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite.")
    return array


def _check_method(method: str) -> None:
    if method not in _METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from {_METHODS}.")


def _check_level(level: float) -> None:
    if not 0 < level < 1:
        raise ValueError(f"level must be between 0 and 1, got {level}.")


def _correlations(x: np.ndarray, y: np.ndarray, method: str) -> np.ndarray:
    # one correlation per row; a constant row gives NaN, which callers refuse or skip
    if method == "spearman":
        x = pd.DataFrame(x).rank(axis=1).to_numpy()
        y = pd.DataFrame(y).rank(axis=1).to_numpy()
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    scale = np.sqrt((x**2).sum(axis=1) * (y**2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        found = (x * y).sum(axis=1) / scale
    found[scale == 0] = np.nan
    return np.clip(found, -1.0, 1.0)


def _means(values: np.ndarray) -> np.ndarray:
    rated = ~np.isnan(values)
    counts = rated.sum(axis=0)
    totals = np.where(rated, values, 0.0).sum(axis=0)
    return np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)


def _standardised(a: np.ndarray, b: np.ndarray, paired: bool) -> np.ndarray:
    # one effect per row, NaN where there is no variance
    if paired:
        differences = a - b
        spread = differences.std(axis=1, ddof=1)
        centre = differences.mean(axis=1)
    else:
        n_a, n_b = a.shape[1], b.shape[1]
        pooled = ((n_a - 1) * a.var(axis=1, ddof=1) + (n_b - 1) * b.var(axis=1, ddof=1)) / (n_a + n_b - 2)
        spread = np.sqrt(pooled)
        centre = a.mean(axis=1) - b.mean(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        found = centre / spread
    found[spread == 0] = np.nan
    return found


def _resample(
    side: list[np.ndarray],
    paired: bool,
    n_resamples: int,
    rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    a, b = side
    if paired:
        picks = rng.integers(len(a), size=(n_resamples, len(a)))
        return a[picks], b[picks]
    return a[rng.integers(len(a), size=(n_resamples, len(a)))], b[rng.integers(len(b), size=(n_resamples, len(b)))]


def _interval(values: np.ndarray, level: float) -> tuple[float, float]:
    kept = values[~np.isnan(values)]
    if len(kept) == 0:
        raise ValueError("Every resample was constant, so there is no interval. Use more items.")
    tail = (1 - level) / 2
    low, high = np.quantile(kept, [tail, 1 - tail])
    return float(low), float(high)
