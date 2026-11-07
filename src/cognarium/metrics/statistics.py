"""Effect sizes, resampling tests, contrasts and multiple comparison correction."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch

# a smaller gap is rounding, so it can't count as an effect
_ROUNDING = 1e-12


def cohens_d(x: torch.Tensor, y: torch.Tensor) -> float:
    """Computes the standardised difference between two means.

    Args:
        x: One group's values.
        y: The other group's values.

    Returns:
        The difference of means, x minus y, in pooled standard deviations.

    Raises:
        ValueError: If a group has fewer than two values, or neither group varies.
    """
    x, y = x.double().cpu(), y.double().cpu()
    if len(x) < 2 or len(y) < 2:
        raise ValueError("Effect sizes need at least two values per group.")
    pooled = ((len(x) - 1) * x.var() + (len(y) - 1) * y.var()) / (len(x) + len(y) - 2)
    if pooled == 0:
        raise ValueError("Effect sizes are undefined when there is no variance.")
    return ((x.mean() - y.mean()) / pooled.sqrt()).item()


def hedges_g(x: torch.Tensor, y: torch.Tensor) -> float:
    """Computes Cohen's d with the small sample correction.

    Args:
        x: One group's values.
        y: The other group's values.

    Returns:
        The corrected difference of means, x minus y.

    Raises:
        ValueError: If a group has fewer than two values, or neither group varies.
    """
    degrees = len(x) + len(y) - 2
    return cohens_d(x, y) * (1 - 3 / (4 * degrees - 1))


def bootstrap_ci(
    values: torch.Tensor,
    *,
    groups: torch.Tensor | None = None,
    statistic: Callable[[torch.Tensor], torch.Tensor] | None = None,
    n_resamples: int = 10_000,
    level: float = 0.95,
    seed: int = 0
) -> tuple[float, float]:
    """Computes a percentile bootstrap confidence interval.

    Pass `groups` when rows aren't independent, such as several rows per chain, item
    or participant. Resampling whole groups then gives an honest interval, where
    resampling rows would give one that's too narrow.

    Args:
        values: The values, shape (n,).
        groups: Which group each value belongs to, shape (n,). Whole groups are
            resampled.
        statistic: Maps a 1-D tensor to a scalar tensor. Defaults to the mean.
        n_resamples: How many resamples to draw.
        level: The interval's coverage, such as 0.95.
        seed: Seeds the resampling.

    Returns:
        The lower and upper bounds.

    Raises:
        ValueError: If `values` is empty or not finite, `groups` differs in length,
            or `level` isn't between 0 and 1.
    """
    values = values.double().cpu()
    if len(values) == 0:
        raise ValueError("Can't bootstrap empty values.")
    if not values.isfinite().all():
        raise ValueError("Values must be finite to bootstrap them.")
    if not 0 < level < 1:
        raise ValueError(f"level must be between 0 and 1, got {level}.")
    if groups is not None and len(groups) != len(values):
        raise ValueError(f"groups and values must have the same length, got {len(groups)} and {len(values)}.")
    generator = torch.Generator().manual_seed(seed)
    if groups is None:
        groups = torch.arange(len(values))
    members = [values[groups.cpu() == group] for group in groups.cpu().unique()]
    totals = torch.stack([part.sum() for part in members])
    counts = torch.tensor([len(part) for part in members], dtype=torch.float64)
    parts = []
    for size in _chunks(n_resamples):
        picks = torch.randint(len(members), (size, len(members)), generator=generator)
        if statistic is None:
            parts.append(totals[picks].sum(dim=1) / counts[picks].sum(dim=1))
        else:
            parts.extend(statistic(torch.cat([members[i] for i in row])).reshape(1) for row in picks.tolist())
    estimates = torch.cat(parts)
    tail = (1 - level) / 2
    bounds = torch.quantile(estimates.double(), torch.tensor([tail, 1 - tail], dtype=torch.float64))
    return bounds[0].item(), bounds[1].item()


def permutation_test(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    paired: bool = False,
    n_resamples: int = 10_000,
    seed: int = 0
) -> float:
    """Tests whether two groups differ in their means, by shuffling.

    Args:
        x: One group's values.
        y: The other group's values.
        paired: Whether `x[i]` and `y[i]` belong together. Pairs then swap sides at
            random, instead of all values being shuffled.
        n_resamples: How many shuffles to draw.
        seed: Seeds the shuffling.

    Returns:
        The two-sided p value. It counts the observed split as one of the shuffles,
        so it is never exactly 0.

    Raises:
        ValueError: If `paired` is set and the lengths differ.
    """
    x, y = x.double().cpu(), y.double().cpu()
    generator = torch.Generator().manual_seed(seed)
    observed = (x.mean() - y.mean()).abs()
    if paired and len(x) != len(y):
        raise ValueError(f"Paired values must have the same length, got {len(x)} and {len(y)}.")
    pooled = torch.cat([x, y])
    extreme = 0
    for size in _chunks(n_resamples):
        if paired:
            signs = torch.randint(
                2,
                (size, len(x)),
                generator=generator,
                dtype=torch.float64
            ) * 2 - 1
            shuffled = (signs * (x - y)).mean(dim=1).abs()
        else:
            order = torch.rand(size, len(pooled), generator=generator).argsort(dim=1)
            ordered = pooled[order]
            shuffled = (ordered[:, : len(x)].mean(dim=1) - ordered[:, len(x) :].mean(dim=1)).abs()
        # the tolerance stops rounding from hiding a shuffle that exactly ties
        extreme += (shuffled >= observed - 1e-12).sum().item()
    return (extreme + 1) / (n_resamples + 1)


def fdr(p_values: torch.Tensor) -> torch.Tensor:
    """Adjusts p values with the Benjamini-Hochberg procedure.

    Args:
        p_values: The p values, in any order.

    Returns:
        Adjusted p values in the input's order. Comparing them with a threshold
        controls the false discovery rate at that threshold.

    Raises:
        ValueError: If a p value isn't between 0 and 1.
    """
    values = p_values.double().cpu()
    if not ((values >= 0) & (values <= 1)).all():
        raise ValueError("p values must be between 0 and 1.")
    ordered, order = values.sort()
    ranks = torch.arange(1, len(values) + 1, dtype=torch.float64)
    scaled = ordered * len(values) / ranks
    # each value takes the smallest adjusted value at or above its rank
    adjusted = scaled.flip(0).cummin(dim=0).values.flip(0).clamp_max(1.0)
    return torch.empty_like(values).scatter_(0, order, adjusted)


def contrast_table(
    frame: pd.DataFrame,
    value: str,
    *,
    by: str | None = None,
    levels: Sequence[Any] | None = None,
    against: float | None = None,
    group: str = "item",
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0
) -> pd.DataFrame:
    """Compares two levels of a condition, or one level with a fixed value.

    Rows are first averaged within each group, because several rows of one item aren't
    independent. When every group appears in both levels the comparison is paired, so
    differences between items don't widen the interval. Otherwise the two levels are
    resampled separately.

    Args:
        frame: The table holding the values.
        value: The column to compare, which must be numeric.
        by: The column naming the levels.
        levels: The levels to compare, as `(level_a, level_b)`. The difference is
            `level_a` minus `level_b`. With exactly two levels present this defaults to
            them in reverse sorted order, so True comes before False. With `against`,
            give one level to test only its rows.
        against: Compares with this value rather than a second level.
        group: The column naming the groups to resample, such as `item` or `participant`.
        n_resamples: How many resamples the interval and the p value each use.
        level: The interval's coverage, such as 0.95.
        seed: Seeds the resampling.

    Returns:
        One row, with the columns `by`, `level_a`, `level_b`, `mean_a`, `mean_b`,
        `difference`, `ci_low`, `ci_high`, `p_value` and `detected`, which says whether
        the interval excludes 0. The interval is a percentile bootstrap and the p value
        comes from a permutation test. With `against`, `level_b` is `"chance"` for 0.5 and
        otherwise the value as text.

    Raises:
        ValueError: If a column is missing, there is nothing to compare with, the levels
            are ambiguous or have no rows, a level has fewer than two groups, or a value
            isn't finite.
    """
    if by is None and against is None:
        raise ValueError("Give `by` or `against`, so there is something to compare with.")
    for column in (value, group, by):
        if column is not None and column not in frame.columns:
            raise ValueError(f"There is no column '{column}'. The table has {list(frame.columns)}.")
    if not np.isfinite(frame[value].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"The values of '{value}' must be finite to compare them.")
    if not 0 < level < 1:
        raise ValueError(f"level must be between 0 and 1, got {level}.")
    if against is not None:
        return _against_value(
            frame,
            value,
            by,
            levels,
            against,
            group,
            n_resamples,
            level,
            seed
        )
    level_a, level_b = _two_levels(frame, by, levels)
    means_a = _group_means(
        frame,
        value,
        by,
        level_a,
        group
    )
    means_b = _group_means(
        frame,
        value,
        by,
        level_b,
        group
    )
    paired = set(means_a.index) == set(means_b.index)
    if paired:
        means_b = means_b.loc[means_a.index]
    x = torch.tensor(means_a.to_numpy(), dtype=torch.float64)
    y = torch.tensor(means_b.to_numpy(), dtype=torch.float64)
    if paired:
        low, high = bootstrap_ci(
            x - y,
            n_resamples=n_resamples,
            level=level,
            seed=seed
        )
    else:
        low, high = _unpaired_ci(
            x,
            y,
            n_resamples,
            level,
            seed
        )
    p_value = permutation_test(
        x,
        y,
        paired=paired,
        n_resamples=n_resamples,
        seed=seed
    )
    return _contrast_row(
        by,
        level_a,
        level_b,
        x.mean().item(),
        y.mean().item(),
        low,
        high,
        p_value
    )


def _against_value(
    frame: pd.DataFrame,
    value: str,
    by: str | None,
    levels: Sequence[Any] | None,
    against: float,
    group: str,
    n_resamples: int,
    level: float,
    seed: int
) -> pd.DataFrame:
    if by is None:
        means = _group_means(
            frame,
            value,
            None,
            None,
            group
        )
        name = "all"
    else:
        if levels is None or len(levels) != 1:
            raise ValueError(f"To compare '{by}' with a value, give one level, such as levels=('some level',).")
        name = levels[0]
        means = _group_means(
            frame,
            value,
            by,
            name,
            group
        )
    x = torch.tensor(means.to_numpy(), dtype=torch.float64)
    low, high = bootstrap_ci(
        x,
        n_resamples=n_resamples,
        level=level,
        seed=seed
    )
    p_value = permutation_test(
        x,
        torch.full_like(x, against),
        paired=True,
        n_resamples=n_resamples,
        seed=seed
    )
    return _contrast_row(
        by,
        name,
        "chance" if against == 0.5 else str(against),
        x.mean().item(),
        against,
        low - against,
        high - against,
        p_value
    )


def _two_levels(
    frame: pd.DataFrame,
    by: str,
    levels: Sequence[Any] | None
) -> tuple[Any, Any]:
    present = frame[by].dropna().unique().tolist()
    if levels is None:
        if len(present) != 2:
            raise ValueError(f"'{by}' has the levels {present}. Choose two with `levels`.")
        return tuple(sorted(present, reverse=True))
    if len(levels) != 2:
        raise ValueError(f"Choose two levels to compare, got {list(levels)}.")
    return tuple(levels)


def _group_means(
    frame: pd.DataFrame,
    value: str,
    by: str | None,
    name: Any,
    group: str
) -> pd.Series:
    rows = frame if by is None else frame[frame[by] == name]
    described = "These rows" if by is None else f"The level {name!r} of '{by}'"
    if rows.empty:
        raise ValueError(f"{described} has no rows.")
    means = rows.groupby(group, sort=True)[value].mean()
    if len(means) < 2:
        raise ValueError(f"{described} needs at least two groups of '{group}' for an interval, got {len(means)}.")
    return means


def _unpaired_ci(
    x: torch.Tensor,
    y: torch.Tensor,
    n_resamples: int,
    level: float,
    seed: int
) -> tuple[float, float]:
    generator = torch.Generator().manual_seed(seed)
    parts = []
    for size in _chunks(n_resamples):
        picks_x = torch.randint(len(x), (size, len(x)), generator=generator)
        picks_y = torch.randint(len(y), (size, len(y)), generator=generator)
        parts.append(x[picks_x].mean(dim=1) - y[picks_y].mean(dim=1))
    tail = (1 - level) / 2
    bounds = torch.quantile(torch.cat(parts), torch.tensor([tail, 1 - tail], dtype=torch.float64))
    return bounds[0].item(), bounds[1].item()


def _contrast_row(
    by: str | None,
    level_a: Any,
    level_b: Any,
    mean_a: float,
    mean_b: float,
    low: float,
    high: float,
    p_value: float
) -> pd.DataFrame:
    return pd.DataFrame({
        "by": [by],
        "level_a": [level_a],
        "level_b": [level_b],
        "mean_a": [mean_a],
        "mean_b": [mean_b],
        "difference": [mean_a - mean_b],
        "ci_low": [low],
        "ci_high": [high],
        "p_value": [p_value],
        "detected": [low > _ROUNDING or high < -_ROUNDING]
    })


def _chunks(total: int, size: int = 1024) -> Iterator[int]:
    # resamples are drawn in chunks so that a long input can't exhaust memory
    for start in range(0, total, size):
        yield min(size, total - start)
