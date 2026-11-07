"""Effect sizes, resampling tests and multiple comparison correction."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import torch


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
            signs = torch.randint(2, (size, len(x)), generator=generator, dtype=torch.float64) * 2 - 1
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


def _chunks(total: int, size: int = 1024) -> Iterator[int]:
    # resamples are drawn in chunks so that a long input can't exhaust memory
    for start in range(0, total, size):
        yield min(size, total - start)
