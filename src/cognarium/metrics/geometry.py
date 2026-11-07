"""Measures of how similar two representations are, and of a single one's shape.

Every function takes matrices with one row per stimulus and flattens any trailing
dimensions. The rows of two matrices must line up: row i of each comes from the same
stimulus. Tensors on CUDA stay there, and MPS tensors move to CPU, since MPS has no
float64.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping
from functools import partial

import pandas as pd
import torch

# relative size below which a difference is only rounding error
_FLAT = 1e-12
_SPLIT = 1024


def linear_cka(x: torch.Tensor, y: torch.Tensor, *, debiased: bool = False) -> float:
    """Computes linear centred kernel alignment (Kornblith et al., 2019).

    CKA ignores rotation and overall scale, so it compares what two representations
    encode rather than how they lay it out across units.

    Args:
        x: One representation, shape (n, ...).
        y: The other representation, with the same number of rows.
        debiased: Whether to use the unbiased estimator, which stays near 0 for
            unrelated matrices even when they have many columns for their rows. The
            plain estimator is biased upwards there.

    Returns:
        The similarity, where 1 means identical up to rotation and scale. The
        debiased form can fall slightly below 0.

    Raises:
        ValueError: If the inputs are invalid, either has no variance, or `debiased`
            is set with fewer than 4 rows.
    """
    x, y = _pair(x, y)
    if debiased and len(x) < 4:
        raise ValueError(f"Debiased CKA needs at least 4 rows, got {len(x)}.")
    x, y = _centered(x), _centered(y)
    if debiased:
        return _cosine(_u_center(x @ x.T), _u_center(y @ y.T))
    if max(x.shape[1], y.shape[1]) < len(x):
        # the feature form is much cheaper when there are fewer columns than rows
        denominator = (x.T @ x).norm() * (y.T @ y).norm()
        return ((y.T @ x).square().sum() / denominator).item()
    return _cosine(x @ x.T, y @ y.T)


def rbf_cka(x: torch.Tensor, y: torch.Tensor, *, sigma_fraction: float = 1.0) -> float:
    """Computes centred kernel alignment with a Gaussian kernel.

    Args:
        x: One representation, shape (n, ...).
        y: The other representation, with the same number of rows.
        sigma_fraction: The kernel width, as a fraction of each matrix's median
            pairwise distance. Smaller values focus on local neighbourhoods.

    Returns:
        The similarity, where 1 means identical up to rotation and scale.

    Raises:
        ValueError: If the inputs are invalid, `sigma_fraction` isn't positive, or
            the median distance within a matrix is 0.
    """
    x, y = _pair(x, y)
    if sigma_fraction <= 0:
        raise ValueError(f"sigma_fraction must be positive, got {sigma_fraction}.")
    return _cosine(_double_center(_rbf_kernel(x, sigma_fraction)), _double_center(_rbf_kernel(y, sigma_fraction)))


def rsa(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    metric: str = "correlation",
    method: str = "spearman"
) -> float:
    """Computes representational similarity analysis.

    This compares the two matrices' pairwise distances between stimuli, so it needs
    no shared units or width.

    Args:
        x: One representation, shape (n, ...).
        y: The other representation, with the same number of rows.
        metric: The distance between stimuli, `"correlation"` or `"euclidean"`.
        method: How to compare the distances, `"spearman"` or `"pearson"`.

    Returns:
        The correlation between the two sets of distances.

    Raises:
        ValueError: If the inputs are invalid, an option is unknown, a row has no
            variance under correlation distance, or all of a matrix's distances are
            equal.
    """
    x, y = _pair(x, y)
    if method not in ("spearman", "pearson"):
        raise ValueError(f"Unknown method '{method}'. Use 'spearman' or 'pearson'.")
    first, second = _distances(x, metric), _distances(y, metric)
    if method == "spearman":
        first, second = _ranks(first), _ranks(second)
    return _cosine(_centered(first), _centered(second))


def procrustes(x: torch.Tensor, y: torch.Tensor, *, scale: bool = True) -> float:
    """Computes the distance left after optimally aligning two representations.

    Args:
        x: One representation, shape (n, ...).
        y: The other representation, with the same number of rows. The narrower
            matrix is padded with columns of zeros.
        scale: Whether to give both matrices a unit norm first, as in
            `scipy.spatial.procrustes`. Otherwise they are only centred, so a
            difference in scale counts towards the distance.

    Returns:
        The sum of squared differences after the best rotation or reflection. It is
        0 when they match, and at most 1 when `scale` is set.

    Raises:
        ValueError: If the inputs are invalid, `scale` is set and either has no
            variance, or `scale` isn't set and the values are too large to square.
    """
    x, y = _pair(x, y, rescale=scale)
    width = max(x.shape[1], y.shape[1])
    x = torch.nn.functional.pad(x, (0, width - x.shape[1]))
    y = torch.nn.functional.pad(y, (0, width - y.shape[1]))
    if scale:
        x, y = _centered(x), _centered(y)
        x, y = x / x.norm(), y / y.norm()
    else:
        x, y = x - x.mean(dim=0), y - y.mean(dim=0)
    if width > len(x):
        # only the rows' coordinates matter, and with few rows those are much smaller
        x, y = _coordinates(x), _coordinates(y)
    aligned = torch.linalg.svdvals(_not_overflowed(x.T @ y)).sum()
    if scale:
        return (1 - aligned**2).clamp_min(0).item()
    return _not_overflowed(x.square().sum() + y.square().sum() - 2 * aligned).clamp_min(0).item()


def mse(x: torch.Tensor, y: torch.Tensor) -> float:
    """Computes the mean squared error between two matrices.

    Args:
        x: One representation, shape (n, ...).
        y: The other representation, with the same shape.

    Returns:
        The mean squared difference across all entries.

    Raises:
        ValueError: If the inputs are invalid, differ in their columns, or are too
            large to square.
    """
    x, y = _pair(x, y, rescale=False)
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"MSE needs the same columns, got {x.shape[1]} and {y.shape[1]}.")
    return _not_overflowed((x - y).square().mean()).item()


def svcca(x: torch.Tensor, y: torch.Tensor, *, variance: float = 0.99) -> float:
    """Computes singular vector canonical correlation (Raghu et al., 2017).

    Each matrix is first reduced to the principal components that hold `variance` of
    its variance, which removes noisy directions before CCA aligns the two.

    Args:
        x: One representation, shape (n, ...).
        y: The other representation, with the same number of rows.
        variance: The share of variance to keep, above 0 and up to 1.

    Returns:
        The mean canonical correlation, between 0 and 1. CCA fits as many directions
        as it keeps, so it reads high when rows are few. It warns when the number of
        rows alone forces a correlation of 1.

    Raises:
        ValueError: If the inputs are invalid, `variance` is out of range, or either
            matrix has no variance.
    """
    x, y = _pair(x, y)
    if not 0 < variance <= 1:
        raise ValueError(f"variance must be above 0 and at most 1, got {variance}.")
    basis_x, basis_y = _principal_basis(x, variance), _principal_basis(y, variance)
    _warn_if_forced(basis_x, basis_y)
    return torch.linalg.svdvals(basis_x.T @ basis_y).clamp(0, 1).mean().item()


def pwcca(x: torch.Tensor, y: torch.Tensor) -> float:
    """Computes projection weighted canonical correlation (Morcos et al., 2018).

    Canonical directions are weighted by how much of a matrix's units they explain,
    so a direction that hardly matters counts for little. Weights come from whichever
    matrix has fewer independent components, or from `x` on a tie, so the score isn't
    symmetric in general.

    Args:
        x: One representation, shape (n, ...).
        y: The other representation, with the same number of rows.

    Returns:
        The weighted mean canonical correlation, between 0 and 1. CCA fits as many
        directions as there are components, so it reads high when rows are few. It
        warns when the number of rows alone forces a correlation of 1.

    Raises:
        ValueError: If the inputs are invalid, or either matrix has no variance.
    """
    x, y = _pair(x, y)
    basis_x, basis_y = _principal_basis(x), _principal_basis(y)
    _warn_if_forced(basis_x, basis_y)
    if basis_x.shape[1] > basis_y.shape[1]:
        x, basis_x, basis_y = y, basis_y, basis_x
    left, correlations, _ = torch.linalg.svd(basis_x.T @ basis_y, full_matrices=False)
    weights = ((basis_x @ left).T @ _centered(x)).abs().sum(dim=1)
    return ((weights / weights.sum()) * correlations.clamp(0, 1)).sum().item()


def participation_ratio(x: torch.Tensor) -> float:
    """Computes how many dimensions share a representation's variance.

    Args:
        x: The representation, shape (n, ...).

    Returns:
        The squared sum of the covariance eigenvalues over the sum of their squares.
        It is 1 when one dimension holds all the variance, and the number of columns
        when they all hold the same amount.

    Raises:
        ValueError: If the input is invalid or has no variance.
    """
    variances = torch.linalg.svdvals(_centered(_matrix(x))).square()
    return (variances.sum().square() / variances.square().sum()).item()


def intrinsic_dimension(x: torch.Tensor) -> float:
    """Estimates the dimension of the manifold a representation lies on.

    This is TwoNN (Facco et al., 2017), which reads the dimension from the ratio of
    each point's second to first nearest neighbour distance. It discards the largest
    10% of ratios, as the paper does. Repeated rows are dropped, since a neighbour at
    distance 0 has no defined ratio.

    Args:
        x: The representation, shape (n, ...).

    Returns:
        The estimated dimension.

    Raises:
        ValueError: If the input is invalid, it has fewer than 3 distinct rows, or
            neighbour distances carry no information, as on a regular grid.
    """
    points = _matrix(x).unique(dim=0)
    if len(points) < 3:
        raise ValueError(f"The intrinsic dimension needs at least 3 distinct rows, got {len(points)}.")
    nearest = torch.cat(
        [
            torch.cdist(part, points, compute_mode="donot_use_mm_for_euclid_dist").topk(3, largest=False).values
            for part in points.split(_SPLIT)
        ]
    )
    # the closest point to each row is the row itself
    first, second = nearest[:, 1], nearest[:, 2]
    ratios = (second[first > 0] / first[first > 0]).sort().values
    kept = int(0.9 * len(ratios))
    log_ratios = ratios[:kept].log()
    log_survival = -torch.log1p(-torch.arange(1, kept + 1, device=ratios.device).double() / len(ratios))
    denominator = log_ratios.square().sum()
    if denominator == 0:
        raise ValueError("Every point's two nearest neighbours are equally far, so TwoNN can't estimate a dimension.")
    return ((log_ratios * log_survival).sum() / denominator).item()


GEOMETRY: dict[str, Callable[[torch.Tensor, torch.Tensor], float]] = {
    "linear_cka": linear_cka,
    "debiased_cka": partial(linear_cka, debiased=True),
    "rbf_cka": rbf_cka,
    "rsa": rsa,
    "procrustes": procrustes,
    "mse": mse,
    "svcca": svcca,
    "pwcca": pwcca
}


def layer_matrix(
    states: Mapping[str, torch.Tensor],
    metric: str | Callable[[torch.Tensor, torch.Tensor], float] = "linear_cka"
) -> pd.DataFrame:
    """Compares every layer with every other layer.

    Args:
        states: Each layer's states, shape (n, ...), with rows that line up.
        metric: A name from `GEOMETRY`, or a function of two matrices.

    Returns:
        A square table labelled by layer. Row a, column b holds `metric(a, b)`.

    Raises:
        ValueError: If `metric` is an unknown name.
    """
    if isinstance(metric, str):
        if metric not in GEOMETRY:
            raise ValueError(f"Unknown metric '{metric}'. Known metrics: {', '.join(sorted(GEOMETRY))}.")
        metric = GEOMETRY[metric]
    names = list(states)
    rows = [[metric(states[first], states[second]) for second in names] for first in names]
    return pd.DataFrame(rows, index=names, columns=names, dtype=float)


def _matrix(x: torch.Tensor, *, rescale: bool = True) -> torch.Tensor:
    if x.ndim == 0 or len(x) < 2:
        raise ValueError("Geometry metrics need at least 2 rows.")
    x = x.detach()
    if x.device.type == "mps":
        x = x.cpu()
    x = x.double()
    x = x.flatten(start_dim=1) if x.ndim > 1 else x[:, None]
    if x.shape[1] == 0:
        raise ValueError("Geometry metrics need at least 1 column.")
    if not x.isfinite().all():
        raise ValueError("Geometry metrics need finite values.")
    return _rescaled(x) if rescale else x


def _rescaled(x: torch.Tensor) -> torch.Tensor:
    # a power of two changes no digits, and squares then can't overflow or underflow
    exponent = math.frexp(x.abs().max().item())[1]
    return x * math.ldexp(1.0, -max(exponent, -1000))


def _not_overflowed(values: torch.Tensor) -> torch.Tensor:
    if not values.isfinite().all():
        raise ValueError("These values are too large to square in float64. Rescale them first.")
    return values


def _pair(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    rescale: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = _matrix(x, rescale=rescale), _matrix(y, rescale=rescale)
    if len(x) != len(y):
        raise ValueError(f"Both matrices need the same number of rows, got {len(x)} and {len(y)}.")
    return x, y.to(x.device)


def _centered(x: torch.Tensor) -> torch.Tensor:
    centered = x - x.mean(dim=0, keepdim=True)
    if centered.abs().max() <= _FLAT * x.abs().max():
        raise ValueError("This metric is undefined for a matrix with no variance.")
    return centered


def _cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    denominator = first.norm() * second.norm()
    if denominator == 0:
        raise ValueError("This metric is undefined for a matrix with no variance.")
    return ((first * second).sum() / denominator).item()


def _u_center(gram: torch.Tensor) -> torch.Tensor:
    gram = gram.clone().fill_diagonal_(0)
    means = gram.sum(dim=0) / (len(gram) - 2)
    means = means - means.sum() / (2 * (len(gram) - 1))
    return (gram - means[:, None] - means[None, :]).fill_diagonal_(0)


def _double_center(kernel: torch.Tensor) -> torch.Tensor:
    return kernel - kernel.mean(dim=0, keepdim=True) - kernel.mean(dim=1, keepdim=True) + kernel.mean()


def _exact_distances(x: torch.Tensor) -> torch.Tensor:
    # the faster matrix mode turns the zero distance of repeated rows into rounding noise
    return torch.cdist(x, x, compute_mode="donot_use_mm_for_euclid_dist")


def _upper_triangle(square: torch.Tensor) -> torch.Tensor:
    first, second = torch.triu_indices(len(square), len(square), offset=1, device=square.device)
    return square[first, second]


def _rbf_kernel(x: torch.Tensor, sigma_fraction: float) -> torch.Tensor:
    squared = _exact_distances(x).square()
    median = _upper_triangle(squared).median()
    if median == 0:
        raise ValueError("RBF CKA is undefined when the median distance is 0, as in a matrix with no variance.")
    return (-squared / (2 * sigma_fraction**2 * median)).exp()


def _distances(x: torch.Tensor, metric: str) -> torch.Tensor:
    if metric == "euclidean":
        _centered(x)
        return _upper_triangle(_exact_distances(x))
    if metric == "correlation":
        centered = x - x.mean(dim=1, keepdim=True)
        if (centered.abs().amax(dim=1) <= _FLAT * x.abs().amax(dim=1)).any():
            raise ValueError("Correlation distance is undefined for a row with no variance. Try metric='euclidean'.")
        unit = centered / centered.norm(dim=1, keepdim=True)
        return _upper_triangle(1 - unit @ unit.T)
    raise ValueError(f"Unknown metric '{metric}'. Use 'correlation' or 'euclidean'.")


def _ranks(values: torch.Tensor) -> torch.Tensor:
    # tied values share the mean of the ranks they span
    _, inverse, counts = values.unique(return_inverse=True, return_counts=True)
    last = counts.cumsum(dim=0).double()
    return (last - (counts - 1) / 2)[inverse]


def _principal_basis(x: torch.Tensor, variance: float = 1.0) -> torch.Tensor:
    left, singular, _ = torch.linalg.svd(_centered(x), full_matrices=False)
    # directions at the rounding level aren't in the data, and CCA would still fit them
    real = singular > singular[0] * max(x.shape) * torch.finfo(x.dtype).eps
    energy = singular[real].square().cumsum(dim=0)
    count = int((energy < variance * energy[-1] * (1 - _FLAT)).sum()) + 1
    return left[:, :count]


def _coordinates(x: torch.Tensor) -> torch.Tensor:
    left, singular, _ = torch.linalg.svd(x, full_matrices=False)
    return left * singular


def _warn_if_forced(basis_x: torch.Tensor, basis_y: torch.Tensor) -> None:
    n_rows = len(basis_x)
    if basis_x.shape[1] + basis_y.shape[1] > n_rows - 1:
        warnings.warn(
            f"{n_rows} rows can't separate {basis_x.shape[1]} and {basis_y.shape[1]} components, "
            "so some canonical correlations are 1 whatever the data. Use more rows, or a metric like CKA.",
            stacklevel=3
        )
