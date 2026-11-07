"""Information theory metrics, working on log probabilities."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pandas as pd
import torch


def entropy(log_probs: torch.Tensor, *, dim: int = -1, base: float = 2.0) -> torch.Tensor:
    """Computes the entropy of a distribution.

    Args:
        log_probs: Log probabilities. Impossible outcomes (-inf) contribute nothing.
        dim: The dimension holding the outcomes.
        base: The logarithm base. 2 gives bits.

    Returns:
        The entropy, with `dim` removed.
    """
    probs = log_probs.exp()
    terms = torch.where(probs > 0, probs * log_probs, torch.zeros_like(probs))
    # adding zero turns -0.0 into 0.0, which would otherwise show up in printed tables
    return -terms.sum(dim=dim) / math.log(base) + 0.0


def kl_divergence(
    log_p: torch.Tensor,
    log_q: torch.Tensor,
    *,
    dim: int = -1,
    base: float = 2.0
) -> torch.Tensor:
    """Computes the KL divergence of p from q.

    Args:
        log_p: Log probabilities of p.
        log_q: Log probabilities of q.
        dim: The dimension holding the outcomes.
        base: The logarithm base. 2 gives bits.

    Returns:
        The divergence, with `dim` removed. It is infinite when p gives
        probability to an outcome q rules out.
    """
    probs = log_p.exp()
    terms = torch.where(probs > 0, probs * (log_p - log_q), torch.zeros_like(probs))
    return terms.sum(dim=dim) / math.log(base)


def surprisal(log_prob: torch.Tensor, *, base: float = 2.0) -> torch.Tensor:
    """Computes how surprising an outcome is.

    Args:
        log_prob: The outcome's log probability.
        base: The logarithm base. 2 gives bits.

    Returns:
        The surprisal.
    """
    return -log_prob / math.log(base)


def cross_entropy(
    log_p: torch.Tensor,
    log_q: torch.Tensor,
    *,
    dim: int = -1,
    base: float = 2.0
) -> torch.Tensor:
    """Computes the cross-entropy of q relative to p.

    Args:
        log_p: Log probabilities of p.
        log_q: Log probabilities of q.
        dim: The dimension holding the outcomes.
        base: The logarithm base. 2 gives bits.

    Returns:
        The cross-entropy, with `dim` removed. It is infinite when p gives
        probability to an outcome q rules out.
    """
    probs = log_p.exp()
    terms = torch.where(probs > 0, probs * log_q, torch.zeros_like(probs))
    return -terms.sum(dim=dim) / math.log(base)


def js_divergence(
    log_p: torch.Tensor,
    log_q: torch.Tensor,
    *,
    dim: int = -1,
    base: float = 2.0
) -> torch.Tensor:
    """Computes the Jensen-Shannon divergence, which is symmetric and always finite.

    Args:
        log_p: Log probabilities of p.
        log_q: Log probabilities of q.
        dim: The dimension holding the outcomes.
        base: The logarithm base. 2 gives bits.

    Returns:
        The divergence, with `dim` removed. In bits it lies between 0 and 1.
    """
    log_middle = torch.logaddexp(log_p, log_q) - math.log(2)
    to_p = kl_divergence(log_p, log_middle, dim=dim, base=base)
    to_q = kl_divergence(log_q, log_middle, dim=dim, base=base)
    return (to_p + to_q) / 2


def joint_entropy(log_joint: torch.Tensor, *, base: float = 2.0) -> torch.Tensor:
    """Computes the entropy of two variables together.

    Args:
        log_joint: Log probabilities of each pair, shape (n_x, n_y).
        base: The logarithm base. 2 gives bits.

    Returns:
        H(X, Y).
    """
    return entropy(log_joint.flatten(), base=base)


def conditional_entropy(log_joint: torch.Tensor, *, base: float = 2.0) -> torch.Tensor:
    """Computes how uncertain Y remains once X is known.

    Args:
        log_joint: Log probabilities of each pair, shape (n_x, n_y).
        base: The logarithm base. 2 gives bits.

    Returns:
        H(Y | X).
    """
    return joint_entropy(log_joint, base=base) - entropy(log_joint.logsumexp(dim=1), base=base)


def mutual_information(log_joint: torch.Tensor, *, base: float = 2.0) -> torch.Tensor:
    """Computes how much knowing X says about Y.

    Args:
        log_joint: Log probabilities of each pair, shape (n_x, n_y).
        base: The logarithm base. 2 gives bits.

    Returns:
        I(X; Y), which is never negative.
    """
    log_independent = log_joint.logsumexp(dim=1, keepdim=True) + log_joint.logsumexp(dim=0, keepdim=True)
    return kl_divergence(log_joint.flatten(), log_independent.flatten(), base=base).clamp_min(0.0)


def perplexity(log_probs: torch.Tensor) -> torch.Tensor:
    """Computes the perplexity of a sequence of observed outcomes.

    Args:
        log_probs: The natural log probability given to each outcome that occurred.

    Returns:
        The exponential of the mean negative log probability. A model that spreads
        its belief evenly over n outcomes has a perplexity of n.
    """
    return (-log_probs.mean()).exp()


def ksg_mutual_information(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    k: int = 3,
    base: float = 2.0
) -> float:
    """Estimates mutual information between continuous variables.

    This is estimator 1 of Kraskov, Stoegbauer and Grassberger (2004), which is
    meant for continuous values such as hidden states. Its memory use grows with
    the square of the sample count.

    Args:
        x: Samples of X, shape (n, ...).
        y: Samples of Y, shape (n, ...).
        k: How many neighbours to use. A larger k has less variance and more bias.
        base: The logarithm base. 2 gives bits.

    Returns:
        The estimate. Sampling error can make it slightly negative.

    Raises:
        ValueError: If `x` and `y` differ in rows, there are no more samples than
            `k`, or a sample is repeated `k` or more times.
    """
    if len(x) != len(y):
        raise ValueError(f"x and y must have the same rows, got {len(x)} and {len(y)}.")
    n = len(x)
    if n <= k:
        raise ValueError(f"KSG needs more samples than k={k}, got {n}.")
    x_distances = torch.cdist(x.flatten(1).double().cpu(), x.flatten(1).double().cpu(), p=float("inf"))
    y_distances = torch.cdist(y.flatten(1).double().cpu(), y.flatten(1).double().cpu(), p=float("inf"))
    joint = torch.maximum(x_distances, y_distances).fill_diagonal_(torch.inf)
    radius = joint.kthvalue(k, dim=1).values[:, None]
    if (radius == 0).any():
        raise ValueError(f"KSG can't use samples with {k} or more duplicates. Add a little noise to them.")
    n_x = (x_distances < radius).sum(dim=1) - 1
    n_y = (y_distances < radius).sum(dim=1) - 1
    digamma = torch.special.digamma
    neighbours = (digamma(n_x.double() + 1) + digamma(n_y.double() + 1)).mean()
    estimate = digamma(torch.tensor(float(k))) + digamma(torch.tensor(float(n))) - neighbours
    return estimate.item() / math.log(base)


def information_bottleneck(
    log_joint: torch.Tensor,
    betas: Sequence[float],
    *,
    n_clusters: int | None = None,
    n_iterations: int = 500,
    tolerance: float = 1e-8,
    seed: int = 0,
    base: float = 2.0
) -> pd.DataFrame:
    """Traces the information bottleneck curve of a joint distribution.

    For each beta this runs the iterative algorithm of Tishby, Pereira and Bialek
    (1999), which compresses X into clusters T that keep what X says about Y. The
    algorithm finds a local optimum, so different seeds can give different points.

    Args:
        log_joint: Log probabilities of each pair, shape (n_x, n_y).
        betas: How much to favour relevance over compression, one point per value.
        n_clusters: How many clusters T may use. Defaults to `n_x`.
        n_iterations: The most updates to run for each beta.
        tolerance: Stops early when the encoder moves less than this.
        seed: Seeds each beta's starting encoder.
        base: The logarithm base. 2 gives bits.

    Returns:
        A table with one row per beta and the columns `beta`, `complexity`, which
        is I(T; X), and `relevance`, which is I(T; Y).
    """
    joint = log_joint.double().cpu()
    log_x = joint.logsumexp(dim=1)
    log_y_given_x = joint - log_x[:, None]
    width = len(joint) if n_clusters is None else n_clusters
    rows = []
    for beta in betas:
        generator = torch.Generator().manual_seed(seed)
        start = torch.rand(len(joint), width, generator=generator, dtype=torch.float64)
        log_encoder = (start / start.sum(dim=1, keepdim=True)).log()
        for _ in range(n_iterations):
            updated = _bottleneck_update(log_encoder, log_x, log_y_given_x, beta)
            moved = (updated.exp() - log_encoder.exp()).abs().max()
            log_encoder = updated
            if moved < tolerance:
                break
        joint_tx = log_encoder + log_x[:, None]
        joint_ty = (log_encoder[:, :, None] + joint[:, None, :]).logsumexp(dim=0)
        rows.append(
            {
                "beta": beta,
                "complexity": mutual_information(joint_tx, base=base).item(),
                "relevance": mutual_information(joint_ty, base=base).item()
            }
        )
    return pd.DataFrame(rows)


def _bottleneck_update(
    log_encoder: torch.Tensor,
    log_x: torch.Tensor,
    log_y_given_x: torch.Tensor,
    beta: float
) -> torch.Tensor:
    log_joint_tx = log_encoder + log_x[:, None]
    log_t = log_joint_tx.logsumexp(dim=0)
    # p(y | t) is a mixture of p(y | x), weighted by how much of t each x makes up
    # a cluster that has died has no probability to divide by, and stays dead
    log_x_given_t = torch.nan_to_num(log_joint_tx - log_t, nan=-torch.inf, posinf=torch.inf, neginf=-torch.inf)
    log_y_given_t = (log_x_given_t[:, :, None] + log_y_given_x[:, None, :]).logsumexp(dim=0)
    divergence = kl_divergence(log_y_given_x[:, None, :], log_y_given_t[None, :, :], base=math.e)
    log_scores = log_t - beta * divergence
    return log_scores - log_scores.logsumexp(dim=1, keepdim=True)
