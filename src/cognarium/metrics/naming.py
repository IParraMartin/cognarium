"""How efficiently a naming system compresses meanings, after Zaslavsky et al. (2018).

A naming system is an encoder q(w | m): one row per item, giving the probability of
each word for it. Each item stands for a meaning, a distribution over the items that
peaks on it and spreads to its neighbours. The information bottleneck (IB) trades
the system's complexity, I(M; W), against its accuracy, I(W; U), the information its
words keep about the items. Imel and Zaslavsky (2026) use these measures to follow
naming systems through iterated in-context learning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
import torch

from cognarium.metrics.information import mutual_information

_MERGE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class Frontier:
    """The optimal naming systems along the information bottleneck curve.

    Attributes:
        betas: The trade-off values, in increasing order.
        complexity: I(M; W) of the optimal system at each beta.
        accuracy: I(W; U) of the optimal system at each beta.
        encoders: The optimal system at each beta, one row per item.
    """

    betas: torch.Tensor
    complexity: torch.Tensor
    accuracy: torch.Tensor
    encoders: tuple[torch.Tensor, ...]

    def encoder_at(self, beta: float) -> torch.Tensor:
        """Gives the optimal system computed at the beta nearest to this one.

        Args:
            beta: The trade-off value.

        Returns:
            The encoder, one row per item.
        """
        return self.encoders[int((self.betas.log() - math.log(beta)).abs().argmin())]

    def to_pandas(self) -> pd.DataFrame:
        """Gives the curve as a table.

        Returns:
            One row per beta, with `beta`, `complexity` and `accuracy`.
        """
        return pd.DataFrame({
            "beta": self.betas.tolist(),
            "complexity": self.complexity.tolist(),
            "accuracy": self.accuracy.tolist()
        })


def meanings(points: torch.Tensor, sigma: float) -> torch.Tensor:
    """Turns each item into a meaning: a Gaussian blur over the items around it.

    Args:
        points: Each item's coordinates, shape (n_items, n_features). For colour these
            are CIELAB coordinates.
        sigma: How far the blur spreads, in the coordinates' units.

    Returns:
        m(u), shape (n_items, n_items), where row m is a distribution over items u.
    """
    points = points.double()
    log_weights = -torch.cdist(points, points).square() / (2 * sigma**2)
    return (log_weights - log_weights.logsumexp(dim=1, keepdim=True)).exp()


def ib_complexity(encoder: torch.Tensor, prior: torch.Tensor, *, base: float = 2.0) -> float:
    """Measures I(M; W), how much information a naming system spends on the items.

    Args:
        encoder: q(w | m), shape (n_items, n_words), with rows that sum to 1.
        prior: How often each item comes up, shape (n_items,).
        base: The logarithm base. 2 gives bits.

    Returns:
        The complexity.
    """
    joint = prior.double()[:, None] * encoder.double()
    return mutual_information(joint.log(), base=base).item()


def ib_accuracy(
    encoder: torch.Tensor,
    prior: torch.Tensor,
    meanings: torch.Tensor,
    *,
    base: float = 2.0
) -> float:
    """Measures I(W; U), how much a naming system's words say about the items meant.

    Args:
        encoder: q(w | m), shape (n_items, n_words), with rows that sum to 1.
        prior: How often each item comes up, shape (n_items,).
        meanings: m(u), shape (n_items, n_items), from `meanings`.
        base: The logarithm base. 2 gives bits.

    Returns:
        The accuracy.
    """
    joint = encoder.double().T @ (prior.double()[:, None] * meanings.double())
    return mutual_information(joint.log(), base=base).item()


def ib_frontier(
    prior: torch.Tensor,
    meanings: torch.Tensor,
    *,
    betas: torch.Tensor | None = None,
    n_iterations: int = 200,
    tolerance: float = 1e-9,
    base: float = 2.0
) -> Frontier:
    """Traces the optimal naming systems by reverse deterministic annealing.

    The search starts at the largest beta from a system with one word per item,
    which is optimal there, and lowers beta step by step, starting each step from
    the last optimum. Words whose meanings become identical are merged, which leaves
    the objective unchanged and keeps the search fast.

    Args:
        prior: How often each item comes up, shape (n_items,).
        meanings: m(u), shape (n_items, n_items), from `meanings`.
        betas: The trade-off values to trace. Defaults to 300 values spaced evenly in
            log scale from 1 to 2^12.
        n_iterations: The most updates to run for each beta.
        tolerance: Stops a beta's updates early when the encoder moves less than this.
        base: The logarithm base of the reported complexity and accuracy.

    Returns:
        The frontier, ordered by increasing beta.
    """
    prior = prior.double() / prior.double().sum()
    meanings = meanings.double()
    schedule = (
        torch.logspace(0, 12, 300, base=2.0, dtype=torch.float64)
        if betas is None
        else torch.as_tensor(betas, dtype=torch.float64).sort().values
    )
    self_information = (meanings * meanings.log()).sum(dim=1)
    encoder = torch.eye(len(prior), dtype=torch.float64)
    found = []
    for beta in schedule.flip(0).tolist():
        for _ in range(n_iterations):
            updated = _bottleneck_step(
                encoder,
                prior,
                meanings,
                self_information,
                beta
            )
            moved = (updated - encoder).abs().max().item()
            encoder = updated
            if moved < tolerance:
                break
        encoder = _merge(encoder, prior, meanings)
        found.append(encoder)
    found.reverse()
    return Frontier(
        betas=schedule,
        complexity=torch.tensor(
            [ib_complexity(system, prior, base=base) for system in found],
            dtype=torch.float64
        ),
        accuracy=torch.tensor(
            [ib_accuracy(system, prior, meanings, base=base) for system in found],
            dtype=torch.float64
        ),
        encoders=tuple(found)
    )


def efficiency_loss(
    encoder: torch.Tensor,
    prior: torch.Tensor,
    meanings: torch.Tensor,
    frontier: Frontier
) -> tuple[float, float]:
    """Measures how far a naming system falls from the optimal trade-off.

    This is epsilon = min over beta of (F_beta[q] - F*_beta) / beta, with
    F_beta = I(M; W) - beta I(W; U), as in Zaslavsky et al. (2018).

    Args:
        encoder: q(w | m), shape (n_items, n_words), with rows that sum to 1.
        prior: How often each item comes up, shape (n_items,).
        meanings: m(u), shape (n_items, n_items), from `meanings`.
        frontier: The optimal systems, from `ib_frontier` with the same prior and
            meanings.

    Returns:
        The efficiency loss, in the frontier's units, and the beta that attains it.
    """
    complexity = ib_complexity(encoder, prior)
    accuracy = ib_accuracy(encoder, prior, meanings)
    gaps = (complexity - frontier.complexity) / frontier.betas - (accuracy - frontier.accuracy)
    best = int(gaps.argmin())
    return gaps[best].item(), frontier.betas[best].item()


def gnid(first: torch.Tensor, second: torch.Tensor, prior: torch.Tensor) -> float:
    """Measures the generalised normalised information distance between two systems.

    It is 1 - I(W1; W2) / max(I(W1; W1'), I(W2; W2')), where each pair of words names
    the same item, so label names don't matter. 0 means the two systems divide the
    items alike, and 1 that they share no information.

    Args:
        first: One system's q(w | m), shape (n_items, n_words).
        second: The other's, shape (n_items, n_other_words).
        prior: How often each item comes up, shape (n_items,).

    Returns:
        The distance, from 0 to 1. Two systems that each give every item one word
        are 0 apart.
    """
    first, second, prior = first.double(), second.double(), prior.double()

    def shared(left: torch.Tensor, right: torch.Tensor) -> float:
        return mutual_information((left.T @ (prior[:, None] * right)).log()).item()

    largest = max(shared(first, first), shared(second, second))
    if largest <= 1e-12:
        return 0.0
    return min(1.0, max(0.0, 1.0 - shared(first, second) / largest))


def co_naming_mse(first: torch.Tensor, second: torch.Tensor) -> float:
    """Compares which items two systems give the same label, whatever the labels are.

    Args:
        first: One system's q(w | m), shape (n_items, n_words).
        second: The other's, shape (n_items, n_other_words).

    Returns:
        The mean squared difference, over every ordered pair of different items, in
        the probability that the two items get the same label.
    """
    first, second = first.double(), second.double()
    difference = (first @ first.T - second @ second.T).square()
    off_diagonal = ~torch.eye(len(first), dtype=torch.bool)
    return difference[off_diagonal].mean().item()


def _bottleneck_step(
    encoder: torch.Tensor,
    prior: torch.Tensor,
    meanings: torch.Tensor,
    self_information: torch.Tensor,
    beta: float
) -> torch.Tensor:
    marginal = prior @ encoder
    # the listener's guess for each word mixes the meanings that word was used for
    decoder = (encoder * prior[:, None]).T @ meanings / marginal[:, None]
    divergence = self_information[:, None] - meanings @ decoder.log().T
    log_scores = marginal.log()[None, :] - beta * divergence
    return (log_scores - log_scores.logsumexp(dim=1, keepdim=True)).exp()


def _merge(encoder: torch.Tensor, prior: torch.Tensor, meanings: torch.Tensor) -> torch.Tensor:
    alive = encoder[:, prior @ encoder > 1e-15]
    decoder = (alive * prior[:, None]).T @ meanings / (prior @ alive)[:, None]
    close = torch.cdist(decoder, decoder, p=math.inf) < _MERGE_TOLERANCE
    # each word joins the first word whose listener guess matches its own
    _, groups = close.int().argmax(dim=1).unique(return_inverse=True)
    merged = torch.zeros(len(alive), int(groups.max()) + 1, dtype=alive.dtype)
    return merged.index_add_(1, groups, alive)
