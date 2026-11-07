"""Information theory metrics, working on log probabilities."""

from __future__ import annotations

import math

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
    return -terms.sum(dim=dim) / math.log(base)


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
