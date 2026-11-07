"""What a subject remembers from one trial to the next."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextPolicy:
    """Which past trials stay in a subject's context.

    Attributes:
        keep: How many past trials stay in context. None keeps every one, and 0 keeps none.
        feedback: Whether past trials show the correct answer rather than the subject's
            own response.
    """

    keep: int | None
    feedback: bool = False


isolated = ContextPolicy(keep=0)
cumulative = ContextPolicy(keep=None)


def window(k: int, *, feedback: bool = False) -> ContextPolicy:
    """Keeps only the most recent trials in context.

    Args:
        k: How many past trials to keep.
        feedback: Whether past trials show the correct answer rather than the subject's
            own response.

    Returns:
        The policy.

    Raises:
        ValueError: If `k` is below 1.
    """
    if k < 1:
        raise ValueError(f"A window must keep at least 1 trial, got {k}. Use cg.context.isolated to keep none.")
    return ContextPolicy(keep=k, feedback=feedback)
