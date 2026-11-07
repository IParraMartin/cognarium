"""Errors and warnings raised by cognarium."""

from __future__ import annotations


class CognariumError(Exception):
    """Base class for errors raised by cognarium."""


class MissingAbilityError(CognariumError):
    """A paradigm or metric needs something the subject or results lack."""


class RunConflictError(CognariumError):
    """A run folder already holds results that a new run must not touch."""


class UntrainedModelWarning(UserWarning):
    """A model with untrained weights is being used as if it had learned."""
