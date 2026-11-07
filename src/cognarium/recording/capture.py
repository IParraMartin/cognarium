"""Collecting a model's layer outputs while it responds."""

from __future__ import annotations

from collections.abc import (
    Callable,
    Iterator,
    Mapping,
    Sequence
)
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class StateCapture:
    """What to record while a subject responds, and where to send it.

    Attributes:
        layers: The names of the layers to record.
        pool: How to reduce the time dimension: `"last"` keeps the last real token,
            `"mean"` averages over real tokens, and None keeps every token.
        write: Called as `write(layer, states, queries, lengths=None)`, where
            `queries` holds the position of each row in the `respond` call, and
            `lengths` holds each row's token count when `pool` is None.
    """

    layers: tuple[str, ...]
    pool: str | None
    write: Callable[..., None]


def pool_states(
    output: Any,
    mask: torch.Tensor,
    pool: str | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Reduces a layer's output to what will be stored.

    Args:
        output: The layer's output, which is a tensor of shape (batch, time, ...), a
            tuple starting with one, or an object with `last_hidden_state`.
        mask: The attention mask of a right-padded batch, shape (batch, time).
        pool: `"last"`, `"mean"` or None. See `StateCapture`.

    Returns:
        The states, and each row's token count when `pool` is None. Otherwise the
        second value is None.

    Raises:
        ValueError: If `pool` is unknown, or the output isn't a tensor that starts
            with the batch and time dimensions.
    """
    if pool not in ("last", "mean", None):
        raise ValueError(f"Unknown pool '{pool}'. Use 'last', 'mean' or None.")
    states = _as_tensor(output)
    mask = mask.to(states.device)
    if states.shape[:2] != mask.shape:
        raise ValueError(
            f"A layer output of shape {tuple(states.shape)} doesn't fit a batch of shape {tuple(mask.shape)}. "
            "It must start with the batch and time dimensions."
        )
    lengths = mask.sum(dim=1)
    if pool is None:
        return states, lengths
    if pool == "last":
        return states[torch.arange(len(states), device=states.device), lengths - 1], None
    weights = mask.to(states.dtype).reshape(*mask.shape, *[1] * (states.ndim - 2))
    return (states * weights).sum(dim=1) / weights.sum(dim=1), None


@contextmanager
def hook_layers(
    modules: Mapping[str, torch.nn.Module],
    capture: StateCapture
) -> Iterator[Callable[[torch.Tensor, Sequence[int]], None]]:
    """Listens to the requested layers for as long as the context is open.

    Only the requested layers are hooked, and the hooks come off on exit, including
    after an exception.

    Args:
        modules: Every layer the subject can record, by name.
        capture: Which layers to record, and where to send them.

    Yields:
        `on_batch(mask, queries)`, to call after a forward pass whose outputs should
        be recorded. A forward pass with no `on_batch` call records nothing.

    Raises:
        ValueError: If a requested layer doesn't exist, or a batch didn't run it.
    """
    unknown = sorted(set(capture.layers) - set(modules))
    if unknown:
        raise ValueError(f"Unknown layers {unknown}. This subject has {list(modules)}.")
    latest = {}

    def remember(name: str) -> Callable[..., None]:
        def hook(module: torch.nn.Module, inputs: Any, output: Any) -> None:
            latest[name] = output

        return hook

    def on_batch(mask: torch.Tensor, queries: Sequence[int]) -> None:
        for name in capture.layers:
            if name not in latest:
                raise ValueError(f"Layer '{name}' didn't run during the forward call, so it can't be recorded.")
            states, lengths = pool_states(latest.pop(name), mask, capture.pool)
            capture.write(
                name,
                states,
                queries,
                lengths
            )

    handles = [modules[name].register_forward_hook(remember(name)) for name in capture.layers]
    try:
        yield on_batch
    finally:
        for handle in handles:
            handle.remove()


def _as_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, tuple) and output:
        output = output[0]
    output = getattr(output, "last_hidden_state", output)
    if not isinstance(output, torch.Tensor):
        raise ValueError(f"Can't read a tensor from a layer output of type {type(output).__name__}.")
    return output
