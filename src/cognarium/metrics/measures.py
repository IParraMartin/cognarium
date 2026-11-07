"""Metrics computed from a run's results table.

Each measure takes `(frame, group_by, manifest, **options)` and returns a table with
the `group_by` columns and a `value` column. Everything runs on CPU in float64, so
device limits never affect a reported number.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch

from cognarium.errors import MissingAbilityError
from cognarium.metrics import information


def _accuracy(frame: pd.DataFrame, group_by: Sequence[str], manifest: Mapping[str, Any]) -> pd.DataFrame:
    _require(frame, "accuracy", ["response", "answer", *group_by])
    scored = frame[frame["answer"].notna()]
    if scored.empty:
        raise MissingAbilityError("Measure 'accuracy' needs trials with a correct answer.")
    return _mean_by(scored, group_by, (scored["response"] == scored["answer"]).to_numpy(dtype=float))


def _fidelity(frame: pd.DataFrame, group_by: Sequence[str], manifest: Mapping[str, Any]) -> pd.DataFrame:
    _require(frame, "fidelity", ["response", "parent_label", *group_by])
    return _mean_by(frame, group_by, (frame["response"] == frame["parent_label"]).to_numpy(dtype=float))


def _entropy(
    frame: pd.DataFrame,
    group_by: Sequence[str],
    manifest: Mapping[str, Any],
    raw: bool = False
) -> pd.DataFrame:
    column = "log_probs" if raw else "shaped_log_probs"
    _require(frame, "entropy", [column, *group_by])
    values = information.entropy(_stack(frame[column])).numpy()
    return _mean_by(frame, group_by, values)


def _surprisal(
    frame: pd.DataFrame,
    group_by: Sequence[str],
    manifest: Mapping[str, Any],
    raw: bool = False
) -> pd.DataFrame:
    column = "log_probs" if raw else "shaped_log_probs"
    _require(frame, "surprisal", [column, "options", "response", *group_by])
    chosen = [
        float(log_probs[list(options).index(response)])
        for log_probs, options, response in zip(frame[column], frame["options"], frame["response"], strict=True)
    ]
    values = information.surprisal(torch.tensor(chosen, dtype=torch.float64)).numpy()
    return _mean_by(frame, group_by, values)


def _chain_entropy(frame: pd.DataFrame, group_by: Sequence[str], manifest: Mapping[str, Any]) -> pd.DataFrame:
    _require(frame, "chain_entropy", ["chain", "group", *group_by])

    def plug_in_entropy(part: pd.DataFrame) -> float:
        counts = torch.tensor(part["group"].value_counts().to_numpy(), dtype=torch.float64)
        return information.entropy((counts / counts.sum()).log()).item()

    return _apply_by(frame.drop_duplicates([*group_by, "chain"]), group_by, plug_in_entropy)


def _kl(
    frame: pd.DataFrame,
    group_by: Sequence[str],
    manifest: Mapping[str, Any],
    reference: Mapping[str, float] | None = None,
    smoothing: float = 0.5
) -> pd.DataFrame:
    _require(frame, "kl", ["chain", "generation", "group", *group_by])
    names = list(manifest.get("groups") or sorted(frame["group"].unique()))
    chains = frame.drop_duplicates(["generation", "chain"])
    if reference is not None:
        target, source = _as_distribution(reference, names), "passed"
    elif manifest.get("reference") is not None:
        target, source = _as_distribution(manifest["reference"], names), "subject_prior"
    else:
        target, source = _final_third(chains, names, smoothing), "final_third"
    in_support = (target > 0).double()

    def divergence(part: pd.DataFrame) -> float:
        counts = _count_groups(part, names) + smoothing * in_support
        return information.kl_divergence((counts / counts.sum()).log(), target.log()).item()

    return _apply_by(chains, group_by, divergence).assign(reference=source)


MEASURES: dict[str, Callable[..., pd.DataFrame]] = {
    "accuracy": _accuracy,
    "fidelity": _fidelity,
    "entropy": _entropy,
    "surprisal": _surprisal,
    "chain_entropy": _chain_entropy,
    "kl": _kl
}


def _require(frame: pd.DataFrame, name: str, columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise MissingAbilityError(f"Measure '{name}' needs columns {missing}, which this run doesn't have.")


def _apply_by(
    frame: pd.DataFrame,
    group_by: Sequence[str],
    function: Callable[[pd.DataFrame], float]
) -> pd.DataFrame:
    if not group_by:
        return pd.DataFrame({"value": [function(frame)]})
    rows = []
    for key, part in frame.groupby(list(group_by), sort=True):
        rows.append({**dict(zip(group_by, key, strict=True)), "value": function(part)})
    return pd.DataFrame(rows)


def _mean_by(frame: pd.DataFrame, group_by: Sequence[str], values: np.ndarray) -> pd.DataFrame:
    with_values = frame.assign(value=values)
    return _apply_by(with_values, group_by, lambda part: float(part["value"].mean()))


def _stack(rows: pd.Series) -> torch.Tensor:
    width = max(len(row) for row in rows)
    stacked = np.full((len(rows), width), -math.inf, dtype=np.float64)
    for index, row in enumerate(rows):
        stacked[index, : len(row)] = row
    return torch.from_numpy(stacked)


def _as_distribution(values: Mapping[str, float], names: Sequence[str]) -> torch.Tensor:
    unknown = sorted(set(values) - set(names))
    if unknown:
        raise ValueError(f"Unknown groups {unknown} in the reference. This run has {list(names)}.")
    target = torch.tensor([values.get(name, 0.0) for name in names], dtype=torch.float64)
    if target.min() < 0 or target.sum() <= 0:
        raise ValueError("A reference must not be negative and must sum above 0.")
    return target / target.sum()


def _count_groups(part: pd.DataFrame, names: Sequence[str]) -> torch.Tensor:
    counts = part["group"].value_counts()
    return torch.tensor([float(counts.get(name, 0)) for name in names], dtype=torch.float64)


def _final_third(chains: pd.DataFrame, names: Sequence[str], smoothing: float) -> torch.Tensor:
    generations = int(chains["generation"].max()) + 1
    last = chains[chains["generation"] >= generations - math.ceil(generations / 3)]
    counts = _count_groups(last, names) + smoothing
    return counts / counts.sum()
