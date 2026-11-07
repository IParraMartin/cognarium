"""Metrics computed from a run's results table.

Each measure takes `(frame, group_by, manifest, **options)` and returns a table with
the `group_by` columns and a `value` column. Everything runs on CPU in float64, so
device limits never affect a reported number.
"""

from __future__ import annotations

import math
from collections.abc import (
    Callable,
    Iterable,
    Mapping,
    Sequence
)
from typing import Any

import numpy as np
import pandas as pd
import torch

from cognarium.errors import MissingAbilityError
from cognarium.metrics import classification, detection, information

_NO_PROBABILITIES = (
    "This measure reads probabilities over options, and free responses have none. "
    "Score the columns the paradigm parses from the text instead."
)


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
    _require(frame, "surprisal", [
        column,
        "options",
        "response",
        *group_by
    ])
    _refuse_free(frame)
    chosen = [
        float(log_probs[list(options).index(response)])
        for log_probs, options, response in zip(
            frame[column],
            frame["options"],
            frame["response"],
            strict=True
        )
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
    _require(frame, "kl", [
        "chain",
        "generation",
        "group",
        *group_by
    ])
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


def _label_measure(name: str, metric: Callable[..., float]) -> Callable[..., pd.DataFrame]:
    def measure(frame: pd.DataFrame, group_by: Sequence[str], manifest: Mapping[str, Any]) -> pd.DataFrame:
        scored, options = _answered(frame, name, ["response"], group_by)
        return _apply_by(scored, group_by, lambda part: metric(*_classes(part, options)))

    return measure


def _one_class_measure(name: str, metric: Callable[..., float]) -> Callable[..., pd.DataFrame]:
    def measure(
        frame: pd.DataFrame,
        group_by: Sequence[str],
        manifest: Mapping[str, Any],
        positive: str | None = None
    ) -> pd.DataFrame:
        scored, options = _answered(frame, name, ["response"], group_by)
        chosen = None if positive is None else options.index(_check_positive(positive, options))
        return _apply_by(scored, group_by, lambda part: metric(*_classes(part, options), positive=chosen))

    return measure


def _probability_measure(name: str, metric: Callable[..., float]) -> Callable[..., pd.DataFrame]:
    def measure(
        frame: pd.DataFrame,
        group_by: Sequence[str],
        manifest: Mapping[str, Any],
        raw: bool = False
    ) -> pd.DataFrame:
        column = "log_probs" if raw else "shaped_log_probs"
        scored, options = _answered(frame, name, [column], group_by)
        return _apply_by(scored, group_by, lambda part: metric(_probabilities(part, column), _answers(part, options)))

    return measure


def _ece(
    frame: pd.DataFrame,
    group_by: Sequence[str],
    manifest: Mapping[str, Any],
    raw: bool = False,
    n_bins: int = 15
) -> pd.DataFrame:
    column = "log_probs" if raw else "shaped_log_probs"
    scored, options = _answered(frame, "ece", [column], group_by)
    return _apply_by(
        scored,
        group_by,
        lambda part: classification.ece(
            _probabilities(part, column),
            _answers(part, options),
            n_bins=n_bins
        )
    )


def _cross_entropy(
    frame: pd.DataFrame,
    group_by: Sequence[str],
    manifest: Mapping[str, Any],
    raw: bool = False
) -> pd.DataFrame:
    column = "log_probs" if raw else "shaped_log_probs"
    scored, options = _answered(frame, "cross_entropy", [column], group_by)
    return _apply_by(
        scored,
        group_by,
        lambda part: information.surprisal(_answer_log_probs(part, column, options)).mean().item()
    )


def _perplexity(
    frame: pd.DataFrame,
    group_by: Sequence[str],
    manifest: Mapping[str, Any],
    raw: bool = False
) -> pd.DataFrame:
    column = "log_probs" if raw else "shaped_log_probs"
    scored, options = _answered(frame, "perplexity", [column], group_by)
    return _apply_by(
        scored,
        group_by,
        lambda part: information.perplexity(_answer_log_probs(part, column, options)).item()
    )


def _detection_measure(name: str, metric: Callable[..., float]) -> Callable[..., pd.DataFrame]:
    def measure(
        frame: pd.DataFrame,
        group_by: Sequence[str],
        manifest: Mapping[str, Any],
        positive: str | None = None
    ) -> pd.DataFrame:
        scored, options = _answered(frame, name, ["response"], group_by)
        signal = options[0] if positive is None else _check_positive(positive, options)

        def score(part: pd.DataFrame) -> float:
            is_signal = part["answer"] == signal
            said_signal = part["response"] == signal
            return metric(
                int((is_signal & said_signal).sum()),
                int((is_signal & ~said_signal).sum()),
                int((~is_signal & said_signal).sum()),
                int((~is_signal & ~said_signal).sum())
            )

        return _apply_by(scored, group_by, score)

    return measure


def _column_mean(name: str) -> Callable[..., pd.DataFrame]:
    def measure(frame: pd.DataFrame, group_by: Sequence[str], manifest: Mapping[str, Any]) -> pd.DataFrame:
        _require(frame, name, [name, *group_by])
        present = frame[frame[name].notna()]
        if present.empty:
            raise MissingAbilityError(f"Measure '{name}' has no values in this run.")
        return _mean_by(present, group_by, present[name].to_numpy(dtype=float))

    return measure


MEASURES: dict[str, Callable[..., pd.DataFrame]] = {
    "accuracy": _accuracy,
    "fidelity": _fidelity,
    "entropy": _entropy,
    "surprisal": _surprisal,
    "chain_entropy": _chain_entropy,
    "kl": _kl,
    "balanced_accuracy": _label_measure("balanced_accuracy", classification.balanced_accuracy),
    "precision": _one_class_measure("precision", classification.precision),
    "recall": _one_class_measure("recall", classification.recall),
    "f1": _one_class_measure("f1", classification.f1),
    "mcc": _label_measure("mcc", classification.mcc),
    "auroc": _probability_measure("auroc", classification.auroc),
    "auprc": _probability_measure("auprc", classification.auprc),
    "brier": _probability_measure("brier", classification.brier),
    "ece": _ece,
    "cross_entropy": _cross_entropy,
    "perplexity": _perplexity,
    "d_prime": _detection_measure("d_prime", detection.d_prime),
    "criterion": _detection_measure("criterion", detection.criterion),
    "a_prime": _detection_measure("a_prime", detection.a_prime),
    "logit_margin": _column_mean("logit_margin"),
    "logit_difference": _column_mean("logit_difference"),
    "target_rank": _column_mean("target_rank"),
    "vocab_entropy": _column_mean("vocab_entropy")
}


def _row_surprisal(frame: pd.DataFrame, raw: bool = False) -> np.ndarray:
    rows = _row_log_probs(frame, raw)
    return information.surprisal(rows[torch.arange(len(rows)), _positions(frame, "response")]).numpy()


def _row_answer_surprisal(frame: pd.DataFrame, raw: bool = False) -> np.ndarray:
    rows = _row_log_probs(frame, raw)
    return information.surprisal(rows[torch.arange(len(rows)), _answer_positions(frame, "answer_surprisal")]).numpy()


def _row_entropy(frame: pd.DataFrame, raw: bool = False) -> np.ndarray:
    return information.entropy(_row_log_probs(frame, raw)).numpy()


def _row_accuracy(frame: pd.DataFrame) -> np.ndarray:
    _check_answers(frame, "accuracy")
    return (frame["response"] == frame["answer"]).to_numpy(dtype=np.float64)


def _row_answer_probability(frame: pd.DataFrame, raw: bool = False) -> np.ndarray:
    rows = _row_log_probs(frame, raw)
    return rows[torch.arange(len(rows)), _answer_positions(frame, "answer_probability")].exp().numpy()


def _row_answer_margin(frame: pd.DataFrame, raw: bool = False) -> np.ndarray:
    rows = _row_log_probs(frame, raw)
    positions = _answer_positions(frame, "answer_margin")
    indices = torch.arange(len(rows))
    best_other = rows.clone()
    best_other[indices, positions] = -torch.inf
    return (rows[indices, positions] - best_other.max(dim=-1).values).numpy()


def _row_option_probability(frame: pd.DataFrame, option: str | int, raw: bool = False) -> np.ndarray:
    rows = _row_log_probs(frame, raw)
    positions = torch.tensor([_option_position(options, option) for options in frame["options"]])
    return rows[torch.arange(len(rows)), positions].exp().numpy()


def _row_expected_option(frame: pd.DataFrame, raw: bool = False) -> np.ndarray:
    rows = _row_log_probs(frame, raw)
    return (rows.exp() * torch.arange(rows.shape[1], dtype=torch.float64)).sum(dim=-1).numpy()


# one value per row, in bits for surprisal and entropy and in nats for the margin
ROW_MEASURES: dict[str, Callable[..., np.ndarray]] = {
    "surprisal": _row_surprisal,
    "answer_surprisal": _row_answer_surprisal,
    "entropy": _row_entropy,
    "accuracy": _row_accuracy,
    "answer_probability": _row_answer_probability,
    "answer_margin": _row_answer_margin,
    "option_probability": _row_option_probability,
    "expected_option": _row_expected_option
}

# what a measure stands in for, so that results never present it as that behaviour
PROXIES: dict[str, str] = {
    "surprisal": "reaction time",
    "answer_surprisal": "reaction time",
    "entropy": "reaction time",
    "answer_margin": "reaction time"
}


def _row_log_probs(frame: pd.DataFrame, raw: bool) -> torch.Tensor:
    return _stack(frame["log_probs" if raw else "shaped_log_probs"])


def _positions(frame: pd.DataFrame, column: str) -> torch.Tensor:
    _refuse_free(frame)
    return torch.tensor([
        list(options).index(chosen)
        for options, chosen in zip(frame["options"], frame[column], strict=True)
    ])


def _answer_positions(frame: pd.DataFrame, name: str) -> torch.Tensor:
    _check_answers(frame, name)
    return _positions(frame, "answer")


def _check_answers(frame: pd.DataFrame, name: str) -> None:
    if frame["answer"].isna().any():
        raise MissingAbilityError(f"Measure '{name}' needs a correct answer on every row. Filter with `where`.")


def _refuse_free(frame: pd.DataFrame) -> None:
    if any(len(options) == 0 for options in frame["options"]):
        raise MissingAbilityError(_NO_PROBABILITIES)


def _option_position(options: Sequence[str], option: str | int) -> int:
    if isinstance(option, str):
        if option not in options:
            raise ValueError(f"The option '{option}' isn't one of {list(options)}.")
        return list(options).index(option)
    if not 0 <= option < len(options):
        raise ValueError(f"There is no option at position {option} of {list(options)}.")
    return option


def _require(frame: pd.DataFrame, name: str, columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise MissingAbilityError(f"Measure '{name}' needs columns {missing}, which this run doesn't have.")


def _answered(
    frame: pd.DataFrame,
    name: str,
    columns: Iterable[str],
    group_by: Sequence[str]
) -> tuple[pd.DataFrame, list[str]]:
    _require(frame, name, [
        "answer",
        "options",
        *columns,
        *group_by
    ])
    _refuse_free(frame)
    scored = frame[frame["answer"].notna()]
    if scored.empty:
        raise MissingAbilityError(f"Measure '{name}' needs trials with a correct answer.")
    lists = {tuple(options) for options in scored["options"]}
    if len(lists) > 1:
        raise ValueError(f"Measure '{name}' needs one option list, but this run has {sorted(lists)}.")
    options = [str(option) for option in lists.pop()]
    outside = sorted(set(scored["answer"]) - set(options))
    if outside:
        raise ValueError(f"Measure '{name}' found answers {outside} outside the options {options}.")
    return scored, options


def _check_positive(positive: str, options: Sequence[str]) -> str:
    if positive not in options:
        raise ValueError(f"The positive option '{positive}' isn't one of {list(options)}.")
    return positive


def _answers(part: pd.DataFrame, options: Sequence[str]) -> torch.Tensor:
    return torch.tensor([options.index(answer) for answer in part["answer"]])


def _classes(part: pd.DataFrame, options: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
    predicted = torch.tensor([options.index(response) for response in part["response"]])
    return predicted, _answers(part, options)


def _probabilities(part: pd.DataFrame, column: str) -> torch.Tensor:
    # a log probability can round up to just above 0
    return _stack(part[column]).exp().clamp(0, 1)


def _answer_log_probs(part: pd.DataFrame, column: str, options: Sequence[str]) -> torch.Tensor:
    rows = _stack(part[column])
    return rows[torch.arange(len(rows)), _answers(part, options)]


def _apply_by(
    frame: pd.DataFrame,
    group_by: Sequence[str],
    function: Callable[[pd.DataFrame], float]
) -> pd.DataFrame:
    if not group_by:
        return pd.DataFrame({"value": [function(frame)]})
    rows = []
    for key, part in frame.groupby(list(group_by), sort=True):
        labels = dict(zip(group_by, key, strict=True))
        try:
            rows.append({**labels, "value": function(part)})
        except ValueError as error:
            described = ", ".join(f"{column}={value}" for column, value in labels.items())
            raise ValueError(f"For {described}: {error}") from error
    return pd.DataFrame(rows)


def _mean_by(frame: pd.DataFrame, group_by: Sequence[str], values: np.ndarray) -> pd.DataFrame:
    with_values = frame.assign(value=values)
    return _apply_by(with_values, group_by, lambda part: float(part["value"].mean()))


def _stack(rows: pd.Series) -> torch.Tensor:
    if any(len(row) == 0 for row in rows):
        raise MissingAbilityError(_NO_PROBABILITIES)
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
