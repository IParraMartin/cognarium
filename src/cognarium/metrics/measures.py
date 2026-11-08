"""Metrics computed from a run's results table.

Each measure takes `(frame, group_by, manifest, **options)` and returns a table with
the `group_by` columns and a `value` column. Everything runs on CPU in float64, so
device limits never affect a reported number.
"""

from __future__ import annotations

import functools
import json
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
from cognarium.metrics import classification, detection, information, naming
from cognarium.tasks.mapping import Mapping as MappingTask

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
    frame = frame[frame["group"].notna()]

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
    frame = frame[frame["group"].notna()]
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


def _ib_complexity(frame: pd.DataFrame, group_by: Sequence[str], manifest: Mapping[str, Any]) -> pd.DataFrame:
    prior, _ = _meaning_space(_task_key(manifest))
    return _by_system(frame, manifest, "ib_complexity", group_by, lambda system: naming.ib_complexity(system, prior))


def _ib_accuracy(frame: pd.DataFrame, group_by: Sequence[str], manifest: Mapping[str, Any]) -> pd.DataFrame:
    prior, meanings = _meaning_space(_task_key(manifest))
    return _by_system(frame, manifest, "ib_accuracy", group_by, lambda system: naming.ib_accuracy(system, prior, meanings))


def _efficiency_loss(frame: pd.DataFrame, group_by: Sequence[str], manifest: Mapping[str, Any]) -> pd.DataFrame:
    key = _task_key(manifest)
    prior, meanings = _meaning_space(key)
    frontier = _frontier(key)
    return _by_system(
        frame,
        manifest,
        "efficiency_loss",
        group_by,
        lambda system: naming.efficiency_loss(system, prior, meanings, frontier)[0]
    )


def _ib_alignment(frame: pd.DataFrame, group_by: Sequence[str], manifest: Mapping[str, Any]) -> pd.DataFrame:
    key = _task_key(manifest)
    prior, meanings = _meaning_space(key)
    frontier = _frontier(key)

    def closeness(system: torch.Tensor) -> float:
        _, beta = naming.efficiency_loss(system, prior, meanings, frontier)
        return 1.0 - naming.gnid(system, frontier.encoder_at(beta), prior)

    return _by_system(frame, manifest, "ib_alignment", group_by, closeness)


def _alignment(
    frame: pd.DataFrame,
    group_by: Sequence[str],
    manifest: Mapping[str, Any],
    reference: Any = None
) -> pd.DataFrame:
    n_items = _item_count(frame, manifest)
    target = _reference_system(reference, n_items, "alignment")
    prior = _item_prior(manifest, n_items)
    return _by_system(frame, manifest, "alignment", group_by, lambda system: 1.0 - naming.gnid(system, target, prior))


def _co_naming_mse(
    frame: pd.DataFrame,
    group_by: Sequence[str],
    manifest: Mapping[str, Any],
    reference: Any = None
) -> pd.DataFrame:
    target = _reference_system(reference, _item_count(frame, manifest), "co_naming_mse")
    return _by_system(frame, manifest, "co_naming_mse", group_by, lambda system: naming.co_naming_mse(system, target))


def _numeric_mse(
    frame: pd.DataFrame,
    group_by: Sequence[str],
    manifest: Mapping[str, Any],
    reference: Any = None
) -> pd.DataFrame:
    values = _label_values(frame, "mse")
    target = _reference_values(reference, frame, manifest, "mse")
    return _by_system(frame, manifest, "mse", group_by, lambda system: ((system @ values - target) ** 2).mean().item())


def _correlation(
    frame: pd.DataFrame,
    group_by: Sequence[str],
    manifest: Mapping[str, Any],
    reference: Any = None
) -> pd.DataFrame:
    values = _label_values(frame, "correlation")
    target = _reference_values(reference, frame, manifest, "correlation")

    def pearson(system: torch.Tensor) -> float:
        answers = system @ values
        if answers.std() == 0 or target.std() == 0:
            return math.nan
        return torch.corrcoef(torch.stack([answers, target]))[0, 1].item()

    return _by_system(frame, manifest, "correlation", group_by, pearson)


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
    "vocab_entropy": _column_mean("vocab_entropy"),
    "ib_complexity": _ib_complexity,
    "ib_accuracy": _ib_accuracy,
    "efficiency_loss": _efficiency_loss,
    "ib_alignment": _ib_alignment,
    "alignment": _alignment,
    "co_naming_mse": _co_naming_mse,
    "mse": _numeric_mse,
    "correlation": _correlation
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


def _by_system(
    frame: pd.DataFrame,
    manifest: Mapping[str, Any],
    name: str,
    group_by: Sequence[str],
    score: Callable[[torch.Tensor], float]
) -> pd.DataFrame:
    """Scores each chain's naming system in each generation, then averages over chains in a group."""
    _require(frame, name, [
        "item",
        "options",
        "response",
        "generation",
        "chain",
        *group_by
    ])
    _refuse_free(frame)
    labels = [str(label) for label in frame["options"].iloc[0]]
    n_items = _item_count(frame, manifest)
    keys = list(dict.fromkeys([*group_by, "generation", "chain"]))
    rows = []
    for key, part in frame.groupby(keys, sort=True):
        counts = torch.zeros(n_items, len(labels), dtype=torch.float64)
        for item, response in zip(part["item"], part["response"], strict=True):
            counts[int(item), labels.index(str(response))] += 1
        named = counts.sum(dim=1, keepdim=True)
        if (named == 0).any():
            described = ", ".join(f"{column}={value}" for column, value in zip(keys, key, strict=True))
            raise ValueError(
                f"Measure '{name}' needs every item named, but for {described} only "
                f"{int((named > 0).sum())} of {n_items} were."
            )
        rows.append({**dict(zip(keys, key, strict=True)), "value": score(counts / named)})
    return _apply_by(pd.DataFrame(rows), group_by, lambda part: float(part["value"].mean()))


def _task_key(manifest: Mapping[str, Any]) -> str:
    return json.dumps(manifest.get("paradigm", {}).get("task", {}), sort_keys=True)


@functools.lru_cache(maxsize=8)
def _meaning_space(task: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Rebuilds the item prior and meanings from a run's task description."""
    config = json.loads(task)
    if "features" not in config or "labels" not in config:
        raise MissingAbilityError(
            "The bottleneck measures need a cg.tasks.Mapping task, whose items are described by numbers."
        )
    points, sigma, prior = MappingTask(
        config["features"],
        config["labels"],
        sigma=config.get("sigma"),
        prior=config.get("prior")
    ).meaning_space()
    return prior, naming.meanings(points, sigma)


@functools.lru_cache(maxsize=8)
def _frontier(task: str) -> naming.Frontier:
    return naming.ib_frontier(*_meaning_space(task))


def _item_count(frame: pd.DataFrame, manifest: Mapping[str, Any]) -> int:
    features = manifest.get("paradigm", {}).get("task", {}).get("features")
    return len(features) if features is not None else int(frame["item"].max()) + 1


def _item_prior(manifest: Mapping[str, Any], n_items: int) -> torch.Tensor:
    weights = manifest.get("paradigm", {}).get("task", {}).get("prior")
    prior = torch.ones(n_items, dtype=torch.float64) if weights is None else torch.tensor(weights, dtype=torch.float64)
    return prior / prior.sum()


def _reference_system(reference: Any, n_items: int, name: str) -> torch.Tensor:
    """Turns a reference labelling, or a table of label probabilities, into an encoder."""
    if reference is None:
        raise ValueError(
            f"Measure '{name}' compares each system with reference=: one label per item, "
            "or an items x labels table of probabilities."
        )
    array = np.asarray(reference.cpu() if isinstance(reference, torch.Tensor) else reference)
    if array.ndim == 2:
        table = torch.as_tensor(array.astype(np.float64))
        if len(table) != n_items or (table < 0).any() or (table.sum(dim=1) <= 0).any():
            raise ValueError(
                f"A reference table needs {n_items} rows of label probabilities, none negative, "
                f"got shape {tuple(table.shape)}."
            )
        return table / table.sum(dim=1, keepdim=True)
    labels = [str(label) for label in array.tolist()] if array.ndim == 1 else []
    if len(labels) != n_items:
        raise ValueError(f"A reference labelling needs one label for each of the {n_items} items, got {len(labels)}.")
    words = sorted(set(labels))
    return torch.tensor([[float(label == word) for word in words] for label in labels], dtype=torch.float64)


def _label_values(frame: pd.DataFrame, name: str) -> torch.Tensor:
    _require(frame, name, ["options"])
    labels = list(frame["options"].iloc[0])
    try:
        return torch.tensor([float(label) for label in labels], dtype=torch.float64)
    except (TypeError, ValueError):
        raise MissingAbilityError(f"Measure '{name}' needs labels that are numbers, got {labels}.") from None


def _reference_values(reference: Any, frame: pd.DataFrame, manifest: Mapping[str, Any], name: str) -> torch.Tensor:
    """Gives the reference answer for each item, from a list or a function of the item's features."""
    n_items = _item_count(frame, manifest)
    if reference is None:
        raise ValueError(
            f"Measure '{name}' compares answers with reference=: one number per item, "
            "or a function of an item's features."
        )
    if callable(reference):
        features = manifest.get("paradigm", {}).get("task", {}).get("features")
        if features is None:
            raise MissingAbilityError(f"Measure '{name}' needs a task that records its items' features.")
        reference = [reference(feature) for feature in features]
    values = torch.tensor([float(value) for value in reference], dtype=torch.float64)
    if len(values) != n_items:
        raise ValueError(f"The reference needs one number for each of the {n_items} items, got {len(values)}.")
    return values
