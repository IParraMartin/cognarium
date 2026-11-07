"""The results of a run."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from safetensors.torch import load_file

from cognarium.errors import MissingAbilityError
from cognarium.metrics.measures import MEASURES


def step_path(root: Path, index: int) -> Path:
    """Gives where one step's results table is stored.

    Args:
        root: The run folder.
        index: The step.

    Returns:
        The path of the step's Parquet file.
    """
    return root / "steps" / f"step-{index:05d}.parquet"


def logits_path(root: Path, index: int) -> Path:
    """Gives where one step's logits are stored.

    Args:
        root: The run folder.
        index: The step.

    Returns:
        The path of the step's safetensors file.
    """
    return root / "logits" / f"step-{index:05d}.safetensors"


class Results:
    """The results of a run: one row per query, plus how they were made.

    Args:
        path: The run folder.
        frame: The results table.
        manifest: The run's settings, seeds and versions.

    Attributes:
        path: The run folder.
        manifest: The run's settings, seeds and versions.
    """

    def __init__(self, path: Path, frame: pd.DataFrame, manifest: dict[str, Any]) -> None:
        self.path = Path(path)
        self.manifest = manifest
        self._frame = frame

    @classmethod
    def load(cls, path: str | Path) -> Results:
        """Reopens a run from its folder.

        Args:
            path: The run folder.

        Returns:
            The results of every finished step. The table is empty if no step has
            finished yet.
        """
        root = Path(path)
        manifest = json.loads((root / "manifest.json").read_text())
        shards = [pd.read_parquet(shard) for shard in sorted((root / "steps").glob("step-*.parquet"))]
        frame = pd.concat(shards, ignore_index=True) if shards else pd.DataFrame()
        return cls(root, frame, manifest)

    def to_pandas(self) -> pd.DataFrame:
        """Gives the results table.

        Returns:
            A copy of the table, with one row per query.
        """
        return self._frame.copy()

    def logits(self, step: int) -> torch.Tensor:
        """Loads the full-vocabulary logits of one step.

        Args:
            step: The step.

        Returns:
            Raw logits of shape (n_queries, vocab).

        Raises:
            MissingAbilityError: If this step's logits weren't recorded.
        """
        path = logits_path(self.path, step)
        if not path.exists():
            raise MissingAbilityError(
                f"No logits were recorded for step {step}. Use a neural subject with record=['logits']."
            )
        return load_file(path)["logits"]

    def metrics(self, *names: str, **options: Any) -> pd.DataFrame:
        """Computes metrics from the results table.

        Args:
            *names: The metrics to compute: accuracy, fidelity, entropy, surprisal,
                chain_entropy or kl.
            **options: Passed to the metrics that accept them. `raw=True` makes
                entropy and surprisal use the distribution before temperature and
                top-p. `reference` and `smoothing` configure kl.

        Returns:
            A table with the paradigm's grouping columns, then `metric`, `value` and
            `reference`, which names what kl compared against.

        Raises:
            ValueError: If no names are given, or a name is unknown.
            TypeError: If an option isn't accepted by any requested metric.
            MissingAbilityError: If this run lacks the columns a metric needs.
        """
        unknown = [name for name in names if name not in MEASURES]
        if unknown or not names:
            raise ValueError(f"Choose metrics from {sorted(MEASURES)}. Got {list(names)}.")
        # the first three parameters are what every measure receives, not options
        accepted = {name: set(list(inspect.signature(MEASURES[name]).parameters)[3:]) for name in names}
        unused = set(options) - set().union(*accepted.values())
        if unused:
            raise TypeError(f"No requested metric accepts the options {sorted(unused)}.")
        group_by = list(self.manifest.get("group_by", ()))
        tables = []
        for name in names:
            chosen = {key: value for key, value in options.items() if key in accepted[name]}
            table = MEASURES[name](self._frame, group_by, self.manifest, **chosen)
            tables.append(table.assign(metric=name))
        combined = pd.concat(tables, ignore_index=True)
        if "reference" not in combined:
            combined["reference"] = None
        return combined[[*group_by, "metric", "value", "reference"]]
