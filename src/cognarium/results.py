"""The results of a run."""

from __future__ import annotations

import inspect
import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import torch
from safetensors.torch import load_file

from cognarium.errors import MissingAbilityError
from cognarium.metrics.measures import MEASURES
from cognarium.recording.lens import read_layers
from cognarium.recording.store import HiddenStates

if TYPE_CHECKING:
    from cognarium.subjects.base import Subject

_TABLE_COLUMNS = ("metric", "value", "reference")


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
        subject: The subject that made these results, if it's still around.

    Attributes:
        path: The run folder.
        manifest: The run's settings, seeds and versions.
        subject: The subject that made these results, or None.
    """

    def __init__(
        self,
        path: Path,
        frame: pd.DataFrame,
        manifest: dict[str, Any],
        *,
        subject: Subject | None = None
    ) -> None:
        self.path = Path(path)
        self.manifest = manifest
        self.subject = subject
        self._frame = frame
        self._hidden: HiddenStates | None = None

    @classmethod
    def load(cls, path: str | Path, *, subject: Subject | None = None) -> Results:
        """Reopens a run from its folder.

        Args:
            path: The run folder.
            subject: The subject that made these results, if you have it.

        Returns:
            The results of every finished step. The table is empty if no step has
            finished yet.
        """
        root = Path(path)
        manifest = json.loads((root / "manifest.json").read_text())
        shards = [pd.read_parquet(shard) for shard in sorted((root / "steps").glob("step-*.parquet"))]
        frame = pd.concat(shards, ignore_index=True) if shards else pd.DataFrame()
        return cls(root, frame, manifest, subject=subject)

    @property
    def hidden(self) -> HiddenStates:
        """The run's hidden states, read lazily from their shards.

        Raises:
            MissingAbilityError: If this run didn't record hidden states.
        """
        if self._hidden is None:
            self._hidden = HiddenStates.open(self.path)
        return self._hidden

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

    def lens(
        self,
        subject: Subject | None = None,
        *,
        layers: Sequence[str] | None = None,
        batch_size: int = 256
    ) -> pd.DataFrame:
        """Reads each layer's states as if they had reached the output head.

        This is the logit lens. It runs each recorded state through the subject's final
        norm and output head, on the subject's device. Options are represented by their
        first token, so measures of multi-token options are approximate.

        Args:
            subject: The subject that made this run. Only needed after `Results.load`.
            layers: The layers to read. Defaults to every recorded layer.
            batch_size: How many states go through the head at once.

        Returns:
            One row per query per layer, with the columns `step`, `query`, which is the
            row's position in its step, `item`, the paradigm's grouping columns, `layer`,
            `top_token`, `top_log_prob`, `entropy`, which is in bits over the whole
            vocabulary, `option_log_probs`, which are normalised over the options,
            `logit_margin`, `target_rank` and `logit_difference`.

        Raises:
            MissingAbilityError: If this run didn't record hidden states, its learners
                were discarded because it trained weights, no subject is available, or
                the subject has no output head or tokens.
            ValueError: If the states weren't recorded with `pool="last"`, or `subject`
                isn't the model this run used.
        """
        hidden = self.hidden
        if hidden.pool != "last":
            raise ValueError(f"The lens reads one position per query, so it needs pool='last', not {hidden.pool!r}.")
        if self.manifest.get("learning") == "weights":
            raise MissingAbilityError(
                "The lens needs the model that made each state, but learning='weights' discards "
                "each chain's learner after its generation."
            )
        reader = self._lens_subject(subject)
        readout = reader.readout()
        if readout is None or reader.first_tokens([]) is None:
            raise MissingAbilityError(f"{type(reader).__name__} has no output head and tokens for the lens to read.")
        return read_layers(
            hidden,
            readout,
            reader.first_tokens,
            group_by=self.manifest.get("group_by", ()),
            layers=hidden.layers if layers is None else layers,
            batch_size=batch_size
        )

    def metrics(self, *names: str, **options: Any) -> pd.DataFrame:
        """Computes metrics from the results table.

        Args:
            *names: The metrics to compute. From the responses and answers: accuracy,
                balanced_accuracy, precision, recall, f1, mcc, d_prime, criterion or
                a_prime. From the probabilities: entropy, surprisal, auroc, auprc,
                brier, ece, cross_entropy or perplexity. From iterated learning:
                fidelity, chain_entropy or kl. With recorded logits: logit_margin,
                logit_difference, target_rank or vocab_entropy.
            **options: Passed to the metrics that accept them. `raw=True` makes a
                probability metric use the distribution before temperature and top-p.
                `positive` names the option treated as the positive class. It defaults
                to the macro average for precision, recall and f1, and to the first
                option for the detection metrics. `n_bins` configures ece, and
                `reference` and `smoothing` configure kl.

        Returns:
            A table with the paradigm's grouping columns, then `metric`, `value` and
            `reference`, which names what kl compared against.

        Raises:
            ValueError: If no names are given, a name is unknown, or a metric is
                undefined for a group, which the message names.
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
            table = MEASURES[name](
                self._frame,
                group_by,
                self.manifest,
                **chosen
            )
            tables.append(table.assign(metric=name))
        combined = pd.concat(tables, ignore_index=True)
        if "reference" not in combined:
            combined["reference"] = None
        return combined[[*group_by, *_TABLE_COLUMNS]]

    def _lens_subject(self, subject: Subject | None) -> Subject:
        if subject is None and self.subject is None:
            raise MissingAbilityError("This run was reopened from disk, so pass its subject: results.lens(subject).")
        if subject is None or subject is self.subject:
            return self.subject
        if subject.readout() is None:
            return subject
        current = json.loads(json.dumps(subject.config()))
        if current.get("pretrained") is False:
            raise ValueError(
                "This subject has untrained weights, which aren't part of a run's identity, so "
                "there's no way to check they are the ones this run used."
            )
        stored = self.manifest["subject"]
        differing = sorted(key for key in stored.keys() | current.keys() if stored.get(key) != current.get(key))
        if differing:
            raise ValueError(f"This subject isn't the one this run used. These differ: {differing}.")
        return subject
