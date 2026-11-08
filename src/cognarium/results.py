"""The results of a run."""

from __future__ import annotations

import inspect
import json
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file

from cognarium.errors import MissingAbilityError
from cognarium.metrics.comparison import item_correlation
from cognarium.metrics.measures import MEASURES, PROXIES, ROW_MEASURES
from cognarium.metrics.statistics import contrast_table
from cognarium.recording.lens import read_layers
from cognarium.recording.store import HiddenStates

if TYPE_CHECKING:
    from cognarium.paradigms.base import Paradigm
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
        paradigm: The paradigm that made these results, if it's still around.

    Attributes:
        path: The run folder.
        manifest: The run's settings, seeds and versions.
        subject: The subject that made these results, or None.
        paradigm: The paradigm that made these results, or None.
    """

    def __init__(
        self,
        path: Path,
        frame: pd.DataFrame,
        manifest: dict[str, Any],
        *,
        subject: Subject | None = None,
        paradigm: Paradigm | None = None
    ) -> None:
        self.path = Path(path)
        self.manifest = manifest
        self.subject = subject
        self.paradigm = paradigm
        self._frame = frame
        self._hidden: HiddenStates | None = None

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        subject: Subject | None = None,
        paradigm: Paradigm | None = None
    ) -> Results:
        """Reopens a run from its folder.

        Args:
            path: The run folder.
            subject: The subject that made these results, if you have it.
            paradigm: The paradigm that made these results, which `score` needs.

        Returns:
            The results of every finished step. The table is empty if no step has
            finished yet.
        """
        root = Path(path)
        manifest = json.loads((root / "manifest.json").read_text())
        shards = [pd.read_parquet(shard) for shard in sorted((root / "steps").glob("step-*.parquet"))]
        frame = pd.concat(shards, ignore_index=True) if shards else pd.DataFrame()
        return cls(
            root,
            frame,
            manifest,
            subject=subject,
            paradigm=paradigm
        )

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

    def metrics(self, *names: str, by: Sequence[str] | None = None, **options: Any) -> pd.DataFrame:
        """Computes metrics from the results table.

        Args:
            *names: The metrics to compute. From the responses and answers: accuracy,
                balanced_accuracy, precision, recall, f1, mcc, d_prime, criterion or
                a_prime. From the probabilities: answer_probability,
                option_probability, entropy, surprisal, auroc, auprc, brier, ece,
                cross_entropy or perplexity. From iterated learning: fidelity,
                chain_entropy or kl. With recorded logits: logit_margin,
                logit_difference, target_rank or vocab_entropy. From naming systems,
                where a system is one chain's answers in a generation of iterated
                learning, or else one participant's answers in a group, one label
                per item: ib_complexity, ib_accuracy, efficiency_loss and
                ib_alignment, which need a `cg.tasks.Mapping` with numeric features,
                and alignment and co_naming_mse, which compare with `reference`.
                With numeric labels, as in function learning: mse and correlation
                against `reference`. These score each system, then average over the
                systems in a group.
            by: A column or columns to group by, in place of the paradigm's own. With
                `["generation", "chain"]` the naming measures give one value per
                chain.
            **options: Passed to the metrics that accept them. `raw=True` makes a
                probability metric use the distribution before temperature and top-p.
                `positive` names the option treated as the positive class. It defaults
                to the macro average for precision, recall and f1, and to the first
                option for the detection metrics. `option` names the option whose
                probability option_probability averages. `n_bins` configures ece, and
                `reference` and `smoothing` configure kl. For alignment and
                co_naming_mse, `reference` is one label per item or an items x labels
                table of probabilities. For mse and correlation, it is one number per
                item or a function of an item's features. auprc and ece see every
                option's probability, so auprc is the macro average over the options
                and ece scores the chosen option, even with two options.

        Returns:
            A table with the grouping columns, then `metric`, `value` and
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
        if isinstance(by, str):
            by = [by]
        group_by = list(self.manifest.get("group_by", ()) if by is None else by)
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

    def contrast(
        self,
        measure: str,
        by: str | None = None,
        *,
        levels: Sequence[Any] | None = None,
        against: float | None = None,
        where: Mapping[str, Any] | None = None,
        ci_by: str = "item",
        n_resamples: int = 2000,
        level: float = 0.95,
        seed: int = 0,
        **options: Any
    ) -> pd.DataFrame:
        """Compares two levels of a condition, or one level with a fixed value.

        Args:
            measure: `surprisal`, `answer_surprisal`, `entropy`, `accuracy`,
                `answer_probability`, `answer_margin`, `option_probability`,
                `expected_option`, or any numeric column of the results table.
            by: The column naming the levels.
            levels: The levels to compare. `level_a` minus `level_b` is reported.
            against: Compares with this value rather than a second level.
            where: Keeps only the rows whose columns equal these values.
            ci_by: The column whose groups the interval resamples.
            n_resamples: How many resamples the interval and the p value each use.
            level: The interval's coverage, such as 0.95.
            seed: Seeds the resampling.
            **options: Passed to the measure, such as `option` for `option_probability`.
                `raw=True` uses the distribution before temperature, top-k and top-p.

        Returns:
            One row, with the columns of `cognarium.metrics.contrast_table`, then
            `measure` and `proxy_for`, which names the human behaviour a proxy measure
            stands in for, and is None for any other measure.

        Raises:
            ValueError: If the measure is unknown, no rows match `where`, a value isn't
                finite, participants are identical and `ci_by` is `participant`, or the
                comparison can't be made.
            MissingAbilityError: If a measure needs answers that some rows lack.
        """
        if ci_by == "participant" and self.manifest["paradigm"].get("participants_identical"):
            raise ValueError(
                "Every participant is identical in this run, so an interval over participants would "
                "have zero width. Use ci_by='item', or give participants context or personas."
            )
        rows = _matching(self._frame, where)
        values = _row_values(rows, measure, options)
        table = contrast_table(
            rows.assign(contrasted=values),
            "contrasted",
            by=by,
            levels=levels,
            against=against,
            group=ci_by,
            n_resamples=n_resamples,
            level=level,
            seed=seed
        )
        return table.assign(measure=measure, proxy_for=PROXIES.get(measure))

    def compare_to_human(
        self,
        human: pd.DataFrame,
        measure: str,
        *,
        on: str = "item",
        value: str = "value",
        where: Mapping[str, Any] | None = None,
        method: str = "pearson",
        ceiling: float | None = None,
        **options: Any
    ) -> pd.DataFrame:
        """Correlates the model's item means of a measure with people's.

        The measure is averaged over every row of each `on` level, and so is the human
        table's `value`. Only levels on both sides are compared, and a warning counts
        the ones dropped.

        Args:
            human: People's data, with one or more rows per level of `on`.
            measure: A measure that `contrast` takes, or a numeric column.
            on: The column both tables share, such as `item`.
            value: The human table's column of values.
            where: Keeps only the result rows whose columns equal these values.
            method: `pearson` or `spearman`.
            ceiling: A noise ceiling to report beside the correlation, such as
                `cognarium.metrics.split_half_ceiling(ratings)["ceiling"]`.
            **options: Passed to the measure, such as `option` for `option_probability`.

        Returns:
            One row with `measure`, `correlation`, its interval over items as `ci_low`
            and `ci_high`, `n`, the number of shared items, `ceiling`, and `proxy_for`,
            which names the human behaviour a proxy measure stands in for and is None
            for any other measure.

        Raises:
            ValueError: If a column is missing, the human values aren't numbers, fewer
                than 3 items are shared, or either side is constant.
        """
        for column in (on, value):
            if column not in human.columns:
                raise ValueError(f"The human table needs a column '{column}'. It has {list(human.columns)}.")
        if not pd.api.types.is_numeric_dtype(human[value]):
            raise ValueError(f"The human column '{value}' must hold numbers.")
        rows = _matching(self._frame, where)
        if on not in rows.columns:
            raise ValueError(f"The results have no column '{on}'. They have {list(rows.columns)}.")
        model = rows.assign(compared=_row_values(rows, measure, options)).groupby(on)["compared"].mean()
        people = human.groupby(on)[value].mean()
        shared = [level for level in model.index if level in people.index]
        only_model, only_human = len(model) - len(shared), len(people) - len(shared)
        if only_model or only_human:
            warnings.warn(
                f"{only_model} items are only in the results and {only_human} items are only in the human "
                "table, so they were dropped.",
                UserWarning,
                stacklevel=2
            )
        if len(shared) < 3:
            raise ValueError(
                f"A correlation needs at least 3 items in both tables, and {len(shared)} are shared. Check "
                f"that '{on}' holds the same values, of the same type, in both."
            )
        found = item_correlation(model[shared].to_numpy(), people[shared].to_numpy(), method=method)
        return pd.DataFrame([
            {
                "measure": measure,
                "correlation": found["correlation"],
                "ci_low": found["ci_low"],
                "ci_high": found["ci_high"],
                "n": found["n"],
                "ceiling": ceiling,
                "proxy_for": PROXIES.get(measure)
            }
        ])

    def score(self) -> pd.DataFrame:
        """Reports the effects the paradigm declares.

        Returns:
            One row per effect, with the primary effect first.

        Raises:
            MissingAbilityError: If this run was reopened without its paradigm, or the
                paradigm declares no effects.
        """
        if self.paradigm is None:
            raise MissingAbilityError(
                "This run was reopened from disk, so pass its paradigm: cg.Results.load(path, paradigm=...)."
            )
        return self.paradigm.score(self)

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


def _matching(frame: pd.DataFrame, where: Mapping[str, Any] | None) -> pd.DataFrame:
    rows = frame
    for column, value in (where or {}).items():
        if column not in frame.columns:
            raise ValueError(f"There is no column '{column}'. The table has {list(frame.columns)}.")
        rows = rows[rows[column] == value]
    if rows.empty:
        raise ValueError(f"There are no rows where {dict(where or {})}.")
    return rows


def _row_values(rows: pd.DataFrame, measure: str, options: Mapping[str, Any]) -> np.ndarray:
    if measure in ROW_MEASURES:
        values = ROW_MEASURES[measure](rows, **options)
    elif measure in rows.columns and pd.api.types.is_numeric_dtype(rows[measure]):
        values = rows[measure].to_numpy(dtype=np.float64)
    else:
        raise ValueError(f"Choose a measure from {sorted(ROW_MEASURES)} or a numeric column, got '{measure}'.")
    if not np.isfinite(values).all() and not options.get("raw", False):
        raise ValueError(
            f"Some values of '{measure}' aren't finite, since top-k or top-p removed an option "
            "they read. Use raw=True for the distribution before those settings."
        )
    return values
