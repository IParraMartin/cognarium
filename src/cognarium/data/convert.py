"""Datasets read from CSV, JSON and Parquet, which convert to arrays or trials."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from cognarium.stimuli import Stimulus, Trial

_METADATA_KEY = b"cognarium"
_ROLE_NAMES = frozenset({"text", "label", "features", "input_ids", "attention_mask"})
_OWN_ROWS = "your own rows"


class Dataset:
    """A table whose columns have roles in an experiment.

    Args:
        table: The rows.
        text: The column holding text stimuli.
        label: The column holding correct answers.
        features: The numeric columns forming a feature vector. A single column may
            hold lists.
        condition: The columns holding condition labels.
        item: The column identifying each item.
        options: The response options. Defaults to the sorted labels.
        vocabularies: Fixed orders for `label` and condition values. Values that aren't
            given here are sorted, so their numbering doesn't depend on row order.
        source: Where the rows came from.

    Attributes:
        text: The column holding text stimuli.
        label: The column holding correct answers.
        features: The numeric columns forming a feature vector.
        condition: The columns holding condition labels.
        item: The column identifying each item.
        options: The response options, if they were given.
        vocabularies: The order of the label values, under `label`, and of each condition
            column's values, under its name.
        source: Where the rows came from.

    Raises:
        ValueError: If there are no rows, a named column doesn't exist, or a cell is blank,
            isn't a number or isn't in its vocabulary. The message names the row and column.
    """

    def __init__(
        self,
        table: pd.DataFrame,
        *,
        text: str | None = None,
        label: str | None = None,
        features: str | Sequence[str] | None = None,
        condition: Sequence[str] = (),
        item: str | None = None,
        options: Sequence[str] | None = None,
        vocabularies: Mapping[str, Sequence[Any]] | None = None,
        source: str | None = None
    ) -> None:
        self.source = source
        described = "" if source is None else f" in {source}"
        if table.empty:
            raise ValueError(f"There are no rows{described}.")
        self._table = table.reset_index(drop=True).copy()
        self.text = text
        self.label = label
        self.features = [features] if isinstance(features, str) else list(features or ())
        self.condition = list(condition)
        self.item = item
        self.options = None if options is None else tuple(options)
        self._check_columns()
        self._features = self._feature_matrix()
        self.vocabularies = self._build_vocabularies(vocabularies or {})

    @property
    def columns(self) -> list[str]:
        """The names of the table's columns."""
        return list(self._table.columns)

    def __len__(self) -> int:
        return len(self._table)

    def to_pandas(self) -> pd.DataFrame:
        """Gives the rows as a table.

        Returns:
            A copy of the table.
        """
        return self._table.copy()

    def to_numpy(self) -> dict[str, np.ndarray]:
        """Converts the columns with roles to arrays.

        Returns:
            `text` as objects, `label` and each condition column as positions in their
            vocabularies, `features` as a float matrix, and `item`, each if the dataset
            has that role.
        """
        arrays: dict[str, np.ndarray] = {}
        if self.text is not None:
            arrays["text"] = self._table[self.text].to_numpy(dtype=object)
        if self.label is not None:
            arrays["label"] = _positions(self._labels(), self.vocabularies["label"])
        for column in self.condition:
            arrays[column] = _positions(self._table[column].tolist(), self.vocabularies[column])
        if self._features is not None:
            arrays["features"] = self._features
        if self.item is not None:
            arrays["item"] = self._table[self.item].to_numpy()
        return arrays

    def to_tensor(self, *, tokenizer: Any = None, max_length: int | None = None) -> dict[str, torch.Tensor]:
        """Converts the columns with roles to tensors.

        Args:
            tokenizer: Turns text into right-padded `input_ids` and `attention_mask`. It
                needs `encode(text)` and `pad_token_id`. Without one, text is left out.
            max_length: The most tokens a text may have.

        Returns:
            `label` and each condition column as positions in their vocabularies,
            `features` in float32, and any tokens.

        Raises:
            ValueError: If there is a tokenizer but no text, or a text is over `max_length`.
        """
        arrays = self.to_numpy()
        arrays.pop("text", None)
        arrays.pop("item", None)
        tensors = {name: torch.from_numpy(array) for name, array in arrays.items()}
        if "features" in tensors:
            tensors["features"] = tensors["features"].float()
        if tokenizer is not None:
            tensors.update(self._tokens(tokenizer, max_length))
        return tensors

    def to_parquet(self, path: str | Path) -> None:
        """Saves the rows, with their roles and vocabularies in the file's metadata.

        Args:
            path: Where to write the file.
        """
        roles = {
            "text": self.text,
            "label": self.label,
            "features": self.features,
            "condition": self.condition,
            "item": self.item,
            "options": None if self.options is None else list(self.options),
            "vocabularies": self.vocabularies,
            "source": self.source
        }
        table = pa.Table.from_pandas(self._table, preserve_index=False)
        table = table.replace_schema_metadata({
            **(table.schema.metadata or {}),
            _METADATA_KEY: json.dumps(roles).encode()
        })
        temporary = Path(f"{path}.tmp")
        pq.write_table(table, temporary)
        os.replace(temporary, path)

    @classmethod
    def from_parquet(cls, path: str | Path) -> Dataset:
        """Opens a Parquet file.

        Args:
            path: The file. One written by `to_parquet` keeps its roles.

        Returns:
            The dataset.
        """
        table = pq.read_table(path)
        roles = json.loads((table.schema.metadata or {}).get(_METADATA_KEY, b"{}"))
        return cls(
            table.to_pandas(),
            text=roles.get("text"),
            label=roles.get("label"),
            features=roles.get("features"),
            condition=roles.get("condition", ()),
            item=roles.get("item"),
            options=roles.get("options"),
            vocabularies=roles.get("vocabularies"),
            source=roles.get("source", Path(path).name)
        )

    def to_trials(self, *, options: Sequence[str] | None = None) -> list[Trial]:
        """Converts each row to a trial.

        Args:
            options: The response options. Defaults to the dataset's options, then to its
                sorted labels.

        Returns:
            One trial per row.

        Raises:
            ValueError: If the dataset has neither text nor features, there are no options,
                or a label isn't one of the options.
        """
        if self.text is None and self._features is None:
            raise ValueError("Trials need text or features. Name them when reading the file.")
        chosen = options if options is not None else self.options
        if chosen is None and self.label is None:
            raise ValueError("Trials need options. Pass options=, or name a label column.")
        chosen = tuple(chosen if chosen is not None else self.vocabularies["label"])
        labels = self._labels()
        if labels is not None:
            _check_labels(labels, chosen, self.label)
        trials = []
        for position in range(len(self)):
            features = None if self._features is None else torch.tensor(self._features[position], dtype=torch.float32)
            text = None if self.text is None else str(self._table[self.text].iloc[position])
            trials.append(
                Trial(
                    Stimulus(text=text, features=features),
                    options=chosen,
                    answer=None if labels is None else labels[position],
                    condition={column: self._table[column].tolist()[position] for column in self.condition},
                    item=None if self.item is None else self._table[self.item].tolist()[position]
                )
            )
        return trials

    def _check_columns(self) -> None:
        named = [
            self.text,
            self.label,
            *self.features,
            *self.condition,
            self.item
        ]
        for column in named:
            if column is not None and column not in self._table.columns:
                raise ValueError(f"There is no column '{column}'. This table has {list(self._table.columns)}.")
        colliding = sorted(_ROLE_NAMES & set(self.condition))
        if colliding:
            raise ValueError(f"The condition columns {colliding} would clash with converted arrays. Rename them.")
        for column in (
            self.text,
            self.label,
            *self.condition,
            self.item
        ):
            if column is not None:
                _check_filled(self._table, column)
        if self.label is not None and self.options is not None:
            _check_labels(self._labels(), self.options, self.label)

    def _labels(self) -> list[str] | None:
        return None if self.label is None else [str(value) for value in self._table[self.label].tolist()]

    def _feature_matrix(self) -> np.ndarray | None:
        if not self.features:
            return None
        rows = [_feature_row(self._table, position, self.features) for position in range(len(self))]
        for position, row in enumerate(rows):
            if len(row) != len(rows[0]):
                raise ValueError(f"Row {position} has {len(row)} features, but row 0 has {len(rows[0])}.")
        return np.array(rows, dtype=np.float64)

    def _build_vocabularies(self, given: Mapping[str, Sequence[Any]]) -> dict[str, list[Any]]:
        values = {column: self._table[column].tolist() for column in self.condition}
        if self.label is not None:
            values["label"] = self._labels()
        built = {}
        for name, cells in values.items():
            vocabulary = list(given[name]) if name in given else sorted(set(cells), key=str)
            for position, cell in enumerate(cells):
                if cell not in vocabulary:
                    raise ValueError(f"Row {position} has '{cell}', which isn't in the given vocabulary of {name}.")
            built[name] = vocabulary
        return built

    def _tokens(self, tokenizer: Any, max_length: int | None) -> dict[str, torch.Tensor]:
        if self.text is None:
            raise ValueError("Tokenizing needs a text column. Name it when reading the file.")
        encoded = [tokenizer.encode(str(text)) for text in self._table[self.text]]
        for position, tokens in enumerate(encoded):
            if max_length is not None and len(tokens) > max_length:
                raise ValueError(f"Row {position} has {len(tokens)} tokens, over max_length of {max_length}.")
        ids = [torch.tensor(tokens, dtype=torch.long) for tokens in encoded]
        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=tokenizer.pad_token_id),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                [torch.ones_like(row) for row in ids],
                batch_first=True
            )
        }


def from_csv(
    path: str | Path,
    *,
    text: str | None = None,
    label: str | None = None,
    features: str | Sequence[str] | None = None,
    condition: Sequence[str] = (),
    item: str | None = None,
    options: Sequence[str] | None = None,
    **read_options: Any
) -> Dataset:
    """Reads a CSV file.

    Only empty cells count as missing, so words like "null" and "NA" stay text.

    Args:
        path: The file, read as UTF-8 unless `encoding` says otherwise.
        text: The column holding text stimuli.
        label: The column holding correct answers.
        features: The numeric columns forming a feature vector.
        condition: The columns holding condition labels.
        item: The column identifying each item.
        options: The response options. Defaults to the sorted labels.
        **read_options: Passed to `pandas.read_csv`.

    Returns:
        The dataset.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file is empty or not UTF-8, or see `Dataset`.
    """
    path = Path(path)
    settings = {
        "encoding": "utf-8",
        "keep_default_na": False,
        "na_values": [""],
        **read_options
    }
    try:
        table = pd.read_csv(path, **settings)
    except UnicodeDecodeError as error:
        raise ValueError(f"{path.name} isn't UTF-8 text. Convert it, or pass its encoding.") from error
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"{path.name} is empty.") from error
    return Dataset(
        table,
        text=text,
        label=label,
        features=features,
        condition=condition,
        item=item,
        options=options,
        source=path.name
    )


def from_json(
    path: str | Path,
    *,
    records: str | None = None,
    text: str | None = None,
    label: str | None = None,
    features: str | Sequence[str] | None = None,
    condition: Sequence[str] = (),
    item: str | None = None,
    options: Sequence[str] | None = None
) -> Dataset:
    """Reads a JSON file: an array of rows, an object of columns, or one row per line.

    Args:
        path: The file, read as UTF-8. A `.jsonl` or `.ndjson` file holds one row per line.
        records: A dotted path to the list of rows, for a file that nests it.
        text: The column holding text stimuli. This and the other columns may be dotted
            paths into each row, such as `stim.sentence`. The column then takes the name
            after the last dot, unless that's already taken.
        label: The column holding correct answers.
        features: The numeric columns forming a feature vector. A single column may hold
            lists.
        condition: The columns holding condition labels.
        item: The column identifying each item.
        options: The response options. Defaults to the sorted labels.

    Returns:
        The dataset.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file isn't UTF-8 or valid JSON, `records` doesn't lead to a list,
            or see `Dataset`.
    """
    path = Path(path)
    table = _json_table(path, records)
    names = {}
    feature_list = [features] if isinstance(features, str) else list(features or ())
    for column in (
        text,
        label,
        *feature_list,
        *condition,
        item
    ):
        if column is not None and "." in column and column in table.columns:
            short = column.rsplit(".", 1)[-1]
            names[column] = column if short in table.columns else short
    table = table.rename(columns=names)
    rename = lambda column: names.get(column, column)
    return Dataset(
        table,
        text=None if text is None else rename(text),
        label=None if label is None else rename(label),
        features=[rename(column) for column in feature_list],
        condition=[rename(column) for column in condition],
        item=None if item is None else rename(item),
        options=options,
        source=path.name
    )


def read_rows(source: Any, columns: Sequence[str]) -> tuple[list[dict[str, Any]], str]:
    """Reads stimulus rows from anything a paradigm's `stimuli=` accepts.

    Args:
        source: A path to a `.csv`, `.json`, `.jsonl` or `.parquet` file, a `Dataset`, a
            table, or a list of mappings.
        columns: The columns every row must fill.

    Returns:
        The rows, and a label for where they came from.

    Raises:
        FileNotFoundError: If a file doesn't exist.
        ValueError: If `source` is of another kind, has no rows, or lacks a column or a
            value, which the message names.
    """
    table, label = _read_table(source)
    if table.empty:
        raise ValueError(f"The stimuli from {label} have no rows.")
    for column in columns:
        if column not in table.columns:
            raise ValueError(f"The stimuli from {label} need a column '{column}'. They have {list(table.columns)}.")
        _check_filled(table, column)
    return table.to_dict("records"), label


def _read_table(source: Any) -> tuple[pd.DataFrame, str]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        readers = {
            ".csv": from_csv,
            ".json": from_json,
            ".jsonl": from_json,
            ".ndjson": from_json,
            ".parquet": Dataset.from_parquet
        }
        if path.suffix not in readers:
            raise ValueError(f"stimuli can't read {path.suffix} files. Use one of {sorted(readers)}.")
        return readers[path.suffix](path).to_pandas(), path.name
    if isinstance(source, Dataset):
        return source.to_pandas(), source.source or _OWN_ROWS
    if isinstance(source, pd.DataFrame):
        return source.reset_index(drop=True), _OWN_ROWS
    if isinstance(source, Sequence) and all(isinstance(row, Mapping) for row in source):
        return pd.DataFrame(list(source)), _OWN_ROWS
    raise ValueError(f"stimuli must be a path, a Dataset, a table or a list of mappings, got {type(source).__name__}.")


def _check_filled(table: pd.DataFrame, column: str) -> None:
    for position, cell in enumerate(table[column].tolist()):
        if np.ndim(cell) == 0 and pd.isna(cell):
            raise ValueError(f"Row {position} has no value in '{column}'.")


def _check_labels(labels: Sequence[str], options: Sequence[str], column: str) -> None:
    for position, label in enumerate(labels):
        if label not in options:
            raise ValueError(f"Row {position} has '{label}' in '{column}', which isn't one of {list(options)}.")


def _feature_row(
    table: pd.DataFrame,
    position: int,
    columns: Sequence[str]
) -> list[float]:
    row = []
    for column in columns:
        cell = table[column].iloc[position]
        try:
            values = [float(value) for value in cell] if np.ndim(cell) > 0 else [float(cell)]
        except (TypeError, ValueError) as error:
            raise ValueError(f"Row {position} of '{column}' isn't a number: {cell!r}.") from error
        if not np.isfinite(values).all():
            raise ValueError(f"Row {position} has no finite value in '{column}'.")
        row.extend(values)
    return row


def _positions(values: Sequence[Any], vocabulary: Sequence[Any]) -> np.ndarray:
    lookup = {value: position for position, value in enumerate(vocabulary)}
    return np.array([lookup[value] for value in values], dtype=np.int64)


def _json_table(path: Path, records: str | None) -> pd.DataFrame:
    try:
        content = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path.name} isn't UTF-8 text. Convert it first.") from error
    if path.suffix in (".jsonl", ".ndjson"):
        return _rows_table(path, _json_lines(path, content))
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{path.name} isn't valid JSON at line {error.lineno}: {error.msg}. "
            "If it holds one object per line, name it .jsonl."
        ) from error
    if records is not None:
        return _rows_table(path, _follow(parsed, records))
    if isinstance(parsed, list):
        return _rows_table(path, parsed)
    if isinstance(parsed, dict) and parsed and all(isinstance(value, list) for value in parsed.values()):
        return pd.DataFrame(parsed)
    raise ValueError(f"{path.name} doesn't hold a list of rows at the top. Pass records= with the path to one.")


def _json_lines(path: Path, content: str) -> list[Any]:
    rows = []
    for number, line in enumerate(content.splitlines(), start=1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path.name} isn't valid JSON at line {number}: {error.msg}.") from error
    return rows


def _follow(parsed: Any, records: str) -> list[Any]:
    current = parsed
    for segment in records.split("."):
        if not isinstance(current, dict):
            raise ValueError(f"records='{records}' doesn't lead to a list of rows.")
        if segment not in current:
            raise ValueError(f"records='{records}' fails at '{segment}'. The keys there are {sorted(current)}.")
        current = current[segment]
    if not isinstance(current, list):
        raise ValueError(f"records='{records}' doesn't lead to a list of rows.")
    return current


def _rows_table(path: Path, rows: list[Any]) -> pd.DataFrame:
    if not rows:
        raise ValueError(f"There are no rows in {path.name}.")
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Row {position} of {path.name} isn't an object with named columns.")
    return pd.json_normalize(rows)
