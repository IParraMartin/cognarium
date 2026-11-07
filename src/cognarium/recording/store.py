"""Storing hidden states on disk, and reading them back.

States are written as safetensors shards, one folder per layer, by a background thread
behind a bounded queue, so a long run's memory use stays flat. Each shard holds the keys
`states` and `queries`, plus `lengths` when every token was kept.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import tempfile
import threading
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from cognarium.errors import MissingAbilityError, RunConflictError
from cognarium.metrics.geometry import layer_matrix

_DTYPES = ("float16", "bfloat16", "float32", "float64")
_SHARED_SECTIONS = ("paradigm", "subject", "learning", "versions")


class StateWriter:
    """Writes one step's states to shards from a background thread.

    Conversion and checks happen in the calling thread, so a bad value is reported at
    the `write` that caused it. A failure in the background thread is raised by the
    next `write` or by `finish`, and the queue keeps draining meanwhile, so a full queue
    can't hang the caller. Use it as a context manager, which always stops the thread.

    Args:
        root: The folder that holds the `states` folder.
        step: The step these states belong to, which goes in each shard's name.
        dtype: The precision to store: `float16`, `bfloat16`, `float32` or `float64`.
        rows_per_shard: How many rows go in a shard. Boundaries depend only on row
            counts, so a rerun of the step overwrites the same names.
        max_pending: How many finished shards may wait for the disk. A write blocks
            once this many are waiting.

    Raises:
        ValueError: If `dtype` isn't allowed.
    """

    def __init__(
        self,
        root: Path,
        step: int,
        *,
        dtype: str,
        rows_per_shard: int = 4096,
        max_pending: int = 8
    ) -> None:
        if dtype not in _DTYPES:
            raise ValueError(f"Unknown state dtype '{dtype}'. Choose from {list(_DTYPES)}.")
        self._root = Path(root)
        self._step = step
        self._dtype_name = dtype
        self._dtype = getattr(torch, dtype)
        self._rows_per_shard = rows_per_shard
        self._buffers: dict[str, _LayerBuffer] = {}
        self._queue: queue.Queue = queue.Queue(maxsize=max_pending)
        self._error: BaseException | None = None
        self._stopping = threading.Event()
        self._closed = False
        self._thread = threading.Thread(target=self._work, daemon=True)
        self._thread.start()

    @property
    def layers(self) -> tuple[str, ...]:
        """The layers written so far."""
        return tuple(self._buffers)

    def write(
        self,
        layer: str,
        states: torch.Tensor,
        queries: Sequence[int],
        lengths: torch.Tensor | None = None
    ) -> None:
        """Adds rows to a layer.

        Args:
            layer: The layer's name.
            states: One row per query. With `lengths`, its shape is (rows, time, ...).
            queries: The position of each row in the step.
            lengths: Each row's token count, when every token is kept.

        Raises:
            RuntimeError: If this writer has already finished.
            ValueError: If a value isn't finite after conversion, the rows' shape is
                empty or differs from earlier rows, or `queries` doesn't match them.
            OSError: If the background thread failed to write an earlier shard. Any
                other error from that thread is raised the same way.
        """
        self._raise_if_failed()
        if self._closed:
            raise RuntimeError("This StateWriter has already called finish, so it can't take more rows.")
        converted = states.detach().cpu().to(self._dtype)
        self._check_finite(layer, states, converted)
        queries = list(queries)
        if len(queries) != len(converted):
            raise ValueError(f"Layer '{layer}' got {len(converted)} rows but {len(queries)} queries.")
        buffer = self._buffers.setdefault(layer, _LayerBuffer(layer))
        buffer.add(converted, queries, None if lengths is None else lengths.detach().cpu().long())
        while buffer.rows >= self._rows_per_shard:
            self._send_shard(layer, buffer, self._rows_per_shard)

    def finish(self, n_queries: int) -> dict[int, tuple[str, int]]:
        """Writes what's left, and waits for everything to reach the disk.

        Args:
            n_queries: How many queries the step has. Every layer must hold each one
                exactly once, in the same order.

        Returns:
            Where each query's states are: its shard's name, and its row in that shard.

        Raises:
            RuntimeError: If nothing was written, or a layer's queries are incomplete
                or differ from another layer's.
            OSError: If the background thread failed to write a shard. Any other error
                from that thread is raised the same way.
        """
        self._raise_if_failed()
        order = self._check_queries(n_queries)
        for layer, buffer in self._buffers.items():
            if buffer.rows > 0:
                self._send_shard(layer, buffer, buffer.rows)
        self._close()
        self._raise_if_failed()
        return {
            query: (self._shard_name(position // self._rows_per_shard), position % self._rows_per_shard)
            for position, query in enumerate(order)
        }

    def __enter__(self) -> StateWriter:
        return self

    def __exit__(self, *exception: Any) -> None:
        self._stopping.set()
        self._close()

    def _shard_name(self, shard: int) -> str:
        return f"step-{self._step:05d}-{shard:05d}"

    def _check_finite(self, layer: str, states: torch.Tensor, converted: torch.Tensor) -> None:
        if converted.isfinite().all():
            return
        if not states.isfinite().all():
            raise ValueError(f"Layer '{layer}' produced values that aren't finite.")
        wider = "float32" if self._dtype_name == "float16" else "float64"
        raise ValueError(f"Layer '{layer}' overflowed {self._dtype_name}. Use state_dtype='{wider}'.")

    def _check_queries(self, n_queries: int) -> list[int]:
        if not self._buffers:
            raise RuntimeError(f"No states were recorded for the {n_queries} queries of this step.")
        expected = list(range(n_queries))
        first = None
        for layer, buffer in self._buffers.items():
            if sorted(buffer.queries) != expected:
                raise RuntimeError(
                    f"Layer '{layer}' should hold each of {n_queries} queries once, but it holds "
                    f"{len(buffer.queries)} rows."
                )
            first = buffer.queries if first is None else first
            if buffer.queries != first:
                raise RuntimeError(f"Layer '{layer}' wrote its queries in a different order from the first layer.")
        return first

    def _send_shard(self, layer: str, buffer: _LayerBuffer, count: int) -> None:
        name = self._shard_name(buffer.shards)
        path = self._root / "states" / layer / f"{name}.safetensors"
        self._queue.put((path, buffer.cut(count)))

    def _work(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            # keeps draining after a failure, so that a blocked write always wakes up
            if self._error is not None or self._stopping.is_set():
                continue
            try:
                _save_shard(*item)
            except BaseException as error:
                self._error = error

    def _close(self) -> None:
        if not self._closed:
            self._closed = True
            self._queue.put(None)
        self._thread.join()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise self._error


class _LayerBuffer:
    """The rows of one layer that haven't reached a shard yet."""

    def __init__(self, layer: str) -> None:
        self.layer = layer
        self.rows = 0
        self.shards = 0
        self.queries: list[int] = []
        self._pieces: deque[tuple[torch.Tensor, torch.Tensor | None]] = deque()
        self._waiting: deque[int] = deque()
        self._shape: torch.Size | None = None
        self._has_lengths: bool | None = None

    def add(
        self,
        states: torch.Tensor,
        queries: list[int],
        lengths: torch.Tensor | None
    ) -> None:
        shape = states.shape[1:] if lengths is None else states.shape[2:]
        if 0 in states.shape[1:]:
            raise ValueError(
                f"Layer '{self.layer}' has nothing to store: its rows have shape {tuple(states.shape[1:])}. "
                "A learner with no context has no states."
            )
        if self._shape is not None and (shape != self._shape or (lengths is not None) != self._has_lengths):
            raise ValueError(
                f"Layer '{self.layer}' changed shape, from {tuple(self._shape)} to {tuple(shape)}. "
                "All of a layer's rows must match."
            )
        self._shape, self._has_lengths = shape, lengths is not None
        self.rows += len(states)
        self.queries.extend(queries)
        self._waiting.extend(queries)
        self._pieces.append((states, lengths))

    def cut(self, count: int) -> dict[str, torch.Tensor]:
        states, lengths = [], []
        remaining = count
        while remaining > 0:
            piece, piece_lengths = self._pieces[0]
            taken = min(remaining, len(piece))
            states.append(piece[:taken])
            if piece_lengths is not None:
                lengths.append(piece_lengths[:taken])
            if taken == len(piece):
                self._pieces.popleft()
            else:
                rest = None if piece_lengths is None else piece_lengths[taken:]
                self._pieces[0] = (piece[taken:], rest)
            remaining -= taken
        self.rows -= count
        self.shards += 1
        queries = [self._waiting.popleft() for _ in range(count)]
        tensors = {"queries": torch.tensor(queries, dtype=torch.long)}
        if lengths:
            tensors["lengths"] = torch.cat(lengths)
            tensors["states"] = _pad_time(states, int(tensors["lengths"].max()))
        else:
            tensors["states"] = torch.cat(states)
        return tensors


class HiddenStates:
    """The hidden states of a run, read lazily from their shards.

    Opening reads no shard. `trials` has one row per recorded query, in order, and every
    method that takes `where` returns its rows in that order.

    Args:
        path: The folder that holds the states.
        manifest: The folder's manifest.
        trials: The table of trials, with `state_shard` and `state_offset` columns.

    Attributes:
        path: The folder that holds the states.
        manifest: The folder's manifest.
        trials: The table of trials.
        layers: The names of the recorded layers.
        pool: How the time dimension was reduced: `"last"`, `"mean"` or None.
        dtype: The stored precision, such as `"float16"`.
    """

    def __init__(
        self,
        path: Path,
        manifest: dict[str, Any],
        trials: pd.DataFrame
    ) -> None:
        settings = manifest.get("states", manifest)
        self.path = Path(path)
        self.manifest = manifest
        self.trials = trials
        self.layers = tuple(settings["layers"])
        self.pool = settings["pool"]
        self.dtype = settings["dtype"]

    @classmethod
    def open(cls, path: str | Path) -> HiddenStates:
        """Opens a saved store, or the states of a run folder.

        Args:
            path: The folder.

        Returns:
            The states. No shard is read yet.

        Raises:
            MissingAbilityError: If the folder is a run that didn't record states.
        """
        root = Path(path)
        manifest = json.loads((root / "manifest.json").read_text())
        if (root / "trials.parquet").exists():
            return cls(root, manifest, pd.read_parquet(root / "trials.parquet"))
        if "states" not in manifest:
            raise MissingAbilityError(f"{root} didn't record states. Run it with record=['hidden_states'].")
        shards = [pd.read_parquet(shard) for shard in sorted((root / "steps").glob("step-*.parquet"))]
        return cls(root, manifest, pd.concat(shards, ignore_index=True) if shards else pd.DataFrame())

    def to_tensor(self, layer: str, *, where: Any = None) -> torch.Tensor:
        """Loads a layer's states into one tensor.

        Args:
            layer: The layer.
            where: A boolean mask over `trials`, as a Series or an array.

        Returns:
            One row per selected trial, in the stored dtype.

        Raises:
            ValueError: If the layer is unknown, `where` isn't a mask of the right
                length or selects nothing, rows differ in shape, or every token was
                kept, in which case `sequences` is the method to use.
        """
        self._check_layer(layer)
        if self.pool is None:
            raise ValueError("This store kept every token, so its rows differ in length. Use sequences().")
        rows = self._select(where)
        chunks = list(self._read(layer, rows))
        shapes = {chunk.shape[1:] for _, chunk, _ in chunks}
        if len(shapes) > 1:
            raise ValueError(f"The rows of layer '{layer}' differ in shape: {sorted(map(tuple, shapes))}.")
        stacked = torch.empty(len(rows), *shapes.pop(), dtype=chunks[0][1].dtype)
        for picks, chunk, _ in chunks:
            stacked[picks] = chunk
        return stacked

    def to_numpy(self, layer: str, *, where: Any = None) -> np.ndarray:
        """Loads a layer's states into one array.

        Args:
            layer: The layer.
            where: A boolean mask over `trials`, as a Series or an array.

        Returns:
            One row per selected trial. NumPy has no bfloat16, so that becomes float32.

        Raises:
            ValueError: In the same cases as `to_tensor`.
        """
        states = self.to_tensor(layer, where=where)
        return (states.float() if states.dtype == torch.bfloat16 else states).numpy()

    def sequences(self, layer: str, *, where: Any = None) -> list[torch.Tensor]:
        """Loads a layer's states with every token, for a store with no pooling.

        Args:
            layer: The layer.
            where: A boolean mask over `trials`, as a Series or an array.

        Returns:
            One tensor of shape (tokens, ...) per selected trial.

        Raises:
            ValueError: If the layer is unknown, `where` isn't a mask of the right
                length or selects nothing, or the store was pooled, in which case
                `to_tensor` is the method to use.
        """
        self._check_layer(layer)
        if self.pool is not None:
            raise ValueError(f"This store was pooled with '{self.pool}', so it holds one row per trial. Use to_tensor().")
        rows = self._select(where)
        sequences: list[torch.Tensor | None] = [None] * len(rows)
        for picks, chunk, lengths in self._read(layer, rows):
            for position, row, length in zip(picks.tolist(), chunk, lengths.tolist(), strict=True):
                sequences[position] = row[:length].clone()
        return sequences

    def compare(
        self,
        metric: str | Callable[[torch.Tensor, torch.Tensor], float] = "linear_cka",
        *,
        layers: Sequence[str] | None = None,
        where: Any = None
    ) -> pd.DataFrame:
        """Compares every layer with every other layer.

        Args:
            metric: A name from `metrics.GEOMETRY`, or a function of two matrices.
            layers: The layers to compare. Defaults to all of them.
            where: A boolean mask over `trials`, as a Series or an array.

        Returns:
            A square table labelled by layer.

        Raises:
            ValueError: In the same cases as `to_tensor`, or if `metric` is unknown.
        """
        chosen = self.layers if layers is None else layers
        return layer_matrix({layer: self.to_tensor(layer, where=where) for layer in chosen}, metric)

    def save(self, path: str | Path) -> HiddenStates:
        """Copies these states to a standalone folder that can be shared.

        The folder holds `manifest.json`, `trials.parquet` and `states/`. It appears
        only once it's complete, so a failed save leaves nothing behind.

        Args:
            path: The folder to create. It may exist if it's empty.

        Returns:
            The saved store.

        Raises:
            RunConflictError: If the folder holds anything already. Nothing is ever
                deleted.
        """
        target = Path(path)
        if target.exists() and any(target.iterdir()):
            raise RunConflictError(f"{target} isn't empty. Choose another folder to save to.")
        target.parent.mkdir(parents=True, exist_ok=True)
        building = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        try:
            self._copy_to(building)
            if target.exists():
                target.rmdir()
            os.replace(building, target)
        finally:
            shutil.rmtree(building, ignore_errors=True)
        return HiddenStates.open(target)

    def push_to_hub(
        self,
        repo_id: str,
        *,
        private: bool = True,
        token: str | None = None
    ) -> None:
        """Uploads a store to the Hugging Face Hub as a dataset.

        Only the store's own files go up, so a run's other files stay local.

        Args:
            repo_id: The dataset repository, which is created if needed.
            private: Whether the repository must be private.
            token: The Hub token. By default, the one you are logged in with.

        Raises:
            ValueError: If `private` is set and the repository already exists as a
                public one, since uploading would publish these states.
        """
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(
            repo_id,
            repo_type="dataset",
            exist_ok=True,
            private=private
        )
        if private and not api.repo_info(repo_id, repo_type="dataset").private:
            raise ValueError(f"{repo_id} already exists as a public repository. Pass private=False to upload to it.")
        if (self.path / "trials.parquet").exists():
            api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(self.path))
            return
        with tempfile.TemporaryDirectory() as folder:
            saved = self.save(Path(folder) / "store")
            api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(saved.path))

    def _check_layer(self, layer: str) -> None:
        if layer not in self.layers:
            raise ValueError(f"Unknown layer '{layer}'. This store has {list(self.layers)}.")

    def _select(self, where: Any) -> pd.DataFrame:
        rows = self.trials
        if where is not None:
            mask = np.asarray(where)
            if mask.dtype != bool or len(mask) != len(rows):
                raise ValueError(f"where must be a boolean mask over the {len(rows)} trials.")
            rows = rows[mask]
        if rows.empty:
            raise ValueError("No trials are selected.")
        return rows

    def _read(self, layer: str, rows: pd.DataFrame) -> Any:
        """Yields each shard's positions in `rows`, states, and lengths if it has them."""
        shards = rows["state_shard"].to_numpy()
        offsets = torch.tensor(rows["state_offset"].to_numpy(dtype=np.int64))
        for shard in pd.unique(shards):
            picks = torch.from_numpy(np.flatnonzero(shards == shard))
            with safe_open(self.path / "states" / layer / f"{shard}.safetensors", framework="pt") as file:
                chosen = offsets[picks]
                lengths = file.get_tensor("lengths")[chosen] if "lengths" in file.keys() else None
                yield picks, file.get_tensor("states")[chosen], lengths

    def _copy_to(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        for layer in self.layers:
            (folder / "states" / layer).mkdir(parents=True)
            for shard in pd.unique(self.trials["state_shard"]):
                name = f"{shard}.safetensors"
                shutil.copy2(self.path / "states" / layer / name, folder / "states" / layer / name)
        self.trials.to_parquet(folder / "trials.parquet")
        manifest = {
            "layers": list(self.layers),
            "pool": self.pool,
            "dtype": self.dtype,
            **{section: self.manifest.get(section) for section in _SHARED_SECTIONS}
        }
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _pad_time(states: list[torch.Tensor], longest: int) -> torch.Tensor:
    total = sum(len(piece) for piece in states)
    padded = states[0].new_zeros(total, longest, *states[0].shape[2:])
    start = 0
    for piece in states:
        kept = piece[:, :longest]
        padded[start : start + len(piece), : kept.shape[1]] = kept
        start += len(piece)
    return padded


def _save_shard(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    save_file({name: tensor.contiguous() for name, tensor in tensors.items()}, temporary)
    os.replace(temporary, path)
