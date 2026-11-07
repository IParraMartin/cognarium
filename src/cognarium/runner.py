"""Runs a paradigm on a subject and saves the results step by step."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from safetensors.torch import save_file
from tqdm.auto import tqdm

from cognarium.errors import MissingAbilityError, RunConflictError
from cognarium.paradigms.base import Paradigm, Step
from cognarium.recording.capture import StateCapture
from cognarium.recording.logits import logit_columns
from cognarium.recording.store import StateWriter
from cognarium.responses import (
    Query,
    Responses,
    concat_responses,
    sample_responses,
    shape_log_probs
)
from cognarium.results import Results, logits_path, step_path
from cognarium.subjects.base import Subject

_RECORDABLE = ("logits", "hidden_states")
_IDENTITY_SECTIONS = (
    "paradigm",
    "subject",
    "learning",
    "record",
    "states"
)
_POOLS = ("last", "mean", None)
_STATE_DTYPES = ("auto", "float16", "bfloat16", "float32", "float64")
# a row with every token can be far bigger than a pooled one
_ROWS_PER_SHARD = {True: 256, False: 4096}


def run(
    paradigm: Paradigm,
    subject: Subject,
    *,
    learning: str = "auto",
    record: Sequence[str] = (),
    layers: Sequence[str] | None = None,
    pool: str | None = "last",
    state_dtype: str = "auto",
    out: str | Path | None = None,
    resume: bool = True,
    progress: bool = True
) -> Results:
    """Runs an experiment, or continues it if it was stopped.

    Every step is saved as soon as it finishes. Running the same call again picks
    up after the last saved step, and a finished run makes no further model calls.

    Args:
        paradigm: The experiment to run.
        subject: Who takes part.
        learning: How the subject learns. `"auto"` picks the subject's natural mode.
        record: What to save beyond the results table: `"logits"`, `"hidden_states"`
            or both. With logits, a subject that has tokens also gets the columns
            `logit_margin`, `logit_difference`, `target_rank`, `vocab_entropy`,
            `top_token_id` and `option_first_tokens`.
        layers: The layers to record, by name. Defaults to every layer the subject has.
        pool: How to reduce each layer's time dimension: `"last"` keeps the last real
            token, `"mean"` averages over real tokens, and None keeps every token.
        state_dtype: The precision to store states in. `"auto"` takes the subject's own
            default, which is float16 for neural subjects.
        out: The run folder. Defaults to `runs/<paradigm name>/<run key>/`.
        resume: Whether an existing folder may be continued.
        progress: Whether to show a progress bar.

    Returns:
        The results of the whole run.

    Raises:
        MissingAbilityError: If the subject can't take part, before any model call.
        ValueError: If `record` names something that can't be recorded, `layers`,
            `pool` or `state_dtype` has a value that doesn't exist, or one is set when
            `record` leaves out `"hidden_states"`.
        RunConflictError: If the folder holds a different run, or `resume` is False
            and the folder already holds a run. Nothing is ever deleted.
    """
    mode = _resolve_learning(paradigm, subject, learning)
    _validate(
        paradigm,
        subject,
        mode,
        record
    )
    _validate_recording(
        subject,
        record,
        layers,
        pool,
        state_dtype
    )
    bound = subject.bind(paradigm, mode)
    described = {
        "paradigm": paradigm.config(),
        "subject": bound.config(),
        "learning": mode,
        "record": sorted(record)
    }
    settings = None
    if "hidden_states" in record:
        settings = {
            "layers": _chosen_layers(bound, layers),
            "pool": pool,
            "dtype": bound.state_dtype if state_dtype == "auto" else state_dtype
        }
        described["states"] = settings
    # a round trip through JSON, so it compares equal to what a manifest stores
    identity = json.loads(json.dumps(described))
    key = _run_key(identity)
    root = Path(out) if out is not None else Path("runs") / paradigm.name / key
    manifest = _open_manifest(
        root,
        key,
        identity,
        paradigm,
        bound,
        resume
    )
    _write_json(root / "manifest.json", manifest)

    index = _count_finished_steps(root)
    previous = pd.read_parquet(step_path(root, index - 1)) if index > 0 else None
    with tqdm(
        desc=paradigm.name,
        unit="step",
        initial=index,
        disable=not progress
    ) as bar:
        while (step := paradigm.step(index, previous)) is not None:
            responses, state_columns = _respond_and_store(
                bound,
                step,
                mode,
                capture_logits="logits" in record,
                root=root,
                settings=settings
            )
            extra = {**state_columns, **_logit_table(bound, step, responses)}
            previous = _build_table(
                paradigm,
                step,
                responses,
                extra
            )
            if "logits" in record and responses.logits is not None:
                _write_logits(logits_path(root, index), responses.logits)
            _write_table(step_path(root, index), previous)
            index += 1
            bar.update()

    manifest["status"] = "complete"
    _write_json(root / "manifest.json", manifest)
    return Results.load(root, subject=bound)


def _resolve_learning(paradigm: Paradigm, subject: Subject, learning: str) -> str | None:
    if learning != "auto":
        return learning
    if paradigm.requires_learning and subject.default_learning is None:
        raise MissingAbilityError(
            f"Paradigm '{paradigm.name}' requires learning, and {type(subject).__name__} has no "
            "learning mode."
        )
    return subject.default_learning


def _validate(
    paradigm: Paradigm,
    subject: Subject,
    mode: str | None,
    record: Sequence[str]
) -> None:
    unknown = sorted(set(record) - set(_RECORDABLE))
    if unknown:
        raise ValueError(f"Can't record {unknown}. Choose from {list(_RECORDABLE)}.")
    if not paradigm.modalities & subject.modalities:
        raise MissingAbilityError(
            f"{type(subject).__name__} reads {sorted(subject.modalities)}, but paradigm "
            f"'{paradigm.name}' provides {sorted(paradigm.modalities)}."
        )
    if paradigm.settings.instructions is not None and "text" not in subject.modalities:
        raise MissingAbilityError(
            f"{type(subject).__name__} doesn't read text, so it can't follow instructions. "
            "Remove them for this subject."
        )
    if paradigm.requires_learning and mode not in subject.learning_modes:
        raise MissingAbilityError(
            f"{type(subject).__name__} can't use learning mode '{mode}'. It supports "
            f"{sorted(subject.learning_modes)}."
        )


def _validate_recording(
    subject: Subject,
    record: Sequence[str],
    layers: Sequence[str] | None,
    pool: str | None,
    state_dtype: str
) -> None:
    if "hidden_states" not in record:
        if layers is not None or pool != "last" or state_dtype != "auto":
            raise ValueError("layers, pool and state_dtype only apply with record=['hidden_states'].")
        return
    if pool not in _POOLS:
        raise ValueError(f"Unknown pool '{pool}'. Choose from {list(_POOLS)}.")
    if state_dtype not in _STATE_DTYPES:
        raise ValueError(f"Unknown state_dtype '{state_dtype}'. Choose from {list(_STATE_DTYPES)}.")
    if not subject.state_layers():
        raise MissingAbilityError(f"{type(subject).__name__} has no internal states to record.")


def _chosen_layers(subject: Subject, layers: Sequence[str] | None) -> list[str]:
    available = subject.state_layers()
    if layers is None:
        return list(available)
    unknown = sorted(set(layers) - set(available))
    if unknown or not layers:
        raise ValueError(f"Choose layers from {list(available)}. Got {list(layers)}.")
    # in the subject's own order, so that the order they're listed in can't change a run
    return [layer for layer in available if layer in layers]


def _run_key(identity: Mapping[str, Any]) -> str:
    text = json.dumps(identity, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _open_manifest(
    root: Path,
    key: str,
    identity: Mapping[str, Any],
    paradigm: Paradigm,
    subject: Subject,
    resume: bool
) -> dict[str, Any]:
    path = root / "manifest.json"
    session = {**subject.runtime(), "started": datetime.now(timezone.utc).isoformat()}
    if not path.exists():
        if (root / "steps").exists() or (root / "logits").exists() or (root / "states").exists():
            raise RunConflictError(
                f"{root} holds step files but no manifest.json, so they can't be told apart from "
                "another run's. Choose another folder."
            )
        return {
            "key": key,
            **identity,
            "group_by": list(paradigm.group_by),
            "groups": identity["paradigm"].get("groups"),
            "reference": subject.group_prior(),
            "versions": _versions(),
            "sessions": [session],
            "uses_examples": len(paradigm.settings.examples) > 0,
            "status": "running"
        }
    if not resume:
        raise RunConflictError(f"{root} already holds a run, and resume=False. Choose another folder.")
    manifest = json.loads(path.read_text())
    if manifest["key"] != key:
        raise RunConflictError(
            f"{root} holds a different run. {_describe_differences(manifest, identity)} "
            "Choose another folder to keep both."
        )
    return {**manifest, "sessions": [*manifest["sessions"], session], "status": "running"}


def _describe_differences(stored: Mapping[str, Any], identity: Mapping[str, Any]) -> str:
    differences = []
    for section in _IDENTITY_SECTIONS:
        old, new = stored.get(section), identity.get(section)
        if old == new:
            continue
        if isinstance(old, Mapping) and isinstance(new, Mapping):
            keys = sorted(key for key in old.keys() | new.keys() if old.get(key) != new.get(key))
            differences.append(f"In {section}, these differ: {keys}.")
        else:
            differences.append(f"{section} differs: {old} then, {new} now.")
    return " ".join(differences)


def _versions() -> dict[str, str]:
    from cognarium import __version__

    versions = {
        "cognarium": __version__,
        "python": platform.python_version(),
        "torch": torch.__version__
    }
    # only when already loaded, since importing it takes seconds
    transformers = sys.modules.get("transformers")
    if transformers is not None:
        versions["transformers"] = transformers.__version__
    return versions


def _count_finished_steps(root: Path) -> int:
    index = 0
    while step_path(root, index).exists():
        index += 1
    return index


def _respond_and_store(
    subject: Subject,
    step: Step,
    mode: str | None,
    *,
    capture_logits: bool,
    root: Path,
    settings: Mapping[str, Any] | None
) -> tuple[Responses, dict[str, list[Any]]]:
    if settings is None:
        return _respond(
            subject,
            step,
            mode,
            capture_logits=capture_logits
        ), {}
    pool = settings["pool"]
    with StateWriter(
        root,
        step.index,
        dtype=settings["dtype"],
        rows_per_shard=_ROWS_PER_SHARD[pool is None]
    ) as writer:
        capture = StateCapture(tuple(settings["layers"]), pool, writer.write)
        responses = _respond(
            subject,
            step,
            mode,
            capture_logits=capture_logits,
            capture_states=capture
        )
        missing = sorted(set(settings["layers"]) - set(writer.layers))
        if missing:
            raise RuntimeError(f"{type(subject).__name__} didn't write the requested layers {missing}.")
        places = writer.finish(len(step.queries))
    return responses, {
        "state_shard": [places[position][0] for position in range(len(step.queries))],
        "state_offset": [places[position][1] for position in range(len(step.queries))]
    }


def _logit_table(subject: Subject, step: Step, responses: Responses) -> dict[str, list[Any]]:
    if responses.logits is None:
        return {}
    tokens = {options: subject.first_tokens(options) for options in {query.trial.options for query in step.queries}}
    if any(row is None for row in tokens.values()):
        return {}
    answers = [
        None if query.trial.answer is None else query.trial.options.index(query.trial.answer)
        for query in step.queries
    ]
    return logit_columns(responses.logits, [tokens[query.trial.options] for query in step.queries], answers)


def _respond(
    subject: Subject,
    step: Step,
    mode: str | None,
    *,
    capture_logits: bool,
    capture_states: StateCapture | None = None
) -> Responses:
    # only when recording, so a subject that can't record needn't take the argument
    extra = {} if capture_states is None else {"capture_states": capture_states}
    if mode != "weights":
        return subject.respond(step.queries, capture_logits=capture_logits, **extra)
    positions: dict[int, list[int]] = {}
    for position, query in enumerate(step.queries):
        positions.setdefault(query.group, []).append(position)
    parts = []
    order = []
    for members in positions.values():
        parts.append(
            _train_and_respond(
                subject,
                [step.queries[i] for i in members],
                capture_logits,
                None if capture_states is None else _in_step_positions(capture_states, members)
            )
        )
        order.extend(members)
    joined = concat_responses(parts)
    restored = torch.argsort(torch.tensor(order))
    return Responses(
        scores=joined.scores[restored],
        choices=None if joined.choices is None else joined.choices[restored],
        logits=None if joined.logits is None else joined.logits[restored]
    )


def _in_step_positions(capture: StateCapture, members: Sequence[int]) -> StateCapture:
    # a learner sees only its own group, so its positions must map back to the step's
    def write(
        layer: str,
        states: torch.Tensor,
        queries: Sequence[int],
        lengths: torch.Tensor | None = None
    ) -> None:
        capture.write(
            layer,
            states,
            [members[query] for query in queries],
            lengths
        )

    return replace(capture, write=write)


def _train_and_respond(
    subject: Subject,
    queries: Sequence[Query],
    capture_logits: bool,
    capture_states: StateCapture | None
) -> Responses:
    first = queries[0]
    learner = subject.fresh(first.init_seed)
    learner.learn(first.context, template=first.template, instructions=first.instructions)
    # the context now lives in the weights, so it mustn't also reach the prompt
    bare = [replace(query, context=()) for query in queries]
    extra = {} if capture_states is None else {"capture_states": capture_states}
    return learner.respond(bare, capture_logits=capture_logits, **extra)


def _build_table(
    paradigm: Paradigm,
    step: Step,
    responses: Responses,
    extra: Mapping[str, list[Any]]
) -> pd.DataFrame:
    log_probs = responses.log_probs
    shaped = shape_log_probs(log_probs, **paradigm.settings.shaping)
    if responses.choices is not None:
        chosen = responses.choices
    else:
        chosen = sample_responses(shaped, [query.response_seed for query in step.queries])
    rows: dict[str, list[Any]] = {
        "step": [step.index] * len(step.queries),
        "item": [],
        "options": [],
        "answer": [],
        "response": [],
        "response_index": chosen.tolist(),
        "scores": [],
        "log_probs": [],
        "shaped_log_probs": []
    }
    for row, query in enumerate(step.queries):
        trial = query.trial
        width = len(trial.options)
        rows["item"].append(trial.item)
        rows["options"].append(list(trial.options))
        rows["answer"].append(trial.answer)
        rows["response"].append(trial.options[chosen[row].item()])
        rows["scores"].append(responses.scores[row, :width].double().tolist())
        rows["log_probs"].append(log_probs[row, :width].tolist())
        rows["shaped_log_probs"].append(shaped[row, :width].tolist())
    return paradigm.annotate(pd.DataFrame({**rows, **step.columns, **extra}))


def _write_table(path: Path, frame: pd.DataFrame) -> None:
    temporary = _temporary(path)
    frame.to_parquet(temporary)
    os.replace(temporary, path)


def _write_logits(path: Path, logits: torch.Tensor) -> None:
    temporary = _temporary(path)
    save_file({"logits": logits.cpu().contiguous()}, temporary)
    os.replace(temporary, path)


def _write_json(path: Path, content: Mapping[str, Any]) -> None:
    temporary = _temporary(path)
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True))
    os.replace(temporary, path)


def _temporary(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(path.name + ".tmp")
