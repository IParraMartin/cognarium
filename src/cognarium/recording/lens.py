"""The logit lens: reading each layer's states as if they were the last layer's."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence

import pandas as pd
import torch

from cognarium.recording.logits import logit_columns
from cognarium.recording.store import HiddenStates
from cognarium.subjects.base import Readout

_COLUMNS = (
    "layer",
    "top_token",
    "top_log_prob",
    "entropy",
    "option_log_probs",
    "logit_margin",
    "target_rank",
    "logit_difference"
)


def read_layers(
    hidden: HiddenStates,
    readout: Readout,
    first_tokens: Callable[[Sequence[str]], list[int]],
    *,
    group_by: Sequence[str],
    layers: Sequence[str],
    batch_size: int
) -> pd.DataFrame:
    """Runs each recorded state through a subject's final norm and output head.

    Args:
        hidden: The recorded states, pooled to one row per query.
        readout: The subject's way of turning states into logits.
        first_tokens: Gives the first token of each option in a list of options.
        group_by: The paradigm's grouping columns, which are copied to the table.
        layers: The layers to read.
        batch_size: How many states go through the head at once.

    Returns:
        One row per query per layer, in the order of `layers` then of the trials.
    """
    trials = hidden.trials
    option_lists = [tuple(str(option) for option in row) for row in trials["options"]]
    tokens_of = {options: first_tokens(options) for options in set(option_lists)}
    tokens = [tokens_of[options] for options in option_lists]
    answers = [
        None if pd.isna(answer) else options.index(answer)
        for answer, options in zip(trials["answer"], option_lists, strict=True)
    ]
    labels = pd.DataFrame(
        {
            "step": trials["step"],
            "query": trials.groupby("step").cumcount(),
            "item": trials["item"],
            **{column: trials[column] for column in group_by}
        }
    )
    parts = []
    for layer in layers:
        columns = _read_layer(
            hidden.to_tensor(layer),
            readout,
            tokens,
            answers,
            batch_size
        )
        parts.append(labels.assign(**{"layer": layer, **columns}))
    table = pd.concat(parts, ignore_index=True)
    return table[[*labels.columns, *_COLUMNS]]


def _read_layer(
    states: torch.Tensor,
    readout: Readout,
    tokens: Sequence[list[int]],
    answers: Sequence[int | None],
    batch_size: int
) -> dict[str, list]:
    columns = defaultdict(list)
    decoded = {}
    for start in range(0, len(states), batch_size):
        stop = start + batch_size
        with torch.inference_mode():
            logits = readout.head(states[start:stop].to(readout.device, dtype=readout.dtype))
        logits = logits.float().cpu()
        measures = logit_columns(logits, tokens[start:stop], answers[start:stop])
        top_log_probs = logits.double().log_softmax(dim=-1).max(dim=-1).values
        for token in measures["top_token_id"]:
            if token not in decoded:
                decoded[token] = readout.decode(token)
            columns["top_token"].append(decoded[token])
        columns["top_log_prob"].extend(top_log_probs.tolist())
        columns["entropy"].extend(measures["vocab_entropy"])
        columns["option_log_probs"].extend(
            row[torch.tensor(row_tokens)].double().log_softmax(dim=-1).tolist()
            for row, row_tokens in zip(logits, tokens[start:stop], strict=True)
        )
        for name in ("logit_margin", "target_rank", "logit_difference"):
            columns[name].extend(measures[name])
    return dict(columns)
