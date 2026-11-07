"""Scoring response options with any model that outputs next-token logits."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from cognarium.responses import Query, Responses


def score_options(
    queries: Sequence[Query],
    *,
    forward: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    encode_prompt: Callable[[str], list[int]],
    encode_option: Callable[[str], list[int]],
    pad_token_id: int,
    device: torch.device,
    unk_token_id: int | None = None,
    batch_size: int = 32,
    option_prefix: str = " ",
    max_length: int | None = None,
    capture_logits: bool = False,
    on_batch: Callable[[torch.Tensor, Sequence[int]], None] | None = None
) -> Responses:
    """Scores each query's options by their log probability after its prompt.

    A score is the log probability under the full vocabulary, summed across the
    option's tokens, so options of different lengths compare fairly. For
    single-token options, score differences equal logit differences.

    Args:
        queries: The queries to answer.
        forward: Maps right-padded ids and an attention mask to logits of shape
            (batch, time, vocab).
        encode_prompt: Turns a prompt into token ids.
        encode_option: Turns an option, after `option_prefix`, into token ids.
        pad_token_id: The id used for padding.
        device: Where the model runs.
        unk_token_id: The id of the unknown token, which an option must not use.
        batch_size: How many sequences go through the model at once.
        option_prefix: What separates a prompt from its option.
        max_length: The longest sequence the model can read.
        capture_logits: Whether to return logits at the last prompt position.
        on_batch: Called as `on_batch(mask, queries)` right after the forward pass of
            each batch of prompts. Every query then goes through that pass exactly
            once, whatever the length of its options.

    Returns:
        One row of scores per query, on CPU.

    Raises:
        ValueError: If an option can't be told apart from another or isn't in the
            vocabulary, a prompt is empty, or a sequence exceeds `max_length`.
    """
    prompts = [encode_prompt(query.prompt()) for query in queries]
    options = [_encode_options(
        query,
        encode_option,
        option_prefix,
        unk_token_id
    ) for query in queries]
    _check_lengths(prompts, options, max_length)
    width = max(len(row) for row in options)
    scores = torch.full((len(queries), width), -torch.inf, dtype=torch.float64)
    single = [i for i, row in enumerate(options) if all(len(tokens) == 1 for tokens in row)]
    multi = sorted(set(range(len(queries))) - set(single))
    logits = None
    with torch.inference_mode():
        last = _score_from_prompts(
            range(len(queries)) if capture_logits or on_batch is not None else single,
            set(single),
            prompts,
            options,
            scores,
            forward=forward,
            pad_token_id=pad_token_id,
            device=device,
            batch_size=batch_size,
            on_batch=on_batch
        )
        if capture_logits:
            logits = torch.cat(last)
        _score_from_pairs(
            multi,
            prompts,
            options,
            scores,
            forward=forward,
            pad_token_id=pad_token_id,
            device=device,
            batch_size=batch_size
        )
    return Responses(scores=scores, logits=logits)


def _score_from_prompts(
    indices: Sequence[int],
    single: set[int],
    prompts: list[list[int]],
    options: list[list[list[int]]],
    scores: torch.Tensor,
    *,
    forward: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    pad_token_id: int,
    device: torch.device,
    batch_size: int,
    on_batch: Callable[[torch.Tensor, Sequence[int]], None] | None
) -> list[torch.Tensor]:
    """Runs each prompt once, which is enough for single-token options."""
    collected = []
    for batch in _batches(indices, batch_size):
        ids, mask = pad_sequences([prompts[i] for i in batch], pad_token_id, device)
        positions = mask.sum(dim=1) - 1
        rows = torch.arange(len(batch), device=device)
        output = forward(ids, mask)
        if on_batch is not None:
            on_batch(mask, batch)
        logits = output[rows, positions].float().cpu()
        log_probs = logits.log_softmax(dim=-1).double()
        for row, index in enumerate(batch):
            if index in single:
                tokens = [token[0] for token in options[index]]
                scores[index, : len(tokens)] = log_probs[row, tokens]
        collected.append(logits)
    return collected


def _score_from_pairs(
    indices: Sequence[int],
    prompts: list[list[int]],
    options: list[list[list[int]]],
    scores: torch.Tensor,
    *,
    forward: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    pad_token_id: int,
    device: torch.device,
    batch_size: int
) -> None:
    """Runs each prompt with each option, which multi-token options need."""
    pairs = [(i, k) for i in indices for k in range(len(options[i]))]
    for batch in _batches(pairs, batch_size):
        sequences = [prompts[i] + options[i][k] for i, k in batch]
        ids, mask = pad_sequences(sequences, pad_token_id, device)
        log_probs = forward(ids, mask).float().log_softmax(dim=-1).cpu().double()
        for row, (i, k) in enumerate(batch):
            start = len(prompts[i]) - 1
            tokens = options[i][k]
            positions = torch.arange(start, start + len(tokens))
            scores[i, k] = log_probs[row, positions, torch.tensor(tokens)].sum()


def _encode_options(
    query: Query,
    encode_option: Callable[[str], list[int]],
    option_prefix: str,
    unk_token_id: int | None
) -> list[list[int]]:
    encoded = []
    for option in query.trial.options:
        tokens = encode_option(option_prefix + option)
        if not tokens:
            raise ValueError(f"Option '{option}' has no tokens, so it can't be scored.")
        if unk_token_id is not None and unk_token_id in tokens:
            raise ValueError(f"Option '{option}' isn't in the model's vocabulary, so it can't be scored.")
        encoded.append(tokens)
    if len({tuple(tokens) for tokens in encoded}) != len(encoded):
        raise ValueError(f"Options {query.trial.options} have the same tokens, so they can't be told apart.")
    return encoded


def _check_lengths(
    prompts: list[list[int]],
    options: list[list[list[int]]],
    max_length: int | None
) -> None:
    for index, (prompt, row) in enumerate(zip(prompts, options, strict=True)):
        if not prompt:
            raise ValueError(f"The prompt of query {index} has no tokens.")
        longest = len(prompt) + max(len(tokens) for tokens in row)
        if max_length is not None and longest > max_length:
            raise ValueError(
                f"Query {index} needs {longest} tokens, over the model's limit of {max_length}. "
                "Keep less context with cg.context.window(k), or use a shorter sequence."
            )


def pad_sequences(
    sequences: list[list[int]],
    pad_token_id: int,
    device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pads token sequences into a batch.

    Args:
        sequences: The token ids of each sequence.
        pad_token_id: The id used for padding.
        device: Where the batch should live.

    Returns:
        The ids and the attention mask, both of shape (batch, longest sequence).
    """
    length = max(len(sequence) for sequence in sequences)
    ids = torch.full((len(sequences), length), pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), length), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        mask[row, : len(sequence)] = 1
    return ids.to(device), mask.to(device)


def _batches(items: Sequence, batch_size: int) -> list[Sequence]:
    items = list(items)
    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]
