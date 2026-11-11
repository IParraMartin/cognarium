"""Scoring response options with any model that outputs next-token logits."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from cognarium.responses import Query, Responses


def score_options(
    queries: Sequence[Query],
    *,
    forward: Callable[..., torch.Tensor],
    encode_prompt: Callable[[str], list[int]] | None = None,
    encode_option: Callable[[str], list[int]],
    pad_token_id: int,
    device: torch.device,
    unk_token_id: int | None = None,
    batch_size: int = 32,
    option_prefix: str = " ",
    max_length: int | None = None,
    capture_logits: bool = False,
    on_batch: Callable[[torch.Tensor, Sequence[int]], None] | None = None,
    score_after_prompt: Callable[[int, list[int], list[list[int]]], torch.Tensor | None] | None = None,
    encode_query: Callable[[int, Query], list[int]] | None = None
) -> Responses:
    """Scores each query's options by their log probability after its prompt.

    A score is the log probability under the full vocabulary, summed across the
    option's tokens, so options of different lengths compare fairly. For
    single-token options, score differences equal logit differences. When one
    option's tokens start another's, as with 1 and 10 under a tokenizer that splits
    digits, every option of that query is scored together with the template's
    separator, which ends an answer in the prompt's examples. Otherwise the shorter
    option would always score at least as high. A template with `complete` set scores
    every option with its separator.

    Args:
        queries: The queries to answer.
        forward: Maps right-padded ids and an attention mask to logits of shape
            (batch, time, vocab). With `encode_query`, it is called as
            `forward(ids, mask, rows)`, where `rows` gives the query behind each row,
            so a subject can add what the query shows besides text, such as images.
            Each row's ids start with that query's prompt.
        encode_prompt: Turns a prompt's text into token ids. Required unless
            `encode_query` is given.
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
        score_after_prompt: Scores one prompt's multi-token options from the query's
            index, the prompt's token ids and the options' token ids, by running the
            prompt once and reusing what the model kept from it. When it returns
            None, that query runs each option with its prompt.
        encode_query: Turns a query, with its index, into token ids, for subjects
            whose prompts hold more than text. Replaces `encode_prompt`.

    Returns:
        One row of scores per query, on CPU.

    Raises:
        ValueError: If an option can't be told apart from another or isn't in the
            vocabulary, a prompt is empty, or a sequence exceeds `max_length`.
    """
    prompts = _encode_prompts(queries, encode_prompt, encode_query)
    # queries in a run often share one option list, so each list is tokenized once
    encoded: dict[tuple[tuple[str, ...], str, bool], list[list[int]]] = {}
    options = []
    for query in queries:
        key = (tuple(query.trial.options), query.template.separator, query.template.complete)
        if key not in encoded:
            encoded[key] = _encode_options(
                query,
                encode_option,
                option_prefix,
                unk_token_id
            )
        options.append(encoded[key])
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
            on_batch=on_batch,
            with_rows=encode_query is not None
        )
        if capture_logits:
            logits = torch.cat(last)
        if score_after_prompt is not None:
            declined = []
            for index in multi:
                row = score_after_prompt(index, prompts[index], options[index])
                if row is None:
                    declined.append(index)
                else:
                    scores[index, : len(row)] = row
            multi = declined
        _score_from_pairs(
            multi,
            prompts,
            options,
            scores,
            forward=forward,
            pad_token_id=pad_token_id,
            device=device,
            batch_size=batch_size,
            with_rows=encode_query is not None
        )
    return Responses(scores=scores, logits=logits)


def _encode_prompts(
    queries: Sequence[Query],
    encode_prompt: Callable[[str], list[int]] | None,
    encode_query: Callable[[int, Query], list[int]] | None
) -> list[list[int]]:
    """Encodes every query's prompt, through the query itself when the subject asks."""
    if encode_query is not None:
        return [encode_query(index, query) for index, query in enumerate(queries)]
    if encode_prompt is None:
        raise TypeError("Scoring needs encode_prompt or encode_query.")
    return [encode_prompt(query.prompt()) for query in queries]


def call_forward(
    forward: Callable[..., torch.Tensor],
    ids: torch.Tensor,
    mask: torch.Tensor,
    rows: Sequence[int],
    with_rows: bool
) -> torch.Tensor:
    """Runs a forward function, passing the rows' queries only if it takes them.

    Args:
        forward: The subject's forward function.
        ids: Right-padded token ids.
        mask: Their attention mask.
        rows: The query behind each row.
        with_rows: Whether `forward` takes `rows`.

    Returns:
        The logits, shape (batch, time, vocab).
    """
    return forward(ids, mask, list(rows)) if with_rows else forward(ids, mask)


def _score_from_prompts(
    indices: Sequence[int],
    single: set[int],
    prompts: list[list[int]],
    options: list[list[list[int]]],
    scores: torch.Tensor,
    *,
    forward: Callable[..., torch.Tensor],
    pad_token_id: int,
    device: torch.device,
    batch_size: int,
    on_batch: Callable[[torch.Tensor, Sequence[int]], None] | None,
    with_rows: bool
) -> list[torch.Tensor]:
    """Runs each prompt once, which is enough for single-token options."""
    collected = []
    for batch in _batches(indices, batch_size):
        ids, mask = pad_sequences([prompts[i] for i in batch], pad_token_id, device)
        positions = mask.sum(dim=1) - 1
        rows = torch.arange(len(batch), device=device)
        output = call_forward(forward, ids, mask, batch, with_rows)
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
    forward: Callable[..., torch.Tensor],
    pad_token_id: int,
    device: torch.device,
    batch_size: int,
    with_rows: bool
) -> None:
    """Runs each prompt with each option, which multi-token options need."""
    pairs = [(i, k) for i in indices for k in range(len(options[i]))]
    for batch in _batches(pairs, batch_size):
        sequences = [prompts[i] + options[i][k] for i, k in batch]
        ids, mask = pad_sequences(sequences, pad_token_id, device)
        rows = [i for i, _ in batch]
        log_probs = call_forward(forward, ids, mask, rows, with_rows).float().log_softmax(dim=-1).cpu().double()
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
    encoded = _tokenize_options(query.trial.options, "", encode_option, option_prefix, unk_token_id)
    if len({tuple(tokens) for tokens in encoded}) != len(encoded):
        raise ValueError(f"Options {query.trial.options} have the same tokens, so they can't be told apart.")
    starts = {tuple(tokens[:end]) for tokens in encoded for end in range(1, len(tokens))}
    ambiguous = any(tuple(tokens) in starts for tokens in encoded)
    if query.template.separator and (query.template.complete or ambiguous):
        return _tokenize_options(
            query.trial.options,
            query.template.separator,
            encode_option,
            option_prefix,
            unk_token_id
        )
    return encoded


def _tokenize_options(
    options: Sequence[str],
    ending: str,
    encode_option: Callable[[str], list[int]],
    option_prefix: str,
    unk_token_id: int | None
) -> list[list[int]]:
    encoded = []
    for option in options:
        tokens = encode_option(option_prefix + option + ending)
        if not tokens:
            raise ValueError(f"Option '{option}' has no tokens, so it can't be scored.")
        if unk_token_id is not None and unk_token_id in tokens:
            raise ValueError(f"Option '{option}' isn't in the model's vocabulary, so it can't be scored.")
        encoded.append(tokens)
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
