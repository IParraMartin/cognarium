"""Writing free responses with any model that outputs next-token logits."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from cognarium.responses import Query, sample_responses, shape_log_probs
from cognarium.subjects.scoring import call_forward, pad_sequences


def generate_texts(
    queries: Sequence[Query],
    *,
    forward: Callable[..., torch.Tensor],
    encode_prompt: Callable[[str], list[int]] | None = None,
    decode: Callable[[list[int]], str],
    eos_token_id: int | None,
    pad_token_id: int,
    device: torch.device,
    batch_size: int = 32,
    max_length: int | None = None,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int | None = None,
    encode_query: Callable[[int, Query], list[int]] | None = None
) -> list[str]:
    """Writes each query's free response, one token at a time.

    Each token is drawn with a seed made from the query's own seed and the token's
    position, so a response doesn't depend on which other queries share its batch.

    Args:
        queries: The queries to answer.
        forward: Maps right-padded ids and an attention mask to logits of shape
            (batch, time, vocab). With `encode_query`, it is called as
            `forward(ids, mask, rows)`, where `rows` gives the query behind each row.
        encode_prompt: Turns a prompt's text into token ids. Required unless
            `encode_query` is given.
        decode: Turns generated token ids into text.
        eos_token_id: The token that ends a response, if the model has one.
        pad_token_id: The id used for padding, which also ends a response.
        device: Where the model runs.
        batch_size: How many sequences go through the model at once.
        max_length: The longest sequence the model can read.
        max_new_tokens: How many tokens each response may use.
        temperature: Divides log probabilities before a token is sampled.
        top_p: Keeps the smallest set of tokens covering this much probability.
        top_k: Keeps only this many of the most likely tokens.
        encode_query: Turns a query, with its index, into token ids, for subjects
            whose prompts hold more than text. Replaces `encode_prompt`.

    Returns:
        One text per query, without the end token. A response that ends at once is empty.

    Raises:
        ValueError: If a prompt is empty, or leaves no room for `max_new_tokens`.
    """
    if encode_query is not None:
        prompts = [encode_query(index, query) for index, query in enumerate(queries)]
    elif encode_prompt is not None:
        prompts = [encode_prompt(query.prompt()) for query in queries]
    else:
        raise TypeError("Generation needs encode_prompt or encode_query.")
    _check_room(prompts, max_length, max_new_tokens)
    generated: list[list[int]] = [[] for _ in queries]
    with torch.inference_mode():
        for start in range(0, len(queries), batch_size):
            active = list(range(start, min(start + batch_size, len(queries))))
            for position in range(max_new_tokens):
                if not active:
                    break
                ids, mask = pad_sequences([prompts[i] + generated[i] for i in active], pad_token_id, device)
                last = mask.sum(dim=1) - 1
                output = call_forward(forward, ids, mask, active, encode_query is not None)
                logits = output[torch.arange(len(active), device=device), last].float().cpu()
                shaped = shape_log_probs(
                    logits.log_softmax(dim=-1),
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k
                )
                seeds = [(*queries[i].response_seed, position) for i in active]
                tokens = sample_responses(shaped, seeds).tolist()
                unfinished = []
                for i, token in zip(active, tokens, strict=True):
                    if token not in (eos_token_id, pad_token_id):
                        generated[i].append(token)
                        unfinished.append(i)
                active = unfinished
    return [decode(tokens) for tokens in generated]


def _check_room(prompts: list[list[int]], max_length: int | None, max_new_tokens: int) -> None:
    for index, prompt in enumerate(prompts):
        if not prompt:
            raise ValueError(f"The prompt of query {index} has no tokens.")
        needed = len(prompt) + max_new_tokens
        if max_length is not None and needed > max_length:
            raise ValueError(
                f"Query {index} needs room for {needed} tokens, over the model's limit of {max_length}. "
                "Keep less context with cg.context.window(k), or lower max_new_tokens."
            )
