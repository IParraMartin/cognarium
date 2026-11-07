"""Per-query measures read from full-vocabulary logits."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from cognarium.metrics.information import entropy


def logit_columns(
    logits: torch.Tensor,
    option_tokens: Sequence[Sequence[int]],
    answers: Sequence[int | None]
) -> dict[str, list]:
    """Computes the logit measures of a step's queries.

    An option is represented by its first token, so two options that start with the
    same token have no margin between them.

    Args:
        logits: The logits at each query's last prompt position, shape (n, vocab).
        option_tokens: The first token of each option, for each query.
        answers: The position of each query's correct option, or None if it has none.

    Returns:
        One list per column, each with one value per query:

        - `option_first_tokens`: the first token of each option.
        - `top_token_id`: the most likely token in the whole vocabulary.
        - `vocab_entropy`: the entropy over the whole vocabulary, in bits.
        - `logit_margin`: the top option minus the second, or None with one option.
        - `target_rank`: the answer's rank in the whole vocabulary, where 1 is the
          most likely token, or None with no answer.
        - `logit_difference`: the answer minus the best other option, or None with no
          answer or with one option.
    """
    logits = logits.detach().cpu().double()
    margins, ranks, differences = [], [], []
    for row, tokens, answer in zip(
        logits,
        option_tokens,
        answers,
        strict=True
    ):
        values = row[torch.tensor(list(tokens))]
        ordered = values.sort(descending=True).values
        margins.append((ordered[0] - ordered[1]).item() if len(ordered) > 1 else None)
        if answer is None:
            ranks.append(None)
            differences.append(None)
            continue
        ranks.append(int((row > values[answer]).sum()) + 1)
        others = torch.cat([values[:answer], values[answer + 1 :]])
        differences.append((values[answer] - others.max()).item() if len(others) > 0 else None)
    return {
        "option_first_tokens": [list(tokens) for tokens in option_tokens],
        "top_token_id": logits.argmax(dim=-1).tolist(),
        "vocab_entropy": entropy(logits.log_softmax(dim=-1)).tolist(),
        "logit_margin": margins,
        "target_rank": ranks,
        "logit_difference": differences
    }
