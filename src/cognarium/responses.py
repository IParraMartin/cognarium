"""Queries to subjects, their responses, and how responses get chosen."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from cognarium.stimuli import Observation, Stimulus, Trial

# guards the comparison against rounding when cumulative mass lands exactly on top_p
_TOP_P_TOLERANCE = 1e-12


@dataclass(frozen=True)
class PromptTemplate:
    """How stimuli and labels are written out for subjects that read text.

    Attributes:
        stimulus: A format string for the stimulus, with a `{text}` field.
        response: The cue the response follows. When it is empty, the response
            continues the stimulus directly, as when scoring the next word of a
            sentence.
        separator: What joins the parts of a prompt.
    """

    stimulus: str = "{text}"
    response: str = "Response:"
    separator: str = "\n"

    def render_query(self, stimulus: Stimulus) -> str:
        """Writes a stimulus up to the point where a response is due.

        Args:
            stimulus: The stimulus to write out.

        Returns:
            The rendered text, ending with the response cue if there is one.

        Raises:
            ValueError: If the stimulus has no text.
        """
        if stimulus.text is None:
            raise ValueError("This stimulus has no text, so it can't be rendered as a prompt.")
        written = self.stimulus.format(text=stimulus.text)
        return self.separator.join([written, self.response]) if self.response else written

    def render_observation(self, observation: Observation) -> str:
        """Writes a labelled stimulus, including its label.

        Args:
            observation: The observation to write out.

        Returns:
            The rendered text, ending with the label.
        """
        return f"{self.render_query(observation.stimulus)} {observation.label}"


@dataclass(frozen=True, eq=False)
class Query:
    """One response requested from a subject.

    Attributes:
        trial: The trial being responded to.
        context: What the subject has observed, in presentation order.
        instructions: Text placed before everything else in a prompt.
        template: How the query is written for subjects that read text.
        group: Queries in a group share one context, such as one chain's
            generation.
        response_seed: Seeds the sampling of this response.
        group_seed: Seeds any single draw shared by a group, such as a hypothesis.
        init_seed: Seeds a chain's starting weights.
    """

    trial: Trial
    context: tuple[Observation, ...] = ()
    instructions: str | None = None
    template: PromptTemplate = PromptTemplate()
    group: int = 0
    response_seed: tuple[int, ...] = (0,)
    group_seed: tuple[int, ...] = (0,)
    init_seed: tuple[int, ...] = (0,)

    def prompt(self) -> str:
        """Writes the full prompt for subjects that read text.

        Returns:
            Instructions, then context, then the trial, ending at the response cue.
        """
        parts = [] if self.instructions is None else [self.instructions]
        parts.extend(self.template.render_observation(item) for item in self.context)
        parts.append(self.template.render_query(self.trial.stimulus))
        return self.template.separator.join(parts)


@dataclass(frozen=True)
class Responses:
    """A subject's answer to a batch of queries.

    Attributes:
        scores: Unnormalised log scores per option on CPU, shape
            (n_queries, max_options), with -inf for padding and impossible options.
        choices: Chosen option indices, shape (n_queries,), for subjects that
            choose jointly across queries. Otherwise the runner samples.
        logits: Full-vocabulary logits at the last prompt position, shape
            (n_queries, vocab), for neural subjects that were asked for them.
    """

    scores: torch.Tensor
    choices: torch.Tensor | None = None
    logits: torch.Tensor | None = None

    @property
    def log_probs(self) -> torch.Tensor:
        """Log probabilities normalised over each query's options, in float64.

        Raises:
            ValueError: If every option of some query is impossible.
        """
        scores = self.scores.double()
        totals = torch.logsumexp(scores, dim=-1, keepdim=True)
        impossible = torch.isneginf(totals).flatten().nonzero()
        if len(impossible) > 0:
            raise ValueError(f"Every option is impossible for query {impossible[0].item()}.")
        return scores - totals


def concat_responses(parts: Sequence[Responses]) -> Responses:
    """Joins responses from separate calls, padding options to the widest.

    Args:
        parts: The responses to join, in order.

    Returns:
        One set of responses. `choices` and `logits` are kept only if every part
        has them.
    """
    width = max(part.scores.shape[1] for part in parts)
    padded = [
        torch.nn.functional.pad(part.scores, (0, width - part.scores.shape[1]), value=-torch.inf)
        for part in parts
    ]
    choices = [part.choices for part in parts]
    logits = [part.logits for part in parts]
    return Responses(
        scores=torch.cat(padded),
        choices=torch.cat(choices) if all(item is not None for item in choices) else None,
        logits=torch.cat(logits) if all(item is not None for item in logits) else None
    )


def shape_log_probs(
    log_probs: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int | None = None
) -> torch.Tensor:
    """Applies temperature, then top-k, then top-p to response distributions.

    Args:
        log_probs: Log probabilities, shape (n_queries, n_options).
        temperature: Divides the log probabilities. Below 1 sharpens, above 1
            flattens.
        top_p: Keeps the smallest set of most likely options covering this much
            probability.
        top_k: Keeps only this many of the most likely options.

    Returns:
        Normalised log probabilities in float64, with -inf for removed options.
    """
    shaped = _normalise(log_probs.double() / temperature)
    order = torch.argsort(
        shaped,
        dim=-1,
        descending=True,
        stable=True
    )
    if top_k is not None:
        ranks = torch.argsort(order, dim=-1)
        shaped = _normalise(shaped.masked_fill(ranks >= top_k, -torch.inf))
    if top_p < 1.0:
        ordered = shaped.gather(-1, order).exp()
        mass_before = ordered.cumsum(dim=-1) - ordered
        keep_ordered = mass_before < top_p - _TOP_P_TOLERANCE
        keep_ordered[..., 0] = True
        keep = torch.zeros_like(keep_ordered).scatter(-1, order, keep_ordered)
        shaped = _normalise(shaped.masked_fill(~keep, -torch.inf))
    return shaped


def sample_responses(log_probs: torch.Tensor, seeds: Sequence[tuple[int, ...]]) -> torch.Tensor:
    """Samples one option per query, using each query's own seed.

    Each draw depends only on its row and seed, so results don't change with
    batch size or device.

    Args:
        log_probs: Log probabilities, shape (n_queries, n_options).
        seeds: One seed per query.

    Returns:
        Chosen option indices, shape (n_queries,).
    """
    cumulative = log_probs.double().exp().cumsum(dim=-1).numpy()
    choices = []
    for row, seed in zip(cumulative, seeds, strict=True):
        draw = np.random.default_rng(seed).random() * row[-1]
        index = int(np.searchsorted(row, draw, side="right"))
        # the first position reaching the total always has positive mass
        choices.append(index if index < len(row) else int(np.argmax(row)))
    return torch.tensor(choices, dtype=torch.long)


def _normalise(log_probs: torch.Tensor) -> torch.Tensor:
    return log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)
