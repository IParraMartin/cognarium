"""Queries to subjects, their responses, and how responses get chosen."""

from __future__ import annotations

import string
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from cognarium.media import MEDIA_KINDS, Media
from cognarium.stimuli import Observation, Stimulus, Trial

Part = str | Media

# guards the comparison against rounding when cumulative mass lands exactly on top_p
_TOP_P_TOLERANCE = 1e-12


@dataclass(frozen=True)
class PromptTemplate:
    """How stimuli and labels are written out for subjects that read text.

    The stimulus format places the stimulus's fields: `{text}` for its text, and
    `{image}`, `{video}` and `{audio}` for its media, as in
    `"Color: {image} -> Label:"`. A multimodal subject reads the media where they
    stand. Media the format doesn't place go before the rest of the stimulus, so a
    stimulus with text and an image shows both under the default format. Subjects
    that only read text skip those, and can't read a format that places media.

    Attributes:
        stimulus: A format string for the stimulus, with any of the fields
            `{text}`, `{image}`, `{video}` and `{audio}`.
        response: The cue the response follows. When it is empty, the response
            continues the stimulus directly, as when scoring the next word of a
            sentence.
        separator: What joins the parts of a prompt.
        complete: Whether every option is scored with the separator after it, as a
            whole answer. Then an option that would usually go on, such as a string
            that stops halfway, scores lower than one that ends where it should.

    Raises:
        ValueError: If the stimulus format has a field other than those four.
    """

    stimulus: str = "{text}"
    response: str = "Response:"
    separator: str = "\n"
    complete: bool = False

    def __post_init__(self) -> None:
        unknown = sorted(self.fields - {"text", *MEDIA_KINDS})
        if unknown:
            raise ValueError(
                f"The stimulus format has the fields {unknown}. It can place text, image, video and audio."
            )

    @property
    def fields(self) -> frozenset[str]:
        """The stimulus fields the format places."""
        return frozenset(field for _, field, _, _ in string.Formatter().parse(self.stimulus) if field is not None)

    @property
    def placed_media(self) -> frozenset[str]:
        """The kinds of media the format places."""
        return self.fields & frozenset(MEDIA_KINDS)

    def query_parts(self, stimulus: Stimulus) -> tuple[Part, ...]:
        """Writes a stimulus up to the point where a response is due, media included.

        Args:
            stimulus: The stimulus to write out.

        Returns:
            Text and media in reading order, ending with the response cue if there
            is one. Adjacent text is joined into one string.

        Raises:
            ValueError: If the format places a field the stimulus doesn't have.
        """
        missing = sorted(field for field in self.fields if getattr(stimulus, field) is None)
        if missing:
            raise ValueError(
                f"This stimulus has no {' or '.join(missing)}, so it can't be written with the format "
                f"'{self.stimulus}'."
            )
        parts: list[Part] = [medium for kind, medium in stimulus.media.items() if kind not in self.fields]
        for literal, field, spec, conversion in string.Formatter().parse(self.stimulus):
            parts.append(literal)
            if field == "text":
                parts.append(_format_text(stimulus.text, spec, conversion))
            elif field is not None:
                parts.append(getattr(stimulus, field))
        if self.response:
            parts.extend([self.separator, self.response])
        return join_parts(parts)

    def observation_parts(self, observation: Observation) -> tuple[Part, ...]:
        """Writes a labelled stimulus, media included, ending with its label.

        Args:
            observation: The observation to write out.

        Returns:
            Text and media in reading order.
        """
        return join_parts([*self.query_parts(observation.stimulus), f" {observation.label}"])

    def render_query(self, stimulus: Stimulus) -> str:
        """Writes a stimulus as text, up to the point where a response is due.

        Args:
            stimulus: The stimulus to write out.

        Returns:
            The rendered text, ending with the response cue if there is one. Media
            the format doesn't place are left out.

        Raises:
            ValueError: If the format places media, which text can't hold, or a field
                the stimulus doesn't have.
        """
        if self.placed_media:
            raise ValueError(
                f"The format '{self.stimulus}' places {sorted(self.placed_media)}, which can't be written as "
                "text. Use a subject that reads them, such as cg.load(..., multimodal=True)."
            )
        return "".join(part for part in self.query_parts(stimulus) if isinstance(part, str))

    def render_observation(self, observation: Observation) -> str:
        """Writes a labelled stimulus as text, including its label.

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

        Raises:
            ValueError: If the template places media, which text can't hold.
        """
        parts = [] if self.instructions is None else [self.instructions]
        parts.extend(self.template.render_observation(item) for item in self.context)
        parts.append(self.template.render_query(self.trial.stimulus))
        return self.template.separator.join(parts)

    def parts(self) -> tuple[Part, ...]:
        """Writes the full prompt with its media, for subjects that see or hear.

        Returns:
            Instructions, then context, then the trial, as text and media in reading
            order, ending at the response cue.
        """
        blocks = [] if self.instructions is None else [(self.instructions,)]
        blocks.extend(self.template.observation_parts(item) for item in self.context)
        blocks.append(self.template.query_parts(self.trial.stimulus))
        joined: list[Part] = []
        for number, block in enumerate(blocks):
            if number:
                joined.append(self.template.separator)
            joined.extend(block)
        return join_parts(joined)

    def describe(self) -> str:
        """Writes the full prompt with each medium named, as in `<image 3f2a9c1b>`.

        Returns:
            The prompt as one string.
        """
        return "".join(part if isinstance(part, str) else part.describe() for part in self.parts())


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


def join_parts(parts: Sequence[Part]) -> tuple[Part, ...]:
    """Joins adjacent text and drops empty strings, keeping media in place.

    Args:
        parts: Text and media in reading order.

    Returns:
        The same content with no two strings next to each other.
    """
    joined: list[Part] = []
    for part in parts:
        if isinstance(part, str):
            if not part:
                continue
            if joined and isinstance(joined[-1], str):
                joined[-1] += part
                continue
        joined.append(part)
    return tuple(joined)


def text_only(parts: Sequence[Part]) -> str:
    """Keeps the text of a prompt, leaving out its media.

    Args:
        parts: Text and media in reading order.

    Returns:
        The text parts joined together.
    """
    return "".join(part for part in parts if isinstance(part, str))


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


def _format_text(text: str, spec: str, conversion: str | None) -> str:
    field = "{0" + (f"!{conversion}" if conversion else "") + (f":{spec}" if spec else "") + "}"
    return field.format(text)


def _normalise(log_probs: torch.Tensor) -> torch.Tensor:
    return log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)
