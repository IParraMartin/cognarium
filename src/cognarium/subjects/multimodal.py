"""Hugging Face models that read images, video or audio along with text."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch

from cognarium.errors import MissingAbilityError
from cognarium.media import MEDIA_KINDS, Audio, Image, Media, Video
from cognarium.recording.layers import LayerMap
from cognarium.responses import Query
from cognarium.subjects.hf import HFSubject

# the sub-processor that handles each kind of media, under the names processors use
_SUB_PROCESSORS = {
    "image": ("image_processor",),
    "video": ("video_processor",),
    "audio": ("feature_extractor", "audio_processor")
}
# tokens that open and close a medium in a prompt, for processors that expect them
_WRAPPERS = {
    "image": (("vision_start_token", "vision_end_token"), ("vision_bos_token", "vision_eos_token")),
    "video": (("vision_start_token", "vision_end_token"), ("vision_bos_token", "vision_eos_token")),
    "audio": (("audio_bos_token", "audio_eos_token"), ("audio_start_token", "audio_end_token"))
}
# a token some processors expand into a whole medium, in place of the bare medium token
_OPENERS = {
    "image": "boi_token",
    "audio": "boa_token"
}
# processor outputs that run along the tokens, which grow with the options
_TOKEN_ALIGNED = ("token_type_ids", "mm_token_type_ids")
_MARKS = ("\x00cg-before\x00", "\x00cg-after\x00")
# cached and uncached option scores may differ by rounding, which half precision makes larger
_CACHE_TOLERANCE = {
    torch.float32: 1e-3,
    torch.float64: 1e-6
}
_HALF_TOLERANCE = 0.1


@dataclass
class _Encoded:
    """One prompt as the model reads it: token ids, inputs along the tokens, and media."""

    ids: list[int]
    aligned: dict[str, torch.Tensor] = field(default_factory=dict)
    media: dict[str, Any] = field(default_factory=dict)


class HFMultimodalSubject(HFSubject):
    """A Hugging Face model that reads images, video or audio along with text.

    Prompts are written by the paradigm's template, which places each medium among
    the text, as in `"Color: {image} -> Label:"`. Each prompt goes through the
    model's processor, which turns media into the model's inputs, and options are
    scored as text after the prompt, exactly as for a text model. Hidden states and
    the logit lens read the language model's blocks, so they cover media tokens too.

    Multi-token options run the prompt once and continue from its cache. Because a
    model may place media in positions that don't survive that, the first such
    query is also scored the slow way, and if the two disagree, every later query is
    scored the slow way, with a warning.

    The raw `model`, `processor` and `tokenizer` stay available as attributes.

    Args:
        model: The model. It is moved to `device` and set to eval mode.
        processor: Its processor, which holds the tokenizer and what turns each kind
            of media into inputs.
        device: Where the model runs.
        batch_size: How many sequences go through the model at once.
        option_prefix: What separates a prompt from its option. Defaults to a space,
            or to nothing in chat format.
        revision: The Hub commit the model came from, if it came from the Hub.
        layer_map: Where the language model keeps its blocks, final norm and head,
            for models that can't be detected.
        chat: Whether to send each prompt as one user message through the chat
            template, as instruction-tuned models expect.
        media_tokens: The text that stands for each kind of medium in a prompt, which
            the processor expands, such as `{"image": "<image>"}`. Detected from the
            processor when left out.
        processor_kwargs: Passed to every processor call, such as
            `{"images_kwargs": {"max_pixels": 64 * 64}}` to keep images small.

    Raises:
        ValueError: If the processor has no tokenizer, or `chat` is on and there is no
            chat template.
        MissingAbilityError: If the processor handles no kind of media.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        device: str = "auto",
        batch_size: int = 8,
        option_prefix: str | None = None,
        revision: str | None = None,
        layer_map: LayerMap | None = None,
        chat: bool = False,
        media_tokens: Mapping[str, str] | None = None,
        processor_kwargs: Mapping[str, Any] | None = None
    ) -> None:
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise ValueError(f"{type(processor).__name__} has no tokenizer, so it can't write prompts.")
        if chat and _chat_source(processor) is None:
            raise ValueError(
                "This processor has no chat template, so prompts can't be sent in chat format. "
                "Load an instruction-tuned model, or leave chat off."
            )
        super().__init__(
            model,
            tokenizer,
            device=device,
            batch_size=batch_size,
            option_prefix=("" if chat else " ") if option_prefix is None else option_prefix,
            revision=revision,
            layer_map=layer_map
        )
        self.chat = chat
        self.processor = processor
        self.processor_kwargs = dict(processor_kwargs or {})
        self.media_tokens = find_media_tokens(processor, media_tokens)
        if not self.media_tokens:
            raise MissingAbilityError(
                f"{type(processor).__name__} handles no images, video or audio. Load the model with cg.load(name) "
                "as a text model instead."
            )
        self.modalities = frozenset({"text", *self.media_tokens})
        self._encoded: dict[int, _Encoded] = {}
        self._cache_trusted: bool | None = None

    def config(self) -> dict[str, Any]:
        """Describes the model and how its prompts are written.

        Returns:
            What a text model reports, with the processor's class, the text standing
            for each medium, and any processor settings.
        """
        return {
            **super().config(),
            "processor": type(self.processor).__name__,
            "media_tokens": dict(sorted(self.media_tokens.items())),
            **({"processor_kwargs": _json_safe(self.processor_kwargs)} if self.processor_kwargs else {})
        }

    @contextmanager
    def _prompts(self) -> Iterator[Callable[[int, Query], list[int]]]:
        """Encodes each query through the processor and keeps its media for the call."""

        def encode(index: int, query: Query) -> list[int]:
            self._encoded[index] = self._encode(query)
            return self._encoded[index].ids

        self._encoded = {}
        try:
            yield encode
        finally:
            self._encoded = {}

    def _encode(self, query: Query) -> _Encoded:
        parts = query.parts()
        media = [part for part in parts if isinstance(part, Media)]
        missing = sorted({medium.kind for medium in media} - set(self.media_tokens))
        if missing:
            raise MissingAbilityError(
                f"{type(self.model).__name__} can't read {missing}. It reads {sorted(self.modalities)}."
            )
        if self.chat:
            content = [
                {"type": "text", "text": part} if isinstance(part, str) else {"type": part.kind}
                for part in parts
            ]
            text = _chat_source(self.processor).apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        else:
            text = "".join(part if isinstance(part, str) else self.media_tokens[part.kind] for part in parts)
        inputs = self._process(text, media, add_special_tokens=not self.chat)
        ids = inputs.pop("input_ids")[0].tolist()
        inputs.pop("attention_mask", None)
        encoded = _Encoded(ids)
        for key, value in inputs.items():
            if key in _TOKEN_ALIGNED:
                encoded.aligned[key] = value[0]
            else:
                encoded.media[key] = value
        return encoded

    def _process(self, text: str, media: Sequence[Media], *, add_special_tokens: bool) -> dict[str, Any]:
        settings: dict[str, Any] = {
            "text": [text],
            "return_tensors": "pt",
            "add_special_tokens": add_special_tokens,
            **self.processor_kwargs
        }
        images = [medium.load() for medium in media if isinstance(medium, Image)]
        if images:
            settings["images"] = images
        videos = [medium.load() for medium in media if isinstance(medium, Video)]
        if videos:
            settings["videos"] = [frames for frames, _ in videos]
            rates = [fps for _, fps in videos]
            if all(rate is not None for rate in rates):
                settings["video_metadata"] = [
                    {"fps": rate, "total_num_frames": len(frames)}
                    for (frames, _), rate in zip(videos, rates, strict=True)
                ]
        sounds = [medium for medium in media if isinstance(medium, Audio)]
        if sounds:
            rate = _sampling_rate(self.processor)
            settings["audio"] = [sound.load(rate) for sound in sounds]
            if rate is not None:
                settings["sampling_rate"] = rate
        return dict(self.processor(**settings))

    def _forward(self, ids: torch.Tensor, mask: torch.Tensor, rows: Sequence[int] | None = None) -> torch.Tensor:
        extra = {} if rows is None else self._row_inputs(rows, ids.shape[1])
        return self.model(input_ids=ids, attention_mask=mask, use_cache=False, **extra).logits

    def _prefill_inputs(self, index: int, prompt: list[int]) -> dict[str, Any]:
        return {
            "input_ids": torch.tensor([prompt], device=self.device),
            "attention_mask": torch.ones((1, len(prompt)), dtype=torch.long, device=self.device),
            **self._row_inputs([index], len(prompt))
        }

    def _continue_inputs(self, ids: torch.Tensor, mask: torch.Tensor, seen: int) -> dict[str, Any]:
        """Continues without a mask, so models that place media in positions count on from the cache.

        Option rows are padded on the right and attention is causal, so no real token
        sees the padding.
        """
        return {"input_ids": ids}

    def _row_inputs(self, rows: Sequence[int], width: int) -> dict[str, Any]:
        """Collects what the given rows show besides their token ids.

        Inputs that run along the tokens are padded with zeros to `width`, which also
        covers option and generated tokens after the prompt. Media inputs are joined
        in row order, as processors join the media of a batch.
        """
        encoded = [self._encoded[row] for row in rows]
        dtype = next(self.model.parameters()).dtype
        inputs: dict[str, Any] = {}
        for key in sorted({key for item in encoded for key in item.aligned}):
            padded = torch.zeros((len(rows), width), dtype=torch.long)
            for row, item in enumerate(encoded):
                if key in item.aligned:
                    values = item.aligned[key][:width]
                    padded[row, : len(values)] = values
            inputs[key] = padded.to(self.device)
        for key in sorted({key for item in encoded for key in item.media}):
            values = [item.media[key] for item in encoded if key in item.media]
            if all(isinstance(value, torch.Tensor) for value in values):
                joined = _join(values)
                if joined.is_floating_point():
                    joined = joined.to(dtype)
                inputs[key] = joined.to(self.device)
            else:
                inputs[key] = [element for value in values for element in value]
        return inputs

    def _score_after_prompt(self, index: int, prompt: list[int], options: list[list[int]]) -> torch.Tensor | None:
        """Scores multi-token options from the prompt's cache, once checked against the slow way."""
        if self._cache_trusted is False:
            return None
        cached = super()._score_after_prompt(index, prompt, options)
        if cached is None or self._cache_trusted:
            return cached
        whole = self._score_whole(index, prompt, options)
        dtype = next(self.model.parameters()).dtype
        tolerance = _CACHE_TOLERANCE.get(dtype, _HALF_TOLERANCE)
        gap = (cached - whole).abs().max().item()
        self._cache_trusted = gap <= tolerance * max(1.0, whole.abs().max().item())
        if not self._cache_trusted:
            warnings.warn(
                f"Scoring options from the prompt's cache misses the full scores by {gap:.3g} on "
                f"{type(self.model).__name__}, so every option is scored with its whole prompt instead.",
                stacklevel=2
            )
            return whole
        return cached

    def _score_whole(self, index: int, prompt: list[int], options: list[list[int]]) -> torch.Tensor:
        """Scores each option by running it with the whole prompt."""
        scores = torch.zeros(len(options), dtype=torch.float64)
        for start in range(0, len(options), self.batch_size):
            batch = list(range(start, min(start + self.batch_size, len(options))))
            width = max(len(prompt) + len(options[number]) for number in batch)
            ids = torch.full((len(batch), width), self.tokenizer.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(batch), width), dtype=torch.long)
            for row, number in enumerate(batch):
                sequence = prompt + options[number]
                ids[row, : len(sequence)] = torch.tensor(sequence)
                mask[row, : len(sequence)] = 1
            logits = self._forward(ids.to(self.device), mask.to(self.device), [index] * len(batch))
            log_probs = logits.float().log_softmax(dim=-1).cpu().double()
            for row, number in enumerate(batch):
                tokens = torch.tensor(options[number])
                positions = torch.arange(len(prompt) - 1, len(prompt) - 1 + len(tokens))
                scores[number] = log_probs[row, positions, tokens].sum()
        return scores


def _join(values: Sequence[torch.Tensor]) -> torch.Tensor:
    """Joins media inputs along their first dimension, padding the others with zeros.

    Each prompt goes through the processor alone, so inputs a processor pads to the
    longest in its call, such as audio features and their masks, can differ in
    length between prompts. Zero padding is what the processor itself would add.
    """
    if len({value.shape[1:] for value in values}) == 1:
        return torch.cat(list(values))
    sizes = [max(value.shape[dim] for value in values) for dim in range(1, values[0].ndim)]
    padded = []
    for value in values:
        widths = []
        for dim in reversed(range(1, value.ndim)):
            widths.extend([0, sizes[dim - 1] - value.shape[dim]])
        padded.append(torch.nn.functional.pad(value, widths))
    return torch.cat(padded)


def find_media_tokens(processor: Any, given: Mapping[str, str] | None = None) -> dict[str, str]:
    """Finds the text that stands for each kind of medium in a prompt.

    A processor expands that text into the model's media tokens. For each kind the
    processor handles, `given` is used first. Otherwise the text is read from the
    chat template, which places media the way the model was trained on, then built
    from the processor's own medium tokens.

    Args:
        processor: The model's processor.
        given: Text chosen by the user, by kind.

    Returns:
        The text for each kind of medium the processor handles and can place.

    Raises:
        ValueError: If `given` names an unknown kind of medium.
    """
    given = dict(given or {})
    unknown = sorted(set(given) - set(MEDIA_KINDS))
    if unknown:
        raise ValueError(f"media_tokens has unknown kinds {unknown}. Choose from {list(MEDIA_KINDS)}.")
    found = {}
    for kind in MEDIA_KINDS:
        if kind in given:
            found[kind] = given[kind]
            continue
        if not any(getattr(processor, name, None) is not None for name in _SUB_PROCESSORS[kind]):
            continue
        token = _from_chat_template(processor, kind) or _from_attributes(processor, kind)
        if token:
            found[kind] = token
    return found


def _from_chat_template(processor: Any, kind: str) -> str | None:
    source = _chat_source(processor)
    if source is None:
        return None
    before, after = _MARKS
    content = [{"type": "text", "text": before}, {"type": kind}, {"type": "text", "text": after}]
    try:
        written = source.apply_chat_template([{"role": "user", "content": content}], tokenize=False)
    except Exception:
        return None
    if not isinstance(written, str) or before not in written or after not in written:
        return None
    token = written.split(before, 1)[1].split(after, 1)[0].strip()
    return token or None


def _from_attributes(processor: Any, kind: str) -> str | None:
    opener = _OPENERS.get(kind)
    if opener is not None and getattr(processor, opener, None):
        return getattr(processor, opener)
    token = getattr(processor, f"{kind}_token", None)
    if not token:
        return None
    for start, end in _WRAPPERS[kind]:
        first = getattr(processor, start, None) or getattr(processor.tokenizer, start, None)
        last = getattr(processor, end, None) or getattr(processor.tokenizer, end, None)
        if first and last:
            return f"{first}{token}{last}"
    return token


def _chat_source(processor: Any) -> Any:
    if getattr(processor, "chat_template", None):
        return processor
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return tokenizer
    return None


def _sampling_rate(processor: Any) -> int | None:
    for name in _SUB_PROCESSORS["audio"]:
        rate = getattr(getattr(processor, name, None), "sampling_rate", None)
        if rate is not None:
            return int(rate)
    return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)
