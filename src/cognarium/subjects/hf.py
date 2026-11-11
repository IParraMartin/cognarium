"""Hugging Face language models as subjects."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, nullcontext
from typing import Any

import torch

from cognarium.device import resolve_device
from cognarium.errors import MissingAbilityError
from cognarium.recording.capture import StateCapture, hook_layers, pool_states
from cognarium.recording.layers import LayerMap, find_layer_map
from cognarium.responses import Query, Responses
from cognarium.subjects.base import Readout, Subject, weights_digest
from cognarium.subjects.generation import generate_texts
from cognarium.subjects.scoring import pad_sequences, score_options

_CHECK_TOKENS = 8
_READOUT_TOLERANCE = 1e-3


class HFSubject(Subject):
    """A Hugging Face causal language model.

    The raw `model` and `tokenizer` stay available as attributes.

    Args:
        model: The language model. It is moved to `device` and set to eval mode.
        tokenizer: Its tokenizer. If it has no pad token, its EOS token is used,
            which can't change results because padding is masked.
        device: Where the model runs.
        batch_size: How many sequences go through the model at once.
        option_prefix: What separates a prompt from its option. Defaults to a space,
            or to nothing in chat format, where an option starts the reply.
        revision: The Hub commit the model came from, if it came from the Hub.
        layer_map: Where the model keeps its blocks, final norm and head, for models
            that can't be recognised or detected.
        chat: Whether to send each prompt as one user message through the
            tokenizer's chat template, as instruction-tuned models expect. The
            options, or a free response, are then the start of the assistant's
            reply, and thinking is turned off for templates that offer it.

    Raises:
        ValueError: If the tokenizer has neither a pad token nor an EOS token, or
            `chat` is on and the tokenizer has no chat template.
    """

    modalities = frozenset({"text"})
    learning_modes = frozenset({"in_context"})
    can_generate = True
    default_learning = "in_context"

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        device: str = "auto",
        batch_size: int = 32,
        option_prefix: str | None = None,
        revision: str | None = None,
        layer_map: LayerMap | None = None,
        chat: bool = False
    ) -> None:
        if chat and getattr(tokenizer, "chat_template", None) is None:
            raise ValueError(
                "This tokenizer has no chat template, so prompts can't be sent in chat format. "
                "Load an instruction-tuned model, or leave chat off."
            )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("The tokenizer needs a pad token or an EOS token to batch prompts.")
            tokenizer.pad_token = tokenizer.eos_token
        self.device = resolve_device(device)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.chat = chat
        self.option_prefix = ("" if chat else " ") if option_prefix is None else option_prefix
        self.revision = revision
        self.layer_map = layer_map

    def respond(
        self,
        queries: Sequence[Query],
        *,
        capture_logits: bool = False,
        capture_states: StateCapture | None = None
    ) -> Responses:
        """Scores each query's options after its prompt.

        Args:
            queries: The queries to answer.
            capture_logits: Whether to return logits at the last prompt position.
            capture_states: Where to send the output of each requested block.

        Returns:
            One row of scores per query.

        Raises:
            MissingAbilityError: If states are requested and the model's blocks can't
                be found.
        """
        hooks = nullcontext() if capture_states is None else hook_layers(self._blocks(), capture_states)
        with hooks as on_batch, self._prompts() as encode_query:
            return score_options(
                queries,
                forward=self._forward,
                encode_prompt=self._encode_prompt,
                encode_option=self._encode_option,
                pad_token_id=self.tokenizer.pad_token_id,
                device=self.device,
                unk_token_id=self.tokenizer.unk_token_id,
                batch_size=self.batch_size,
                option_prefix=self.option_prefix,
                max_length=self._max_length(),
                capture_logits=capture_logits,
                on_batch=on_batch,
                score_after_prompt=self._score_after_prompt,
                encode_query=encode_query
            )

    def generate(
        self,
        queries: Sequence[Query],
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int | None = None
    ) -> list[str]:
        """Writes a free response to each query.

        Args:
            queries: The queries to answer.
            max_new_tokens: How many tokens each response may use.
            temperature: Divides log probabilities before a token is sampled.
            top_p: Keeps the smallest set of tokens covering this much probability.
            top_k: Keeps only this many of the most likely tokens.

        Returns:
            One text per query.

        Raises:
            ValueError: If a prompt leaves no room for `max_new_tokens`.
        """
        with self._prompts() as encode_query:
            return generate_texts(
                queries,
                forward=self._forward,
                encode_prompt=self._encode_prompt,
                decode=lambda tokens: self.tokenizer.decode(tokens, skip_special_tokens=True),
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                device=self.device,
                batch_size=self.batch_size,
                max_length=self._max_length(),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                encode_query=encode_query
            )

    def state_layers(self) -> tuple[str, ...]:
        """Names the model's blocks.

        Returns:
            `layer_00`, `layer_01` and so on, counted from the input.

        Raises:
            MissingAbilityError: If the model's blocks can't be found.
        """
        return tuple(self._blocks())

    def readout(self) -> Readout:
        """Gives the model's final norm and output head.

        Returns:
            A readout whose head computes `head(norm(states))`, followed by the soft
            cap that some models put on their logits.

        Raises:
            MissingAbilityError: If the model's head can't be found, or reading its
                last block doesn't give its own logits, as when a model rescales them
                after the head or the layer map is wrong.
        """
        layer_map = self._layer_map()
        head = self.model.get_submodule(layer_map.head)
        norm = None if layer_map.norm is None else self.model.get_submodule(layer_map.norm)
        cap = getattr(self.model.config, "final_logit_softcapping", None)

        def read(states: torch.Tensor) -> torch.Tensor:
            logits = head(states if norm is None else norm(states))
            return logits if cap is None else torch.tanh(logits / cap) * cap

        self._check_readout(read)
        return Readout(
            head=read,
            decode=self.tokenizer.decode,
            device=self.device,
            dtype=next(head.parameters()).dtype
        )

    def first_tokens(self, options: Sequence[str]) -> list[int]:
        """Gives the token each option starts with, after the option prefix.

        Args:
            options: The response options.

        Returns:
            One token id per option.
        """
        return [self._encode_option(self.option_prefix + option)[0] for option in options]

    def config(self) -> dict[str, Any]:
        """Describes the model, including a fingerprint of its weights.

        The fingerprint is there even when the revision is known, because `model`
        stays available and may have been edited since it was loaded.

        Returns:
            The class, model name, revision, dtype, option prefix and weights, and
            `chat` when prompts go through the chat template.
        """
        return {
            "class": type(self).__name__,
            "model": self.model.name_or_path,
            "revision": self.revision,
            "dtype": str(next(self.model.parameters()).dtype),
            "option_prefix": self.option_prefix,
            "weights": weights_digest(self.model),
            **({"chat": True} if self.chat else {})
        }

    def runtime(self) -> dict[str, Any]:
        """Describes how the model runs.

        Returns:
            The device and batch size.
        """
        return {"device": str(self.device), "batch_size": self.batch_size}

    @contextmanager
    def _prompts(self) -> Iterator[Callable[[int, Query], list[int]] | None]:
        """Opens a call's prompts. A text model encodes each prompt from its text alone."""
        yield None

    def _forward(self, ids: torch.Tensor, mask: torch.Tensor, rows: Sequence[int] | None = None) -> torch.Tensor:
        return self.model(input_ids=ids, attention_mask=mask, use_cache=False).logits

    def _prefill_inputs(self, index: int, prompt: list[int]) -> dict[str, Any]:
        """Gives the model's inputs for one query's prompt alone."""
        return {"input_ids": torch.tensor([prompt], device=self.device)}

    def _continue_inputs(self, ids: torch.Tensor, mask: torch.Tensor, seen: int) -> dict[str, Any]:
        """Gives the inputs for option tokens that continue from a prompt's cache."""
        prompt_mask = torch.ones((len(ids), seen), dtype=mask.dtype, device=self.device)
        return {"input_ids": ids, "attention_mask": torch.cat([prompt_mask, mask], dim=1)}

    def _max_length(self) -> int | None:
        config = self.model.config
        found = getattr(config, "max_position_embeddings", None)
        if found is None:
            found = getattr(getattr(config, "text_config", None), "max_position_embeddings", None)
        return found

    def _score_after_prompt(self, index: int, prompt: list[int], options: list[list[int]]) -> torch.Tensor | None:
        """Scores multi-token options by running the prompt once and reusing its cache.

        Each batch of options continues from a copy of the prompt's cache, so a long
        prompt with many options, such as the points of a numeric scale, costs one
        pass over the prompt instead of one per option.

        Args:
            index: The query's position in the call.
            prompt: The prompt's token ids.
            options: Each option's token ids.

        Returns:
            One summed log probability per option, or None if the model keeps no cache
            that can be copied, in which case the caller scores the options the slow way.
        """
        output = self.model(**self._prefill_inputs(index, prompt), use_cache=True)
        cache = getattr(output, "past_key_values", None)
        if cache is None or not hasattr(cache, "reorder_cache"):
            return None
        first = output.logits[0, -1].float().log_softmax(dim=-1).cpu().double()
        scores = torch.tensor([first[tokens[0]].item() for tokens in options], dtype=torch.float64)
        longer = [number for number, tokens in enumerate(options) if len(tokens) > 1]
        for start in range(0, len(longer), self.batch_size):
            batch = longer[start : start + self.batch_size]
            copied = copy.deepcopy(cache)
            # every row of the batch continues from the same prompt
            copied.reorder_cache(torch.zeros(len(batch), dtype=torch.long, device=self.device))
            ids, mask = pad_sequences(
                [options[number][:-1] for number in batch],
                self.tokenizer.pad_token_id,
                self.device
            )
            log_probs = self.model(
                **self._continue_inputs(ids, mask, len(prompt)),
                past_key_values=copied,
                use_cache=True
            ).logits.float().log_softmax(dim=-1).cpu().double()
            for row, number in enumerate(batch):
                rest = torch.tensor(options[number][1:])
                scores[number] += log_probs[row, torch.arange(len(rest)), rest].sum()
        return scores

    def _encode_prompt(self, text: str) -> list[int]:
        if not self.chat:
            return self.tokenizer(text)["input_ids"]
        written = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        # the template already holds any special tokens the model starts with
        return self.tokenizer(written, add_special_tokens=False)["input_ids"]

    def _encode_option(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def _check_readout(self, read: Callable[[torch.Tensor], torch.Tensor]) -> None:
        # a model can do anything to its logits after the head, so this looks at the result
        vocabulary = self.model.get_input_embeddings().num_embeddings
        ids = (torch.arange(_CHECK_TOKENS, device=self.device) % vocabulary)[None]
        mask = torch.ones_like(ids)
        states = []
        last_block = list(self._blocks().values())[-1]
        handle = last_block.register_forward_hook(
            lambda _module, _inputs, output: states.append(pool_states(output, mask, None)[0])
        )
        try:
            with torch.inference_mode():
                logits = self._forward(ids, mask)
                try:
                    gap = (read(states[0]) - logits).abs().max().item()
                except RuntimeError as error:
                    raise MissingAbilityError(
                        f"The head of {type(self.model).__name__} doesn't fit its last block's states, "
                        f"so the logit lens can't read them: {error}"
                    ) from error
        finally:
            handle.remove()
        if gap > _READOUT_TOLERANCE * logits.abs().max().item():
            raise MissingAbilityError(
                f"Reading the last block of {type(self.model).__name__} misses its own logits by {gap:.3g}, so the "
                "logit lens would be wrong. The model may change its logits after its head, or its layer map is wrong."
            )

    def _layer_map(self) -> LayerMap:
        return self.layer_map if self.layer_map is not None else find_layer_map(self.model)

    def _blocks(self) -> dict[str, torch.nn.Module]:
        blocks = self.model.get_submodule(self._layer_map().layers)
        return {f"layer_{index:02d}": block for index, block in enumerate(blocks)}
