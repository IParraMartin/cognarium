"""Hugging Face language models as subjects."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from cognarium.device import resolve_device
from cognarium.responses import Query, Responses
from cognarium.subjects.base import Subject, weights_digest
from cognarium.subjects.scoring import score_options


class HFSubject(Subject):
    """A Hugging Face causal language model.

    The raw `model` and `tokenizer` stay available as attributes.

    Args:
        model: The language model. It is moved to `device` and set to eval mode.
        tokenizer: Its tokenizer. If it has no pad token, its EOS token is used,
            which can't change results because padding is masked.
        device: Where the model runs.
        batch_size: How many sequences go through the model at once.
        option_prefix: What separates a prompt from its option.

    Raises:
        ValueError: If the tokenizer has neither a pad token nor an EOS token.
    """

    modalities = frozenset({"text"})
    learning_modes = frozenset({"in_context"})
    default_learning = "in_context"

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        device: str = "auto",
        batch_size: int = 32,
        option_prefix: str = " "
    ) -> None:
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("The tokenizer needs a pad token or an EOS token to batch prompts.")
            tokenizer.pad_token = tokenizer.eos_token
        self.device = resolve_device(device)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.option_prefix = option_prefix

    def respond(self, queries: Sequence[Query], *, capture_logits: bool = False) -> Responses:
        """Scores each query's options after its prompt.

        Args:
            queries: The queries to answer.
            capture_logits: Whether to return logits at the last prompt position.

        Returns:
            One row of scores per query.
        """
        return score_options(
            queries,
            forward=self._forward,
            encode_prompt=lambda text: self.tokenizer(text)["input_ids"],
            encode_option=lambda text: self.tokenizer(text, add_special_tokens=False)["input_ids"],
            pad_token_id=self.tokenizer.pad_token_id,
            device=self.device,
            unk_token_id=self.tokenizer.unk_token_id,
            batch_size=self.batch_size,
            option_prefix=self.option_prefix,
            max_length=getattr(self.model.config, "max_position_embeddings", None),
            capture_logits=capture_logits
        )

    def config(self) -> dict[str, Any]:
        """Describes the model, including its weights when there is no revision.

        Returns:
            The class, model name, revision, dtype and option prefix.
        """
        revision = getattr(self.model.config, "_commit_hash", None)
        described = {
            "class": type(self).__name__,
            "model": self.model.name_or_path,
            "revision": revision,
            "dtype": str(next(self.model.parameters()).dtype),
            "option_prefix": self.option_prefix
        }
        if revision is None:
            described["weights"] = weights_digest(self.model)
        return described

    def runtime(self) -> dict[str, Any]:
        """Describes how the model runs.

        Returns:
            The device and batch size.
        """
        return {"device": str(self.device), "batch_size": self.batch_size}

    def _forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=ids, attention_mask=mask, use_cache=False).logits
