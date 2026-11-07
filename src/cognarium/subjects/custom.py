"""Custom PyTorch modules, including the bundled LSTM, as subjects."""

from __future__ import annotations

import copy
import warnings
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from cognarium.device import resolve_device
from cognarium.errors import UntrainedModelWarning
from cognarium.models.lstm import LSTMLanguageModel, SimpleTokenizer
from cognarium.recording.capture import StateCapture, hook_layers
from cognarium.responses import PromptTemplate, Query, Responses
from cognarium.stimuli import Observation
from cognarium.subjects.base import Readout, Subject, weights_digest
from cognarium.subjects.scoring import pad_sequences, score_options

if TYPE_CHECKING:
    from cognarium.paradigms.base import Paradigm

_IGNORED_TARGET = -100


class CustomSubject(Subject):
    """Any PyTorch module that outputs next-token logits.

    The raw `model` stays available as an attribute.

    Args:
        model: The module. By default it is called as `model(input_ids)` and may
            return logits, a tuple starting with logits, or an object with a
            `.logits` attribute.
        tokenizer: Needs `encode(text)` and `pad_token_id`.
        adapter: Maps `(model, input_ids, attention_mask)` to logits of shape
            (batch, time, vocab), for modules the default call doesn't fit.
        pretrained: Whether the module's weights are already trained. If not,
            `fresh` draws new weights from its seed, and `learning="auto"` picks
            `weights`.
        device: Where the module runs.
        batch_size: How many sequences go through the module at once.
        option_prefix: What separates a prompt from its option.
        train_epochs: Full-batch training steps per `learn` call.
        learning_rate: The Adam learning rate used by `learn`.
        layers: The layers that can be recorded, as a name for each and the path of
            the submodule that computes it.
        readout: The path of the submodule that turns states into logits, which the
            logit lens needs.
    """

    modalities = frozenset({"text"})
    learning_modes = frozenset({"in_context", "weights"})

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        *,
        adapter: Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        pretrained: bool = True,
        device: str = "auto",
        batch_size: int = 32,
        option_prefix: str = " ",
        train_epochs: int = 50,
        learning_rate: float = 0.01,
        layers: Mapping[str, str] | None = None,
        readout: str | None = None
    ) -> None:
        self.device = resolve_device(device)
        # a cpu copy, so starting weights never depend on the device
        self._starting = copy.deepcopy(model).cpu()
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.adapter = adapter
        self.pretrained = pretrained
        self.batch_size = batch_size
        self.option_prefix = option_prefix
        self.train_epochs = train_epochs
        self.learning_rate = learning_rate
        self.layers = dict(layers or {})
        self.readout_path = readout

    @property
    def default_learning(self) -> str:
        """The mode `learning="auto"` picks: `weights` unless already trained."""
        return "in_context" if self.pretrained else "weights"

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
            capture_states: Where to send the output of each requested layer.

        Returns:
            One row of scores per query.
        """
        self.model.eval()
        modules = {name: self.model.get_submodule(path) for name, path in self.layers.items()}
        hooks = nullcontext() if capture_states is None else hook_layers(modules, capture_states)
        with hooks as on_batch:
            return score_options(
                queries,
                forward=self._forward,
                encode_prompt=self.tokenizer.encode,
                encode_option=self._encode_option,
                pad_token_id=self.tokenizer.pad_token_id,
                device=self.device,
                unk_token_id=getattr(self.tokenizer, "unk_token_id", None),
                batch_size=self.batch_size,
                option_prefix=self.option_prefix,
                capture_logits=capture_logits,
                on_batch=on_batch
            )

    def state_layers(self) -> tuple[str, ...]:
        """Names the layers that can be recorded.

        Returns:
            The names passed as `layers`.
        """
        return tuple(self.layers)

    def readout(self) -> Readout | None:
        """Gives the submodule that turns states into logits.

        Returns:
            The readout, or None if no `readout` path was passed.
        """
        if self.readout_path is None:
            return None
        head = self.model.get_submodule(self.readout_path)
        return Readout(
            head=head,
            decode=getattr(self.tokenizer, "decode", str),
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

    def bind(self, paradigm: Paradigm, learning: str | None) -> CustomSubject:
        """Warns when an untrained model is asked to use its context.

        Args:
            paradigm: The paradigm about to run.
            learning: The learning mode the run will use.

        Returns:
            This subject, or one with reproducible starting weights if it is
            untrained and reading its context.
        """
        if learning == "in_context" and not self.pretrained:
            warnings.warn(
                "Running in_context on untrained weights. Train the model first, or use "
                "learning='weights'.",
                UntrainedModelWarning,
                stacklevel=2
            )
            return self.fresh((paradigm.settings.seed,))
        return self

    def fresh(self, seed: tuple[int, ...]) -> CustomSubject:
        """Starts a new learner from the starting weights.

        Args:
            seed: Seeds the new weights when the module isn't pretrained.

        Returns:
            A new subject. This one is left unchanged.

        Raises:
            ValueError: If an untrained module can't reset its parameters.
        """
        model = copy.deepcopy(self._starting)
        if not self.pretrained:
            _reinitialise(model, seed)
        clone = copy.copy(self)
        clone.model = model.to(self.device)
        return clone

    def learn(
        self,
        observations: Sequence[Observation],
        *,
        template: PromptTemplate,
        instructions: str | None = None
    ) -> None:
        """Trains on labelled observations by predicting each next token.

        Args:
            observations: What to learn from.
            template: How observations are written out.
            instructions: Text placed before each observation.
        """
        lead = "" if instructions is None else instructions + template.separator
        lines = [lead + template.render_observation(observation) for observation in observations]
        pad_token_id = self.tokenizer.pad_token_id
        encoded = [self.tokenizer.encode(line) for line in lines]
        ids, mask = pad_sequences(encoded, pad_token_id, self.device)
        targets = ids[:, 1:].masked_fill(mask[:, 1:] == 0, _IGNORED_TARGET)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model.train()
        for _ in range(self.train_epochs):
            optimizer.zero_grad()
            logits = self._forward(ids[:, :-1], mask[:, :-1])
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=_IGNORED_TARGET
            )
            loss.backward()
            optimizer.step()
        self.model.eval()

    def config(self) -> dict[str, Any]:
        """Describes the module, including its weights if they're pretrained.

        Returns:
            The architecture, training settings, and vocabulary size.
        """
        described = {
            "class": type(self).__name__,
            "architecture": repr(self._starting),
            "train_epochs": self.train_epochs,
            "learning_rate": self.learning_rate,
            "option_prefix": self.option_prefix,
            "vocabulary": len(self.tokenizer),
            "pretrained": self.pretrained
        }
        if self.pretrained:
            described["weights"] = weights_digest(self._starting)
        return described

    def runtime(self) -> dict[str, Any]:
        """Describes how the module runs.

        Returns:
            The device and batch size.
        """
        return {"device": str(self.device), "batch_size": self.batch_size}

    def _encode_option(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        adapter = _default_adapter if self.adapter is None else self.adapter
        output = adapter(self.model, ids, mask)
        if not isinstance(output, torch.Tensor) or output.ndim != 3:
            raise ValueError("The adapter must return logits of shape (batch, time, vocab).")
        return output


class LSTM(Subject):
    """An untrained LSTM that builds its vocabulary from the paradigm it runs on.

    To use your own trained LSTM, pass it to `cg.load` with its tokenizer instead.

    Args:
        embedding_dim: The size of the token embeddings.
        hidden_dim: The size of the LSTM's hidden state.
        num_layers: How many LSTM layers to stack.
        device: Where the model runs.
        batch_size: How many sequences go through the model at once.
        train_epochs: Full-batch training steps per generation.
        learning_rate: The Adam learning rate.
    """

    modalities = frozenset({"text"})
    learning_modes = frozenset({"in_context", "weights"})
    default_learning = "weights"

    def __init__(
        self,
        *,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 1,
        device: str = "auto",
        batch_size: int = 32,
        train_epochs: int = 50,
        learning_rate: float = 0.01
    ) -> None:
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        self.batch_size = batch_size
        self.train_epochs = train_epochs
        self.learning_rate = learning_rate

    def state_layers(self) -> tuple[str, ...]:
        """Names the layers that can be recorded, which exist once this is bound.

        Returns:
            `lstm`, which holds the LSTM's hidden state.
        """
        return ("lstm",)

    def bind(self, paradigm: Paradigm, learning: str | None) -> CustomSubject:
        """Builds the model with a vocabulary covering the paradigm's text.

        Args:
            paradigm: The paradigm about to run.
            learning: The learning mode the run will use.

        Returns:
            An untrained subject, ready to run.
        """
        tokenizer = SimpleTokenizer.from_texts(paradigm.texts())
        model = LSTMLanguageModel(
            len(tokenizer),
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        )
        subject = CustomSubject(
            model,
            tokenizer,
            pretrained=False,
            device=self.device,
            batch_size=self.batch_size,
            train_epochs=self.train_epochs,
            learning_rate=self.learning_rate,
            layers={"lstm": "lstm"},
            readout="readout"
        )
        return subject.bind(paradigm, learning)

    def respond(
        self,
        queries: Sequence[Query],
        *,
        capture_logits: bool = False,
        capture_states: StateCapture | None = None
    ) -> Responses:
        """Always raises, because the model only exists once it's bound.

        Args:
            queries: Unused.
            capture_logits: Unused.
            capture_states: Unused.

        Raises:
            RuntimeError: Always. `cg.run` binds subjects before calling them.
        """
        raise RuntimeError("Call bind(paradigm, learning) first, or let cg.run do it.")


def _default_adapter(model: torch.nn.Module, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    output = model(ids)
    if isinstance(output, tuple):
        output = output[0]
    return getattr(output, "logits", output)


def _reinitialise(model: torch.nn.Module, seed: tuple[int, ...]) -> None:
    derived = int(np.random.SeedSequence(list(seed)).generate_state(1)[0])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(derived)
        for module in model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
            elif any(True for _ in module.parameters(recurse=False)):
                raise ValueError(
                    f"{type(module).__name__} has parameters but no reset_parameters, so it can't "
                    "draw starting weights. Pass trained weights with pretrained=True."
                )
