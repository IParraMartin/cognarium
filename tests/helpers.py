from __future__ import annotations

from collections import defaultdict

import torch

from cognarium.responses import Responses
from cognarium.subjects.base import Subject

_DECODER = {
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "hidden_size": 16,
    "max_position_embeddings": 64
}
_GPT = {
    "n_layer": 2,
    "n_head": 2,
    "n_embd": 16,
    "n_positions": 64
}
_KEY_VALUE = {
    **_DECODER,
    "intermediate_size": 32,
    "num_key_value_heads": 2
}
TINY_ARCHITECTURES = {
    "gpt2": _GPT,
    "gptj": {**_GPT, "rotary_dim": 4},
    "falcon": {
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "hidden_size": 16
    },
    "bloom": {
        "n_layer": 2,
        "n_head": 2,
        "hidden_size": 16
    },
    "llama": {**_DECODER, "intermediate_size": 32},
    "mistral": _KEY_VALUE,
    "gemma": {**_KEY_VALUE, "head_dim": 8},
    "gemma2": {**_KEY_VALUE, "head_dim": 8},
    "qwen2": _KEY_VALUE,
    "phi3": _KEY_VALUE,
    "gpt_neox": {**_DECODER, "intermediate_size": 32},
    "opt": {
        **_DECODER,
        "ffn_dim": 32,
        "word_embed_proj_dim": 16
    }
}


def tiny_architecture(name, tokenizer):
    """Builds a small random model of a real architecture, without network access."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.for_model(
        name,
        vocab_size=len(tokenizer),
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
        **TINY_ARCHITECTURES[name]
    )
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(config).eval()


class RecordingCapture:
    """Stands in for a state capture and keeps everything it was given."""

    def __init__(self, layers=("layer_00", "layer_01"), pool="last"):
        self.layers = tuple(layers)
        self.pool = pool
        self.written = defaultdict(list)

    def write(
        self,
        layer,
        states,
        queries,
        lengths=None
    ):
        kept_lengths = None if lengths is None else lengths.cpu()
        self.written[layer].append((states.cpu().clone(), list(queries), kept_lengths))

    def queries(self, layer):
        return [query for _, queries, _ in self.written[layer] for query in queries]

    def stacked(self, layer):
        """Gives one layer's states in query order."""
        rows = torch.cat([states for states, _, _ in self.written[layer]])
        return rows[torch.tensor(self.queries(layer)).argsort()]


class ScriptedSubject(Subject):
    """Gives fixed probabilities or choices, and counts how often it is called."""

    def __init__(
        self,
        probabilities=(0.5, 0.5),
        choices=None,
        modes=("in_context",),
        modalities=("text",),
        fail_after=None
    ):
        self.probabilities = probabilities
        self.choices = choices
        self.learning_modes = frozenset(modes)
        self.modalities = frozenset(modalities)
        self.default_learning = next(iter(self.learning_modes), None)
        self.fail_after = fail_after
        self.calls = 0

    def respond(self, queries, *, capture_logits=False):
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError("scripted failure")
        self.calls += 1
        row = torch.tensor(self.probabilities, dtype=torch.float64).log()
        scores = row.expand(len(queries), -1).clone()
        choices = None
        if self.choices is not None:
            choices = torch.full((len(queries),), self.choices, dtype=torch.long)
        return Responses(scores=scores, choices=choices)


class TrainableScriptedSubject(ScriptedSubject):
    """A scripted subject that keeps state, and records how it was trained."""

    def __init__(self, **kwargs):
        super().__init__(modes=("weights",), **kwargs)
        self.fresh_seeds = []
        self.learned = []
        self.max_context_seen = 0

    def fresh(self, seed):
        self.fresh_seeds.append(tuple(seed))
        return self

    def learn(
        self,
        observations,
        *,
        template,
        instructions=None
    ):
        self.learned.append(tuple(observations))

    def respond(self, queries, *, capture_logits=False):
        longest = max(len(query.context) for query in queries)
        self.max_context_seen = max(self.max_context_seen, longest)
        return super().respond(queries, capture_logits=capture_logits)
