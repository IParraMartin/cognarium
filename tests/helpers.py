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
    "gptj": {
        **_GPT,
        "rotary_dim": 4
    },
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
    "llama": {
        **_DECODER,
        "intermediate_size": 32
    },
    "mistral": _KEY_VALUE,
    "gemma": {
        **_KEY_VALUE,
        "head_dim": 8
    },
    "gemma2": {
        **_KEY_VALUE,
        "head_dim": 8
    },
    "qwen2": _KEY_VALUE,
    "phi3": _KEY_VALUE,
    "gpt_neox": {
        **_DECODER,
        "intermediate_size": 32
    },
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
        fail_after=None,
        logits=None,
        first_token_ids=None
    ):
        self.logits = logits
        self.first_token_ids = first_token_ids
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
        logits = None
        if capture_logits and self.logits is not None:
            logits = self.logits.expand(len(queries), -1).clone()
        return Responses(scores=scores, choices=choices, logits=logits)

    def first_tokens(self, options):
        return self.first_token_ids


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


_OMNI_TOKENS = {
    "image_token": "<|IMAGE|>",
    "audio_token": "<|AUDIO|>",
    "video_token": "<|VIDEO|>",
    "vision_bos_token": "<|vision_bos|>",
    "vision_eos_token": "<|vision_eos|>",
    "audio_bos_token": "<|audio_bos|>",
    "audio_eos_token": "<|audio_eos|>"
}
_VISION_TOKENS = ("<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>")
_SMALL_IMAGES = {"shortest_edge": 32 * 32, "longest_edge": 64 * 64}
_SMALL_VIDEOS = {"shortest_edge": 32 * 32, "longest_edge": 4 * 64 * 64}


def media_tokenizer(special, named=None):
    """Builds a byte-level tokenizer that also knows the given special tokens."""
    from tokenizers import (
        Tokenizer,
        decoders,
        models,
        pre_tokenizers
    )
    from transformers import PreTrainedTokenizerFast

    symbols = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocabulary = {token: index for index, token in enumerate(["<|endoftext|>", *special, *symbols])}
    backend = Tokenizer(models.BPE(vocab=vocabulary, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        extra_special_tokens=named or {}
    )
    tokenizer.add_special_tokens({"additional_special_tokens": list(special)})
    return tokenizer


def tiny_vision_model():
    """Builds a random two-layer Qwen3.5 that reads images and video, with its processor."""
    from transformers import (
        Qwen2VLImageProcessor,
        Qwen3_5Config,
        Qwen3_5ForConditionalGeneration,
        Qwen3VLProcessor,
        Qwen3VLVideoProcessor
    )

    tokenizer = media_tokenizer(_VISION_TOKENS)
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(patch_size=16, size=_SMALL_IMAGES),
        video_processor=Qwen3VLVideoProcessor(patch_size=16, size=_SMALL_VIDEOS),
        tokenizer=tokenizer
    )
    ids = tokenizer.convert_tokens_to_ids
    config = Qwen3_5Config(
        text_config={
            "vocab_size": len(tokenizer),
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "max_position_embeddings": 4096,
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [2, 1, 1],
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0
            },
            "eos_token_id": 0,
            "pad_token_id": 0
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": 16,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 64,
            "deepstack_visual_indexes": []
        },
        image_token_id=ids("<|image_pad|>"),
        video_token_id=ids("<|video_pad|>"),
        vision_start_token_id=ids("<|vision_start|>"),
        vision_end_token_id=ids("<|vision_end|>")
    )
    torch.manual_seed(0)
    return Qwen3_5ForConditionalGeneration(config).eval(), processor


def tiny_omni_model():
    """Builds a random Qwen2.5-Omni thinker that reads images, video and audio, with its processor."""
    from transformers import (
        Qwen2_5OmniProcessor,
        Qwen2_5OmniThinkerConfig,
        Qwen2_5OmniThinkerForConditionalGeneration,
        Qwen2VLImageProcessor,
        Qwen2VLVideoProcessor,
        WhisperFeatureExtractor
    )

    tokenizer = media_tokenizer(tuple(_OMNI_TOKENS.values()), _OMNI_TOKENS)
    processor = Qwen2_5OmniProcessor(
        image_processor=Qwen2VLImageProcessor(patch_size=16, size=_SMALL_IMAGES),
        video_processor=Qwen2VLVideoProcessor(patch_size=16, size=_SMALL_VIDEOS, cap_pixels_per_frame=True),
        feature_extractor=WhisperFeatureExtractor(feature_size=16, sampling_rate=16000),
        tokenizer=tokenizer
    )
    ids = tokenizer.convert_tokens_to_ids
    config = Qwen2_5OmniThinkerConfig(
        audio_config={
            "num_mel_bins": 16,
            "encoder_layers": 1,
            "encoder_attention_heads": 2,
            "encoder_ffn_dim": 32,
            "d_model": 16,
            "output_dim": 16,
            "n_window": 50
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": 16,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "window_size": 64,
            "fullatt_block_indexes": [0]
        },
        text_config={
            "vocab_size": len(tokenizer),
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "max_position_embeddings": 8192,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "mrope_section": [2, 1, 1]
            }
        },
        audio_token_index=ids("<|AUDIO|>"),
        image_token_index=ids("<|IMAGE|>"),
        video_token_index=ids("<|VIDEO|>"),
        audio_start_token_id=ids("<|audio_bos|>"),
        audio_end_token_id=ids("<|audio_eos|>")
    )
    # the thinker reads these when it places media, though its config doesn't declare them
    config.vision_start_token_id = ids("<|vision_bos|>")
    config.vision_end_token_id = ids("<|vision_eos|>")
    torch.manual_seed(0)
    return Qwen2_5OmniThinkerForConditionalGeneration(config).eval(), processor
