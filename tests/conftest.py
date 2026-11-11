from __future__ import annotations

import pytest
import torch


@pytest.fixture(
    params=[
        "cpu",
        pytest.param("mps", marks=pytest.mark.mps),
        pytest.param("cuda", marks=pytest.mark.cuda)
    ]
)
def device(request: pytest.FixtureRequest) -> str:
    """Yields each device this machine has, skipping the others."""
    name = request.param
    if name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("mps is not available")
    if name == "cuda" and not torch.cuda.is_available():
        pytest.skip("cuda is not available")
    return name


@pytest.fixture
def tiny_tokenizer():
    """Builds a real word-level tokenizer without network access."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = [
        "<pad>",
        "<unk>",
        "Object:",
        "small",
        "large",
        "red",
        "blue",
        "circle",
        "square",
        "Category:",
        "A",
        "B",
        "two",
        "tokens"
    ]
    backend = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", unk_token="<unk>")


@pytest.fixture
def tiny_model_and_tokenizer(tiny_tokenizer):
    """Builds a real two-layer GPT-2 and a real tokenizer without network access."""
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=len(tiny_tokenizer),
        n_layer=2,
        n_head=2,
        n_embd=16,
        n_positions=64,
        bos_token_id=0,
        eos_token_id=0
    )
    torch.manual_seed(0)
    return GPT2LMHeadModel(config).eval(), tiny_tokenizer


@pytest.fixture(scope="session")
def byte_model_and_tokenizer():
    """Builds a small random GPT-2 whose byte-level tokenizer can encode any text."""
    from tokenizers import (
        Tokenizer,
        decoders,
        models,
        pre_tokenizers
    )
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    symbols = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocabulary = {"<pad>": 0, **{symbol: position + 1 for position, symbol in enumerate(symbols)}}
    backend = Tokenizer(models.BPE(vocab=vocabulary, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", eos_token="<pad>")
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_layer=1,
        n_head=2,
        n_embd=16,
        n_positions=4096,
        bos_token_id=0,
        eos_token_id=0
    )
    torch.manual_seed(0)
    return GPT2LMHeadModel(config).eval(), tokenizer


@pytest.fixture(scope="session")
def vision_model_and_processor():
    """Builds a random Qwen3.5 that reads images and video, with its processor, without network access."""
    from helpers import tiny_vision_model

    return tiny_vision_model()


@pytest.fixture(scope="session")
def omni_model_and_processor():
    """Builds a random Qwen2.5-Omni thinker that reads images, video and audio, without network access."""
    from helpers import tiny_omni_model

    return tiny_omni_model()
