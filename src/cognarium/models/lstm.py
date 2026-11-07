"""A small LSTM language model and the tokenizer that goes with it."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import torch


class SimpleTokenizer:
    """Splits text on whitespace and maps words to ids.

    Args:
        vocabulary: The known words, in id order. Ids 0 and 1 are reserved for
            padding and unknown words.
    """

    pad_token_id = 0
    unk_token_id = 1

    def __init__(self, vocabulary: Sequence[str]) -> None:
        self._ids = {word: index + 2 for index, word in enumerate(vocabulary)}

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> SimpleTokenizer:
        """Builds a tokenizer from the words in some texts.

        Args:
            texts: The texts to read. The vocabulary is sorted, so ids don't
                depend on the order of the texts.

        Returns:
            The tokenizer.
        """
        return cls(sorted({word for text in texts for word in text.split()}))

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Turns text into token ids.

        Args:
            text: The text to encode.
            add_special_tokens: Accepted to match Hugging Face tokenizers. This
                tokenizer has no special tokens to add.

        Returns:
            One id per word.
        """
        return [self._ids.get(word, self.unk_token_id) for word in text.split()]

    def __len__(self) -> int:
        return len(self._ids) + 2


class LSTMLanguageModel(torch.nn.Module):
    """Predicts the next token with an LSTM.

    Args:
        vocab_size: How many tokens the model knows.
        embedding_dim: The size of the token embeddings.
        hidden_dim: The size of the LSTM's hidden state.
        num_layers: How many LSTM layers to stack.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 1
    ) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True
        )
        self.readout = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor, state: Any = None) -> tuple[torch.Tensor, Any]:
        """Computes next-token logits.

        Args:
            input_ids: Token ids, shape (batch, time).
            state: The LSTM state to continue from, if any.

        Returns:
            Logits of shape (batch, time, vocab), and the final LSTM state.
        """
        hidden, state = self.lstm(self.embedding(input_ids), state)
        return self.readout(hidden), state
