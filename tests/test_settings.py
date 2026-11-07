from __future__ import annotations

import json

import pytest

from cognarium.settings import Settings


@pytest.mark.parametrize(
    "bad",
    [
        {"temperature": 0.0},
        {"top_p": 0.0},
        {"top_p": 1.5},
        {"top_k": 0}
    ]
)
def test_invalid_shaping_raises(bad):
    with pytest.raises(ValueError):
        Settings(**bad)


def test_examples_are_normalised_and_json_safe():
    settings = Settings(examples=[("red circle", "A")])
    assert settings.examples[0].label == "A"
    json.dumps(settings.config())
