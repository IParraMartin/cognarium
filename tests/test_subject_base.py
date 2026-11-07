from __future__ import annotations

import pytest
import torch

from cognarium.errors import MissingAbilityError
from cognarium.responses import PromptTemplate
from helpers import ScriptedSubject
from cognarium.subjects.base import weights_digest


def test_optional_abilities_say_what_is_missing():
    subject = ScriptedSubject()
    with pytest.raises(MissingAbilityError, match="fresh"):
        subject.fresh((0,))
    with pytest.raises(MissingAbilityError, match="learn"):
        subject.learn([], template=PromptTemplate())


def test_bind_returns_the_subject_by_default():
    subject = ScriptedSubject()
    assert subject.bind(None, None) is subject


def test_weights_digest_follows_the_weights():
    module = torch.nn.Linear(2, 2)
    before = weights_digest(module)
    assert weights_digest(module) == before
    with torch.no_grad():
        module.weight.add_(1.0)
    assert weights_digest(module) != before
