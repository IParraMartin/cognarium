from __future__ import annotations

import copy

import pytest
import torch

import cognarium as cg
from cognarium.responses import Query
from cognarium.subjects.hf import HFSubject

TEMPLATE = (
    "{% for message in messages %}<|{{ message['role'] }}|>{{ message['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


@pytest.fixture
def chat_pair(byte_model_and_tokenizer):
    model, tokenizer = byte_model_and_tokenizer
    # the tokenizer is shared across the session, so the template goes on a copy
    tokenizer = copy.deepcopy(tokenizer)
    tokenizer.chat_template = TEMPLATE
    return model, tokenizer


def query(text, options=("wug", "dax")):
    return Query(
        trial=cg.Trial(cg.Stimulus(text=text), options=options),
        template=cg.PromptTemplate(stimulus="{text}", response="")
    )


def test_chat_sends_the_prompt_as_one_user_message_and_scores_the_reply(chat_pair):
    model, tokenizer = chat_pair
    asked = query("Features: [1, 2] -> Label:")
    chat = HFSubject(model, tokenizer, device="cpu", chat=True).respond([asked])
    written = query("<|user|>Features: [1, 2] -> Label:\n<|assistant|>")
    plain = HFSubject(model, tokenizer, device="cpu", option_prefix="").respond([written])
    assert torch.allclose(chat.scores, plain.scores)


def test_chat_needs_a_tokenizer_with_a_chat_template(byte_model_and_tokenizer):
    model, tokenizer = byte_model_and_tokenizer
    with pytest.raises(ValueError, match="chat template"):
        HFSubject(model, tokenizer, device="cpu", chat=True)


def test_chat_marks_the_run_only_when_it_is_on(chat_pair):
    model, tokenizer = chat_pair
    assert HFSubject(model, tokenizer, device="cpu", chat=True).config()["chat"] is True
    assert "chat" not in HFSubject(model, tokenizer, device="cpu").config()


def test_load_passes_chat_on(chat_pair):
    model, tokenizer = chat_pair
    assert cg.load(model, tokenizer=tokenizer, device="cpu", chat=True).chat


def test_chat_also_formats_free_generation(chat_pair):
    model, tokenizer = chat_pair
    subject = cg.load(model, tokenizer=tokenizer, device="cpu", chat=True)
    texts = subject.generate([query("Name a colour.")], max_new_tokens=3, temperature=1.0)
    assert len(texts) == 1


def test_a_paradigm_runs_in_chat_format(chat_pair, tmp_path):
    model, tokenizer = chat_pair
    subject = cg.load(model, tokenizer=tokenizer, device="cpu", chat=True)
    task = cg.tasks.Mapping([[0, 1], [1, 0], [1, 1]], ("wug", "dax"))
    paradigm = cg.paradigms.IteratedLearning(task, generations=2, chains=2, n_prev_examples=2, memory=1)
    results = cg.run(paradigm, subject, out=tmp_path / "run", progress=False)
    assert results.manifest["subject"]["chat"] is True
    assert len(results.to_pandas()) == 2 * 2 * 3
