from __future__ import annotations

import pytest
import torch

from cognarium.errors import UntrainedModelWarning
from cognarium.loading import load
from cognarium.models.lstm import LSTMLanguageModel, SimpleTokenizer
from cognarium.paradigms.learning import IteratedLearning
from cognarium.responses import PromptTemplate, Query
from cognarium.stimuli import Observation, Stimulus, Trial
from cognarium.subjects.custom import LSTM, CustomSubject
from cognarium.tasks.shepard import ShepardCategories

WORDS = [
    "Object:",
    "small",
    "large",
    "red",
    "blue",
    "circle",
    "square",
    "Category:",
    "A",
    "B"
]
TEMPLATE = PromptTemplate(stimulus="Object: {text}", response="Category:")


def tokenizer():
    return SimpleTokenizer.from_texts([" ".join(WORDS)])


def untrained(**kwargs):
    tok = tokenizer()
    return CustomSubject(
        LSTMLanguageModel(len(tok)),
        tok,
        pretrained=False,
        device="cpu",
        **kwargs
    )


def query_for(stimulus):
    return Query(trial=Trial(stimulus, options=("A", "B")), template=TEMPLATE)


def weights(subject):
    return torch.cat([p.flatten() for p in subject.model.parameters()])


def test_tokenizer_is_deterministic_and_marks_unknowns():
    tok = SimpleTokenizer.from_texts(["b a", "a c"])
    assert tok.encode("a b c") == [2, 3, 4]
    assert tok.encode("zzz") == [tok.unk_token_id]
    assert len(tok) == 5


def test_lstm_forward_shape():
    logits, _ = LSTMLanguageModel(vocab_size=7)(torch.zeros((2, 3), dtype=torch.long))
    assert logits.shape == (2, 3, 7)


def test_fresh_draws_per_chain_weights_and_restores_them():
    subject = untrained()
    first, again, other = subject.fresh((0, 1)), subject.fresh((0, 1)), subject.fresh((0, 2))
    assert torch.equal(weights(first), weights(again))
    assert not torch.equal(weights(first), weights(other))


def test_fresh_of_pretrained_returns_the_given_weights_after_learning():
    tok = tokenizer()
    subject = CustomSubject(
        LSTMLanguageModel(len(tok)),
        tok,
        device="cpu",
        train_epochs=3
    )
    start = weights(subject).clone()
    learner = subject.fresh((0,))
    learner.learn([Observation(Stimulus(text="small"), "A")], template=TEMPLATE)
    assert not torch.equal(weights(learner), start)
    assert torch.equal(weights(subject), start)
    assert torch.equal(weights(subject.fresh((1,))), start)


def test_learning_fits_the_items_it_saw(device):
    tok = tokenizer()
    subject = CustomSubject(
        LSTMLanguageModel(len(tok)),
        tok,
        pretrained=False,
        device=device
    )
    seen = [
        ("small red circle", "A"),
        ("small red square", "B"),
        ("large blue circle", "A"),
        ("large blue square", "B")
    ]
    learner = subject.fresh((0,))
    learner.learn([Observation(Stimulus(text=t), label) for t, label in seen], template=TEMPLATE)
    probs = learner.respond([query_for(Stimulus(text=t)) for t, _ in seen]).log_probs.exp()
    for row, (_, label) in zip(probs, seen):
        assert row[("A", "B").index(label)] > 0.5


def test_default_learning_follows_pretrained():
    assert untrained().default_learning == "weights"
    tok = SimpleTokenizer.from_texts(["a b"])
    assert CustomSubject(LSTMLanguageModel(len(tok)), tok, device="cpu").default_learning == "in_context"


def test_in_context_on_untrained_warns_and_is_reproducible():
    made = IteratedLearning(task=ShepardCategories())
    with pytest.warns(UntrainedModelWarning):
        one = untrained().bind(made, "in_context")
    with pytest.warns(UntrainedModelWarning):
        two = untrained().bind(made, "in_context")
    assert torch.equal(weights(one), weights(two))


def test_custom_module_and_adapter_are_used():
    class DictGRU(torch.nn.Module):
        def __init__(self, vocab):
            super().__init__()
            self.embed = torch.nn.Embedding(vocab, 4)
            self.gru = torch.nn.GRU(4, 4, batch_first=True)
            self.head = torch.nn.Linear(4, vocab)

        def forward(self, ids):
            return {"out": self.head(self.gru(self.embed(ids))[0])}

    tok = tokenizer()
    adapter = lambda model, ids, mask: model(ids)["out"]
    subject = load(
        DictGRU(len(tok)),
        tokenizer=tok,
        adapter=adapter,
        device="cpu"
    )
    assert isinstance(subject, CustomSubject)
    assert subject.respond([query_for(Stimulus(text="small"))]).scores.shape == (1, 2)


def test_adapter_with_wrong_shape_raises():
    tok = tokenizer()
    adapter = lambda model, ids, mask: ids
    subject = CustomSubject(
        LSTMLanguageModel(len(tok)),
        tok,
        adapter=adapter,
        device="cpu"
    )
    with pytest.raises(ValueError, match="adapter"):
        subject.respond([query_for(Stimulus(text="small"))])


def test_module_that_cannot_be_reset_says_so():
    class Bare(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    subject = CustomSubject(
        Bare(),
        tokenizer(),
        pretrained=False,
        device="cpu"
    )
    with pytest.raises(ValueError, match="reset_parameters"):
        subject.fresh((0,))


def test_unknown_option_on_simple_tokenizer_raises():
    tok = SimpleTokenizer.from_texts(["A small"])
    subject = CustomSubject(LSTMLanguageModel(len(tok)), tok, device="cpu")
    with pytest.raises(ValueError, match="B"):
        subject.respond([query_for(Stimulus(text="small"))])


def test_config_tracks_weights_only_when_pretrained():
    tok = tokenizer()
    one = CustomSubject(LSTMLanguageModel(len(tok)), tok, device="cpu").config()
    two = CustomSubject(LSTMLanguageModel(len(tok)), tok, device="cpu").config()
    assert one != two
    first = untrained().config()
    assert first == untrained().config()


def test_lstm_binds_to_the_paradigms_vocabulary():
    task = ShepardCategories()
    bound = LSTM(device="cpu").bind(IteratedLearning(task=task), "weights")
    assert isinstance(bound, CustomSubject)
    query = Query(trial=Trial(task.stimuli[0], options=task.options), template=task.template)
    assert bound.fresh((0,)).respond([query]).scores.isfinite().all()


def test_unbound_lstm_says_so():
    with pytest.raises(RuntimeError, match="bind"):
        LSTM(device="cpu").respond([])
