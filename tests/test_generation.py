from __future__ import annotations

import pandas as pd
import pytest
import torch
from helpers import ScriptedSubject

import cognarium as cg
from cognarium.models.lstm import SimpleTokenizer
from cognarium.responses import Query
from cognarium.stimuli import Stimulus, Trial
from cognarium.subjects.base import Subject
from cognarium.subjects.generation import generate_texts
from cognarium.subjects.hf import HFSubject


class Recall(cg.Paradigm):
    response_mode = "generate"

    def trials(self, rng):
        yield cg.Trial(cg.Stimulus(text="small red"), answer="red")
        yield cg.Trial(cg.Stimulus(text="large blue circle"))


class Remembering(Recall):
    context = cg.context.cumulative
    ordered = True


class TextSubject(Subject):
    """Writes numbered text, and counts how often it is called."""

    modalities = frozenset({"text"})
    can_generate = True

    def __init__(self, fail_after=None):
        self.fail_after = fail_after
        self.calls = 0
        self.seen = []

    def respond(self, queries, *, capture_logits=False):
        raise AssertionError("A free response must not ask for scores.")

    def generate(
        self,
        queries,
        *,
        max_new_tokens,
        temperature=1.0,
        top_p=1.0,
        top_k=None
    ):
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError("scripted failure")
        self.calls += 1
        self.seen.extend(queries)
        return [f"text {self.calls}" for _ in queries]


def query(text="small", seed=(0,)):
    return Query(Trial(Stimulus(text=text)), response_seed=seed)


def scripted_forward(*favoured):
    """Gives logits that favour one token per position."""

    def forward(ids, mask):
        logits = torch.zeros(ids.shape[0], ids.shape[1], 8)
        for row, length in enumerate(mask.sum(dim=1).tolist()):
            logits[row, length - 1, favoured[min(length - 1, len(favoured) - 1)]] = 50.0
        return logits

    return forward


def generate(forward, queries=None, **options):
    return generate_texts(
        queries or [query()],
        forward=forward,
        encode_prompt=lambda text: [1],
        decode=lambda ids: " ".join(str(token) for token in ids),
        **{
            "eos_token_id": 0,
            "pad_token_id": 0,
            "device": torch.device("cpu"),
            "max_new_tokens": 3,
            **options
        }
    )


def run_quietly(
    paradigm,
    subject,
    folder,
    **options
):
    return cg.run(
        paradigm,
        subject,
        out=folder,
        progress=False,
        **options
    )


def test_generation_runs_and_stores_the_text(tiny_model_and_tokenizer, tmp_path):
    model, tokenizer = tiny_model_and_tokenizer
    subject = HFSubject(model, tokenizer, device="cpu")
    frame = run_quietly(Recall(max_new_tokens=4), subject, tmp_path / "run").to_pandas()
    assert frame["response"].map(lambda text: isinstance(text, str)).all()
    assert frame["response_index"].tolist() == [-1, -1]
    assert [len(options) for options in frame["options"]] == [0, 0]
    assert [len(values) for values in frame["log_probs"]] == [0, 0]


def test_generated_text_does_not_depend_on_batch_size(tiny_model_and_tokenizer):
    model, tokenizer = tiny_model_and_tokenizer
    queries = [query(text, seed=(index,)) for index, text in enumerate(["small", "red circle", "large blue", "A B"])]
    texts = {
        size: HFSubject(
            model,
            tokenizer,
            device="cpu",
            batch_size=size
        ).generate(queries, max_new_tokens=5)
        for size in (1, 2, 4)
    }
    assert texts[1] == texts[2] == texts[4]


def test_generation_is_seeded(tiny_model_and_tokenizer, tmp_path):
    model, tokenizer = tiny_model_and_tokenizer
    subject = HFSubject(model, tokenizer, device="cpu")
    text = lambda seed, name: run_quietly(
        Recall(max_new_tokens=8, temperature=5.0, seed=seed),
        subject,
        tmp_path / name
    ).to_pandas()["response"].tolist()
    assert text(1, "a") == text(1, "b")
    assert text(1, "a") != text(2, "c")


def test_generation_stops_at_end_of_text():
    assert generate(scripted_forward(0)) == [""]
    assert generate(scripted_forward(3, 0)) == ["3"]


def test_generation_stops_at_padding_when_there_is_no_end_token():
    assert generate(scripted_forward(3, 5), eos_token_id=None, pad_token_id=5) == ["3"]


def test_top_k_one_is_greedy():
    assert generate(scripted_forward(3, 4, 5), top_k=1) == ["3 4 5"]


def test_generation_stops_at_the_budget():
    assert generate(scripted_forward(3), max_new_tokens=2) == ["3 3"]


def test_temperature_changes_the_sampling():
    def forward(ids, mask):
        logits = torch.zeros(ids.shape[0], ids.shape[1], 8)
        logits[:, :, 3] = 2.0
        return logits

    queries = [query(seed=(index,)) for index in range(20)]
    sharp = generate(forward, queries, temperature=0.01)
    flat = generate(forward, queries, temperature=100.0)
    assert set(sharp) == {"3 3 3"}
    assert len(set(flat)) > 1


def test_generation_refuses_a_prompt_with_no_room():
    with pytest.raises(ValueError, match="window"):
        generate(scripted_forward(3), max_new_tokens=3, max_length=3)
    assert generate(scripted_forward(3), max_new_tokens=2, max_length=3) == ["3 3"]


def test_generation_refuses_an_empty_prompt():
    with pytest.raises(ValueError, match="no tokens"):
        generate_texts(
            [query()],
            forward=scripted_forward(3),
            encode_prompt=lambda text: [],
            decode=str,
            eos_token_id=0,
            pad_token_id=0,
            device=torch.device("cpu")
        )


def test_a_subject_that_cannot_generate_fails_before_any_call(tmp_path):
    subject = ScriptedSubject()
    with pytest.raises(cg.MissingAbilityError, match="free"):
        run_quietly(Recall(), subject, tmp_path / "run")
    assert subject.calls == 0
    with pytest.raises(cg.MissingAbilityError, match="free"):
        subject.generate([query()], max_new_tokens=1)


def test_recording_with_generation_is_refused(tmp_path):
    with pytest.raises(cg.MissingAbilityError, match="record"):
        run_quietly(
            Recall(),
            TextSubject(),
            tmp_path / "run",
            record=["logits"]
        )


def test_a_free_trial_in_a_choice_paradigm_fails():
    class Mixed(cg.Paradigm):
        def trials(self, rng):
            yield cg.Trial(cg.Stimulus(text="a"))

    with pytest.raises(ValueError, match="response_mode"):
        Mixed().sequences()


def test_options_in_a_generate_paradigm_fail():
    class Mixed(Recall):
        def trials(self, rng):
            yield cg.Trial(cg.Stimulus(text="a"), options=["a", "b"])

    with pytest.raises(ValueError, match="response_mode"):
        Mixed().sequences()


def test_generation_receives_the_settings(tmp_path):
    received = {}

    class Keeping(TextSubject):
        def generate(self, queries, **options):
            received.update(options)
            return super().generate(queries, **options)

    paradigm = Recall(
        max_new_tokens=7,
        temperature=0.5,
        top_p=0.9,
        top_k=3
    )
    run_quietly(paradigm, Keeping(), tmp_path / "run")
    assert received == {
        "max_new_tokens": 7,
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 3
    }


def test_probability_metrics_refuse_free_responses(tmp_path):
    results = run_quietly(Recall(), TextSubject(), tmp_path / "run")
    for name in ("entropy", "surprisal", "brier", "f1", "cross_entropy"):
        with pytest.raises(cg.MissingAbilityError, match="free responses"):
            results.metrics(name)
    for name in ("surprisal", "entropy", "answer_probability", "expected_option"):
        with pytest.raises(cg.MissingAbilityError, match="free responses"):
            results.contrast(name, against=0.5, where={"item": 0})


def test_free_answers_can_still_be_compared_as_text(tmp_path):
    class Exact(TextSubject):
        def generate(self, queries, **options):
            return ["red" for _ in queries]

    results = run_quietly(Recall(), Exact(), tmp_path / "run")
    assert results.metrics("accuracy")["value"].item() == 1.0


def test_parsing_runs_on_the_text(tmp_path):
    class Parsed(Recall):
        def annotate(self, frame):
            return frame.assign(words=frame["response"].str.split().str.len())

    frame = run_quietly(Parsed(), TextSubject(), tmp_path / "run").to_pandas()
    assert frame["words"].tolist() == [2, 2]


def test_empty_text_gives_a_row_not_an_error(tmp_path):
    class Silent(TextSubject):
        def generate(self, queries, **options):
            return ["" for _ in queries]

    frame = run_quietly(Recall(), Silent(), tmp_path / "run").to_pandas()
    assert frame["response"].tolist() == ["", ""]


def test_lstm_generates(tmp_path):
    subject = cg.subjects.LSTM(
        device="cpu",
        train_epochs=1,
        embedding_dim=4,
        hidden_dim=4
    )
    frame = run_quietly(Recall(max_new_tokens=3), subject, tmp_path / "run").to_pandas()
    assert frame["response"].map(lambda text: isinstance(text, str)).all()


def test_lstm_trains_on_context_then_generates(tmp_path):
    subject = cg.subjects.LSTM(
        device="cpu",
        train_epochs=1,
        embedding_dim=4,
        hidden_dim=4
    )
    paradigm = Remembering(max_new_tokens=2)
    frame = run_quietly(paradigm, subject, tmp_path / "run").to_pandas()
    assert frame["history"].map(len).tolist() == [0, 1]


def test_weights_mode_with_no_context_skips_training(tmp_path):
    class Choosing(cg.Paradigm):
        def trials(self, rng):
            yield cg.Trial(cg.Stimulus(text="yes or no"), options=["yes", "no"])

    subject = cg.subjects.LSTM(
        device="cpu",
        train_epochs=1,
        embedding_dim=4,
        hidden_dim=4
    )
    frame = run_quietly(Choosing(), subject, tmp_path / "run").to_pandas()
    assert frame["response"].isin(["yes", "no"]).all()


def test_simple_tokenizer_decodes_one_id_or_many():
    tokenizer = SimpleTokenizer(["a", "b"])
    assert tokenizer.decode(2) == "a"
    assert tokenizer.decode([2, 3]) == "a b"
    assert tokenizer.decode([]) == ""


def test_a_tokenizer_with_no_decode_says_so():
    class Bare:
        pad_token_id = 0

        def encode(self, text, add_special_tokens=True):
            return [1]

        def __len__(self):
            return 8

    subject = cg.subjects.CustomSubject(
        cg.models.LSTMLanguageModel(8),
        Bare(),
        device="cpu"
    )
    with pytest.raises(cg.MissingAbilityError, match="decode"):
        subject.generate([query()], max_new_tokens=1)


def test_generation_resumes(tmp_path):
    whole = run_quietly(Remembering(), TextSubject(), tmp_path / "whole").to_pandas()
    with pytest.raises(RuntimeError, match="scripted failure"):
        run_quietly(Remembering(), TextSubject(fail_after=1), tmp_path / "killed")
    resumed = TextSubject()
    frame = run_quietly(Remembering(), resumed, tmp_path / "killed").to_pandas()
    assert resumed.calls == 1
    pd.testing.assert_frame_equal(frame[["trial"]], whole[["trial"]])
    assert list(frame["history"].iloc[1]) == ["text 1"]
    assert [item.label for item in resumed.seen[0].context] == ["text 1"]


def test_generation_runs_on_each_device(device, tiny_model_and_tokenizer, tmp_path):
    model, tokenizer = tiny_model_and_tokenizer
    subject = HFSubject(model, tokenizer, device=device)
    frame = run_quietly(Recall(max_new_tokens=3), subject, tmp_path / "run").to_pandas()
    assert len(frame) == 2
