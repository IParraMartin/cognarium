from __future__ import annotations

import hashlib
import json
import math

import pandas as pd
import pytest
import torch
from helpers import RecordingCapture, ScriptedSubject

from cognarium import runner
from cognarium.errors import MissingAbilityError, RunConflictError
from cognarium.metrics.geometry import rsa
from cognarium.metrics.information import entropy
from cognarium.paradigms.learning import IteratedLearning
from cognarium.recording.logits import logit_columns
from cognarium.results import Results
from cognarium.runner import run
from cognarium.subjects.bayesian import BayesianLearner
from cognarium.subjects.custom import LSTM
from cognarium.subjects.exemplar import ExemplarLearner
from cognarium.subjects.hf import HFSubject
from cognarium.tasks.shepard import ShepardCategories


def paradigm():
    return IteratedLearning(ShepardCategories(type="I"), generations=3, chains=2)


def recorded(
    tmp_path,
    subject,
    *,
    name="run",
    **kwargs
):
    return run(
        paradigm(),
        subject,
        record=["hidden_states"],
        out=tmp_path / name,
        progress=False,
        **kwargs
    )


def test_states_line_up_with_the_table(tiny_model_and_tokenizer, tmp_path):
    model, tok = tiny_model_and_tokenizer
    subject = HFSubject(model, tok, device="cpu")
    results = recorded(tmp_path, subject)
    states = results.hidden.to_tensor("layer_01")
    assert states.shape == (48, 16)
    assert len(results.to_pandas()) == 48
    table = results.to_pandas()
    previous = pd.read_parquet(tmp_path / "run" / "steps" / "step-00000.parquet")
    query = paradigm().step(1, previous).queries[5]
    capture = RecordingCapture()
    subject.respond([query], capture_states=capture)
    row = table.index[table["step"] == 1][5]
    assert torch.allclose(states[row].float(), capture.stacked("layer_01")[0], atol=2e-3)


def test_batch_size_does_not_change_the_states(tiny_model_and_tokenizer, tmp_path):
    model, tok = tiny_model_and_tokenizer
    one = recorded(tmp_path, HFSubject(
        model,
        tok,
        device="cpu",
        batch_size=1
    ), name="one")
    many = recorded(tmp_path, HFSubject(
        model,
        tok,
        device="cpu",
        batch_size=8
    ), name="many")
    assert one.manifest["key"] == many.manifest["key"]
    for layer in ["layer_00", "layer_01"]:
        assert torch.allclose(one.hidden.to_tensor(layer).float(), many.hidden.to_tensor(layer).float(), atol=2e-3)


def test_subject_without_states_fails_before_any_call(tmp_path):
    subject = ScriptedSubject()
    with pytest.raises(MissingAbilityError, match="ScriptedSubject"):
        run(
            paradigm(),
            subject,
            record=["hidden_states"],
            out=tmp_path / "run",
            progress=False
        )
    assert subject.calls == 0
    assert not (tmp_path / "run").exists()


def test_layers_choose_what_is_recorded(tiny_model_and_tokenizer, tmp_path):
    model, tok = tiny_model_and_tokenizer
    results = recorded(tmp_path, HFSubject(model, tok, device="cpu"), layers=["layer_01"])
    assert results.hidden.layers == ("layer_01",)
    assert results.manifest["states"]["layers"] == ["layer_01"]
    assert not (tmp_path / "run" / "states" / "layer_00").exists()


def test_every_layer_is_recorded_by_default(tiny_model_and_tokenizer, tmp_path):
    model, tok = tiny_model_and_tokenizer
    results = recorded(tmp_path, HFSubject(model, tok, device="cpu"))
    assert results.hidden.layers == ("layer_00", "layer_01")
    assert results.hidden.dtype == "float16"
    assert results.hidden.pool == "last"


def test_unknown_layer_and_pool_raise(tiny_model_and_tokenizer, tmp_path):
    model, tok = tiny_model_and_tokenizer
    subject = HFSubject(model, tok, device="cpu")
    with pytest.raises(ValueError, match="layer_00"):
        recorded(tmp_path, subject, layers=["nope"])
    with pytest.raises(ValueError, match="last"):
        recorded(tmp_path, subject, pool="nope")
    with pytest.raises(ValueError, match="bfloat16"):
        recorded(tmp_path, subject, state_dtype="int8")
    with pytest.raises(ValueError, match="layers"):
        recorded(tmp_path, subject, layers=[])
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "settings",
    [
        {"layers": ["layer_00"]},
        {"pool": "mean"},
        {"state_dtype": "float32"}
    ]
)
def test_recording_settings_without_recording_raise(tiny_model_and_tokenizer, tmp_path, settings):
    model, tok = tiny_model_and_tokenizer
    with pytest.raises(ValueError, match="hidden_states"):
        run(
            paradigm(),
            HFSubject(model, tok, device="cpu"),
            out=tmp_path / "run",
            progress=False,
            **settings
        )


def test_hidden_without_recording_explains(tmp_path):
    with pytest.raises(MissingAbilityError, match="hidden_states"):
        run(
            paradigm(),
            ScriptedSubject(),
            out=tmp_path / "run",
            progress=False
        ).hidden


def test_a_run_without_states_keeps_its_key(tmp_path):
    results = run(
        paradigm(),
        ScriptedSubject(),
        out=tmp_path / "run",
        progress=False
    )
    assert "states" not in results.manifest
    identity = {section: results.manifest[section] for section in ["paradigm", "subject", "learning", "record"]}
    text = json.dumps(identity, sort_keys=True)
    assert results.manifest["key"] == hashlib.sha256(text.encode()).hexdigest()[:12]


@pytest.mark.parametrize(
    "settings",
    [
        {"layers": ["layer_00"]},
        {"pool": "mean"},
        {"state_dtype": "float32"}
    ]
)
def test_changed_recording_settings_refuse_and_name_the_section(tiny_model_and_tokenizer, tmp_path, settings):
    model, tok = tiny_model_and_tokenizer
    subject = HFSubject(model, tok, device="cpu")
    recorded(tmp_path, subject)
    with pytest.raises(RunConflictError, match="states"):
        recorded(tmp_path, subject, **settings)


def test_layer_order_does_not_change_the_identity(tiny_model_and_tokenizer, tmp_path):
    model, tok = tiny_model_and_tokenizer
    subject = HFSubject(model, tok, device="cpu")
    first = recorded(tmp_path, subject, layers=["layer_00", "layer_01"])
    second = recorded(tmp_path, subject, layers=["layer_01", "layer_00"])
    assert first.manifest["key"] == second.manifest["key"]


def test_a_finished_run_can_be_opened_again_without_new_calls(tiny_model_and_tokenizer, tmp_path):
    model, tok = tiny_model_and_tokenizer
    subject = HFSubject(model, tok, device="cpu")
    first = recorded(tmp_path, subject)
    states = first.hidden.to_tensor("layer_00")
    again = recorded(tmp_path, subject)
    assert torch.equal(again.hidden.to_tensor("layer_00"), states)


def test_killed_run_resumes_to_identical_states(tiny_model_and_tokenizer, tmp_path, monkeypatch):
    model, tok = tiny_model_and_tokenizer
    subject = HFSubject(model, tok, device="cpu")
    whole = recorded(tmp_path, subject, name="whole")
    real = runner._write_table

    def stop_at_step_one(path, frame):
        if "step-00001" in path.name:
            raise KeyError("killed")
        real(path, frame)

    monkeypatch.setattr(runner, "_write_table", stop_at_step_one)
    with pytest.raises(KeyError, match="killed"):
        recorded(tmp_path, subject, name="cut")
    monkeypatch.undo()
    assert (tmp_path / "cut" / "states" / "layer_00" / "step-00001-00000.safetensors").exists()
    cut = Results.load(tmp_path / "cut")
    assert set(cut.to_pandas()["step"]) == {0}
    assert cut.hidden.to_tensor("layer_00").shape == (16, 16)
    resumed = recorded(tmp_path, subject, name="cut")
    for layer in ["layer_00", "layer_01"]:
        assert torch.equal(resumed.hidden.to_tensor(layer), whole.hidden.to_tensor(layer))


def test_a_failing_subject_marks_nothing_done(tiny_model_and_tokenizer, tmp_path, monkeypatch):
    model, tok = tiny_model_and_tokenizer
    subject = HFSubject(model, tok, device="cpu")

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("cognarium.recording.store.save_file", fail)
    with pytest.raises(OSError, match="disk full"):
        recorded(tmp_path, subject)
    assert not list((tmp_path / "run").glob("steps/*.parquet"))


def test_no_pool_records_every_token(tiny_model_and_tokenizer, tmp_path):
    model, tok = tiny_model_and_tokenizer
    results = recorded(tmp_path, HFSubject(model, tok, device="cpu"), pool=None)
    sequences = results.hidden.sequences("layer_00")
    assert len(sequences) == 48
    assert all(row.ndim == 2 and row.shape[1] == 16 for row in sequences)
    prompt = paradigm().step(0, None).queries[0].prompt()
    assert sequences[0].shape[0] == len(tok(prompt)["input_ids"])


def test_mean_pool_records_one_row_per_query(tiny_model_and_tokenizer, tmp_path):
    model, tok = tiny_model_and_tokenizer
    results = recorded(tmp_path, HFSubject(model, tok, device="cpu"), pool="mean")
    assert results.hidden.to_tensor("layer_00").shape == (48, 16)


def test_lstm_records_in_query_order_in_weights_mode(tmp_path):
    results = recorded(tmp_path, LSTM(hidden_dim=8, train_epochs=2, device="cpu"))
    assert results.manifest["learning"] == "weights"
    states = results.hidden.to_tensor("lstm")
    assert states.shape == (48, 8)
    table = results.hidden.trials
    assert table[table["step"] == 0]["state_offset"].tolist() == list(range(16))
    # chains learn separately, so their rows of one step should differ
    first_step = states[:16]
    assert not torch.equal(first_step[:8], first_step[8:])


def test_exemplar_and_bayesian_record_their_states(tmp_path):
    similarity = recorded(tmp_path, ExemplarLearner(guess=0.05), name="exemplar").hidden
    posterior = recorded(tmp_path, BayesianLearner(), name="bayesian").hidden
    assert similarity.dtype == posterior.dtype == "float32"
    assert similarity.to_tensor("similarity").shape == (48, 4)
    assert posterior.to_tensor("posterior").shape == (48, 256)
    assert similarity.to_tensor("similarity").dtype == torch.float32


def test_rsa_compares_an_lstm_layer_with_an_exemplar_model(tmp_path):
    lstm = recorded(tmp_path, LSTM(hidden_dim=8, train_epochs=2, device="cpu"), name="lstm").hidden
    exemplar = recorded(tmp_path, ExemplarLearner(guess=0.05), name="exemplar").hidden
    score = rsa(lstm.to_tensor("lstm"), exemplar.to_tensor("similarity"), metric="euclidean")
    assert -1 <= score <= 1


def test_a_store_saved_from_a_run_reproduces_the_cka(tiny_model_and_tokenizer, tmp_path):
    model, tok = tiny_model_and_tokenizer
    hidden = recorded(tmp_path, HFSubject(model, tok, device="cpu")).hidden
    saved = hidden.save(tmp_path / "shared")
    assert saved.compare().equals(hidden.compare())


def test_a_subject_that_skips_a_requested_layer_fails(tiny_model_and_tokenizer, tmp_path):
    model, tok = tiny_model_and_tokenizer

    class Skipping(HFSubject):
        def respond(
            self,
            queries,
            *,
            capture_logits=False,
            capture_states=None
        ):
            partial = type(capture_states)(("layer_00",), capture_states.pool, capture_states.write)
            return super().respond(queries, capture_logits=capture_logits, capture_states=partial)

    with pytest.raises(RuntimeError, match="layer_01"):
        recorded(tmp_path, Skipping(model, tok, device="cpu"))
    assert not list((tmp_path / "run").glob("steps/*.parquet"))


def test_logit_columns_hand_values():
    columns = logit_columns(torch.tensor([[3.0, 1.0, 0.0]]), [[1, 0]], [0])
    expected_entropy = entropy(torch.tensor([3.0, 1.0, 0.0], dtype=torch.float64).log_softmax(dim=-1)).item()
    assert columns["option_first_tokens"] == [[1, 0]]
    assert columns["target_rank"] == [2]
    assert columns["top_token_id"] == [0]
    assert columns["logit_margin"] == [pytest.approx(2.0)]
    assert columns["logit_difference"] == [pytest.approx(-2.0)]
    assert columns["vocab_entropy"] == [pytest.approx(expected_entropy)]


def test_logit_columns_without_an_answer_or_a_second_option():
    columns = logit_columns(torch.tensor([[3.0, 1.0, 0.0], [0.0, 1.0, 2.0]]), [[1, 0], [2]], [None, 0])
    assert columns["target_rank"] == [None, 1]
    assert columns["logit_difference"] == [None, None]
    assert columns["logit_margin"][0] == pytest.approx(2.0)
    assert columns["logit_margin"][1] is None


def scripted_logits():
    return ScriptedSubject(
        probabilities=(0.5, 0.5),
        logits=torch.tensor([3.0, 1.0, 0.0]),
        first_token_ids=[1, 0]
    )


def test_runs_write_the_logit_columns(tmp_path):
    results = run(
        paradigm(),
        scripted_logits(),
        record=["logits"],
        out=tmp_path / "run",
        progress=False
    )
    table = results.to_pandas()
    assert table["top_token_id"].tolist() == [0] * 48
    assert table["logit_margin"].tolist() == pytest.approx([2.0] * 48)
    assert list(table["option_first_tokens"].iloc[0]) == [1, 0]
    assert math.isfinite(table["vocab_entropy"].iloc[0])


def test_rows_without_an_answer_have_no_target_rank(tmp_path):
    class Unanswered(IteratedLearning):
        def step(self, index, previous):
            step = super().step(index, previous)
            if step is None:
                return None
            for query in step.queries:
                object.__setattr__(query.trial, "answer", None)
            return step

    results = run(
        Unanswered(ShepardCategories(type="I"), generations=1, chains=1),
        scripted_logits(),
        record=["logits"],
        out=tmp_path / "run",
        progress=False
    )
    table = results.to_pandas()
    assert table["target_rank"].isna().all()
    assert table["logit_difference"].isna().all()
    assert table["logit_margin"].notna().all()
    with pytest.raises(MissingAbilityError, match="target_rank"):
        results.metrics("target_rank")


def test_a_subject_without_tokens_gets_no_logit_columns(tmp_path):
    subject = ScriptedSubject(logits=torch.tensor([3.0, 1.0, 0.0]))
    results = run(
        paradigm(),
        subject,
        record=["logits"],
        out=tmp_path / "run",
        progress=False
    )
    assert "logit_margin" not in results.to_pandas()


def test_results_metrics_read_the_logit_columns(tiny_model_and_tokenizer, tmp_path):
    model, tok = tiny_model_and_tokenizer
    results = run(
        paradigm(),
        HFSubject(model, tok, device="cpu"),
        record=["logits"],
        out=tmp_path / "run",
        progress=False
    )
    table = results.metrics("logit_margin", "target_rank", "vocab_entropy")
    assert len(table) == 9
    assert table["value"].map(math.isfinite).all()


def test_results_can_carry_their_subject(tiny_model_and_tokenizer, tmp_path):
    model, tok = tiny_model_and_tokenizer
    subject = HFSubject(model, tok, device="cpu")
    results = recorded(tmp_path, subject)
    assert results.subject is not None
    assert Results.load(tmp_path / "run").subject is None
    assert Results.load(tmp_path / "run", subject=subject).subject is subject
