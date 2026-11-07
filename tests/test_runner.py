from __future__ import annotations

import json

import numpy as np
import pytest
from helpers import ScriptedSubject, TrainableScriptedSubject

from cognarium.errors import MissingAbilityError, RunConflictError
from cognarium.paradigms.learning import IteratedLearning
from cognarium.results import Results
from cognarium.runner import run
from cognarium.subjects.bayesian import BayesianLearner
from cognarium.subjects.exemplar import ExemplarLearner
from cognarium.subjects.hf import HFSubject
from cognarium.tasks.shepard import ShepardCategories


def paradigm(**kwargs):
    defaults = {
        "task": ShepardCategories(type="I"),
        "generations": 3,
        "chains": 2
    }
    return IteratedLearning(**{**defaults, **kwargs})


def scripted(**kwargs):
    defaults = {"modes": {"in_context"}, "modalities": {"text"}}
    return ScriptedSubject(**{**defaults, **kwargs})


def go(made, subject, tmp_path, **kwargs):
    return run(made, subject, out=tmp_path / "run", progress=False, **kwargs)


def test_writes_one_row_per_query(tmp_path):
    frame = go(paradigm(), scripted(), tmp_path).to_pandas()
    assert len(frame) == 3 * 2 * 8
    expected = [
        "chain",
        "generation",
        "item",
        "response",
        "scores",
        "log_probs",
        "shaped_log_probs",
        "seen",
        "group"
    ]
    for column in expected:
        assert column in frame


def test_each_generation_learns_from_the_last(tmp_path):
    frame = go(paradigm(), scripted(probabilities=(0.0, 1.0)), tmp_path).to_pandas()
    assert set(frame[frame["generation"] > 0]["parent_label"]) == {"B"}


def test_auto_learning_uses_the_subject_default(tmp_path):
    assert go(paradigm(), scripted(), tmp_path).manifest["learning"] == "in_context"


def test_missing_learning_mode_fails_before_any_call(tmp_path):
    subject = scripted(modes=set())
    with pytest.raises(MissingAbilityError, match="in_context|learning"):
        go(paradigm(), subject, tmp_path, learning="in_context")
    assert subject.calls == 0
    assert not (tmp_path / "run").exists()


def test_auto_without_a_mode_fails_clearly(tmp_path):
    subject = scripted(modes=set())
    with pytest.raises(MissingAbilityError, match="learning"):
        go(paradigm(), subject, tmp_path)
    assert subject.calls == 0


def test_unsupported_mode_names_the_supported_ones(tmp_path):
    with pytest.raises(MissingAbilityError, match="in_context"):
        go(paradigm(), scripted(), tmp_path, learning="weights")


def test_no_shared_modality_fails(tmp_path):
    subject = scripted(modalities={"audio"})
    with pytest.raises(MissingAbilityError, match="audio"):
        go(paradigm(), subject, tmp_path)
    assert subject.calls == 0


def test_instructions_on_a_non_text_subject_fail(tmp_path):
    subject = scripted(modalities={"features"})
    with pytest.raises(MissingAbilityError, match="instructions"):
        go(paradigm(instructions="Sort them."), subject, tmp_path)


def test_unknown_record_value_fails(tmp_path):
    with pytest.raises(ValueError, match="logits"):
        go(paradigm(), scripted(), tmp_path, record=["hidden_states"])


def test_batching_does_not_change_the_table(tiny_model_and_tokenizer, tmp_path):
    model, tok = tiny_model_and_tokenizer
    made = paradigm()
    one = run(made, HFSubject(model, tok, device="cpu", batch_size=1), out=tmp_path / "a", progress=False)
    two = run(made, HFSubject(model, tok, device="cpu", batch_size=16), out=tmp_path / "b", progress=False)
    assert one.to_pandas()["response"].tolist() == two.to_pandas()["response"].tolist()


def test_temperature_changes_shaped_but_not_raw(tmp_path):
    frame = go(paradigm(temperature=0.5), scripted(probabilities=(0.8, 0.2)), tmp_path).to_pandas()
    assert np.exp(frame["log_probs"].iloc[0]).tolist() == pytest.approx([0.8, 0.2])
    assert np.exp(frame["shaped_log_probs"].iloc[0]).tolist() == pytest.approx([0.941176, 0.058824], abs=1e-5)


def test_subject_choices_override_sampling(tmp_path):
    frame = go(paradigm(), scripted(choices=1), tmp_path).to_pandas()
    assert set(frame["response"]) == {"B"}


def test_interrupted_run_resumes_to_the_same_table(tmp_path):
    made = paradigm()
    full = run(made, scripted(), out=tmp_path / "full", progress=False).to_pandas()
    with pytest.raises(RuntimeError):
        run(made, scripted(fail_after=2), out=tmp_path / "cut", progress=False)
    subject = scripted()
    resumed = run(made, subject, out=tmp_path / "cut", progress=False).to_pandas()
    assert subject.calls == 1
    assert resumed["response"].tolist() == full["response"].tolist()


def test_partial_shard_is_ignored_on_resume(tmp_path):
    made = paradigm()
    with pytest.raises(RuntimeError):
        go(made, scripted(fail_after=1), tmp_path)
    (tmp_path / "run" / "steps" / "step-00001.parquet.tmp").write_bytes(b"junk")
    assert len(go(made, scripted(), tmp_path).to_pandas()) == 48


def test_finished_run_makes_no_calls(tmp_path):
    made = paradigm()
    go(made, scripted(), tmp_path)
    subject = scripted()
    assert len(go(made, subject, tmp_path).to_pandas()) == 48
    assert subject.calls == 0


def test_each_session_is_recorded(tmp_path):
    made = paradigm()
    with pytest.raises(RuntimeError):
        go(made, scripted(fail_after=1), tmp_path)
    interrupted = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert interrupted["status"] == "running"
    assert len(interrupted["sessions"]) == 1
    assert len(go(made, scripted(), tmp_path).manifest["sessions"]) == 2


def test_changed_settings_refuse_and_name_the_change(tmp_path):
    go(paradigm(), scripted(), tmp_path)
    with pytest.raises(RunConflictError, match="temperature"):
        go(paradigm(temperature=0.5), scripted(), tmp_path)


def test_conflict_names_only_what_changed(tmp_path):
    class TupleConfig(ScriptedSubject):
        def config(self):
            return {"class": "TupleConfig", "shape": (1, 2)}

    go(paradigm(), TupleConfig(), tmp_path)
    with pytest.raises(RunConflictError) as caught:
        go(paradigm(temperature=0.5), TupleConfig(), tmp_path)
    assert "temperature" in str(caught.value)
    assert "subject" not in str(caught.value)


def test_changed_subject_refuses(tmp_path):
    made = paradigm(instructions=None)
    run(made, ExemplarLearner(c=1.0), out=tmp_path / "run", progress=False)
    with pytest.raises(RunConflictError, match="subject"):
        run(made, ExemplarLearner(c=2.0), out=tmp_path / "run", progress=False)


def test_a_conflict_leaves_the_earlier_run_untouched(tmp_path):
    go(paradigm(), scripted(), tmp_path)
    before = (tmp_path / "run" / "manifest.json").read_text()
    with pytest.raises(RunConflictError):
        go(paradigm(temperature=0.5), scripted(), tmp_path)
    assert (tmp_path / "run" / "manifest.json").read_text() == before
    assert len(Results.load(tmp_path / "run").to_pandas()) == 48


def test_resume_false_refuses_an_existing_folder(tmp_path):
    go(paradigm(), scripted(), tmp_path)
    with pytest.raises(RunConflictError):
        go(paradigm(), scripted(), tmp_path, resume=False)


def test_default_folder_uses_the_run_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    results = run(paradigm(), scripted(), progress=False)
    key = results.manifest["key"]
    assert len(key) == 12
    assert (tmp_path / "runs" / "iterated_learning" / key / "manifest.json").exists()


def test_device_and_batch_size_do_not_change_the_key(tiny_model_and_tokenizer, tmp_path):
    model, tok = tiny_model_and_tokenizer
    made = paradigm(generations=1, chains=1)
    one = run(made, HFSubject(model, tok, device="cpu", batch_size=1), out=tmp_path / "r", progress=False)
    two = run(made, HFSubject(model, tok, device="cpu", batch_size=8), out=tmp_path / "r", progress=False)
    assert one.manifest["key"] == two.manifest["key"]
    assert two.manifest["sessions"][-1]["batch_size"] == 8


def test_logits_are_recorded_and_reload(tiny_model_and_tokenizer, tmp_path):
    model, tok = tiny_model_and_tokenizer
    results = go(paradigm(), HFSubject(model, tok, device="cpu"), tmp_path, record=["logits"])
    assert results.logits(0).shape == (2 * 8, len(tok))
    assert Results.load(results.path).logits(2).shape == (2 * 8, len(tok))


def test_logits_not_recorded_raises_clearly(tmp_path):
    with pytest.raises(MissingAbilityError, match="logits"):
        go(paradigm(), scripted(), tmp_path).logits(0)


def test_weights_mode_trains_one_learner_per_chain(tmp_path):
    subject = TrainableScriptedSubject()
    go(paradigm(), subject, tmp_path, learning="weights")
    assert len(subject.fresh_seeds) == 6
    assert len(set(subject.fresh_seeds)) == 2
    assert all(len(seen) == 4 for seen in subject.learned)
    assert subject.max_context_seen == 0


def test_weights_mode_keeps_query_order(tmp_path):
    subject = TrainableScriptedSubject()
    frame = go(paradigm(), subject, tmp_path, learning="weights").to_pandas()
    first = frame[frame["generation"] == 0]
    assert first["chain"].tolist() == [0] * 8 + [1] * 8
    assert first["item"].tolist() == list(range(8)) * 2


def test_manifest_records_settings_seeds_and_examples(tmp_path):
    made = paradigm(examples=[("red circle", "A")], seed=5)
    manifest = go(made, scripted(), tmp_path).manifest
    assert manifest["paradigm"]["seed"] == 5
    assert manifest["uses_examples"] is True
    assert manifest["status"] == "complete"
    assert "torch" in manifest["versions"]


def test_bayesian_prior_reaches_the_manifest(tmp_path):
    task = ShepardCategories(type="VI")
    prior = task.type_prior({"I": 1.0, "VI": 1.0})
    results = run(
        IteratedLearning(task=task, generations=2, chains=2),
        BayesianLearner(prior=prior),
        out=tmp_path / "b",
        progress=False
    )
    assert results.manifest["reference"]["I"] == pytest.approx(0.5)
    assert set(results.metrics("kl")["reference"]) == {"subject_prior"}
