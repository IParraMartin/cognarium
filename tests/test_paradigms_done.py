"""The definition of done, checked for every built-in paradigm."""

from __future__ import annotations

import numpy as np
import pytest
from paradigm_cases import CASES

import cognarium as cg
from cognarium.paradigms import PARADIGMS
from cognarium.subjects.hf import HFSubject

NAMES = sorted(set(PARADIGMS) - {"iterated_learning"})
_NUMBERS = ["difference", "ci_low", "ci_high", "p_value"]


def run_quietly(paradigm, subject, folder):
    return cg.run(
        paradigm,
        subject,
        out=folder,
        progress=False
    )


def texts(paradigm):
    return [trial.stimulus.text for trial in paradigm.sequences()[0]]


def test_every_paradigm_has_a_case():
    assert sorted(CASES) == NAMES


@pytest.mark.parametrize("name", NAMES)
def test_scoring_detects_the_declared_effect(name, tmp_path):
    case = CASES[name]
    paradigm = case.build()
    results = run_quietly(paradigm, case.subject(paradigm.effects[0]), tmp_path / "run")
    primary = results.score().iloc[0]
    assert primary["detected"] and primary["difference"] > 0


@pytest.mark.parametrize("name", NAMES)
def test_scoring_reports_nothing_when_there_is_no_effect(name, tmp_path):
    case = CASES[name]
    results = run_quietly(case.build(), case.subject(None), tmp_path / "run")
    assert not results.score().iloc[0]["detected"]


@pytest.mark.parametrize("name", NAMES)
def test_a_tiny_model_runs_it(name, byte_model_and_tokenizer, tmp_path):
    model, tokenizer = byte_model_and_tokenizer
    subject = HFSubject(model, tokenizer, device="cpu")
    table = run_quietly(CASES[name].build(), subject, tmp_path / "run").score()
    assert np.isfinite(table[_NUMBERS].to_numpy(dtype=float)).all()


@pytest.mark.parametrize("name", NAMES)
def test_the_bundled_lstm_runs_it(name, tmp_path):
    subject = cg.subjects.LSTM(
        device="cpu",
        train_epochs=1,
        embedding_dim=4,
        hidden_dim=4
    )
    table = run_quietly(CASES[name].build(), subject, tmp_path / "run").score()
    assert np.isfinite(table[_NUMBERS].to_numpy(dtype=float)).all()


@pytest.mark.parametrize("name", NAMES)
def test_it_resumes_without_new_calls(name, tmp_path):
    case = CASES[name]
    first = case.subject(None)
    run_quietly(case.build(), first, tmp_path / "run")
    again = case.subject(None)
    run_quietly(case.build(), again, tmp_path / "run")
    assert first.calls > 0 and again.calls == 0


@pytest.mark.parametrize("name", NAMES)
def test_trials_depend_only_on_the_seed(name):
    build = CASES[name].build
    assert texts(build(seed=3)) == texts(build(seed=3))
    assert texts(build(seed=3)) != texts(build(seed=4))


@pytest.mark.parametrize("name", NAMES)
def test_default_stimuli_are_labelled_not_human_normed(name):
    assert CASES[name].build().config()["stimuli"] == "built-in, not human-normed"


@pytest.mark.parametrize("name", NAMES)
def test_the_docstring_states_a_mapping(name):
    docstring = " ".join(PARADIGMS[name].__doc__.split())
    assert "Maps onto" in docstring and "can't capture" in docstring and "not human-normed" in docstring


@pytest.mark.parametrize("name", NAMES)
def test_it_is_exposed(name):
    assert getattr(cg.paradigms, PARADIGMS[name].__name__) is PARADIGMS[name]
