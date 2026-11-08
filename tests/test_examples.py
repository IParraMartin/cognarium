from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

EXAMPLES = Path(__file__).parent.parent / "examples"
TOURED = (
    "iterated_learning",
    "category_learning",
    "artificial_grammar"
)


def load_main(name="iterated_learning"):
    spec = importlib.util.spec_from_file_location(f"{name}_example", EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def test_example_runs_all_four_subject_types(tiny_model_and_tokenizer, tmp_path):
    model, tok = tiny_model_and_tokenizer
    model.save_pretrained(tmp_path / "model")
    tok.save_pretrained(tmp_path / "model")
    args = [
        "--model",
        str(tmp_path / "model"),
        "--device",
        "cpu",
        "--chains",
        "2",
        "--generations",
        "2",
        "--out",
        str(tmp_path / "runs")
    ]
    results = load_main()(args)
    assert set(results) == {"hf", "lstm", "exemplar", "bayesian", "store"}
    assert results["hf"].logits(0).shape[0] == 16
    assert results["lstm"].manifest["learning"] == "weights"
    assert not results["hf"].lens().empty
    assert results["store"].layers == results["hf"].hidden.layers
    rerun = load_main()(args)
    assert rerun["store"].path == results["store"].path
    assert rerun["store"].manifest["key"] == results["hf"].manifest["key"]


def test_tour_runs_on_a_tiny_model(
    byte_model_and_tokenizer,
    tmp_path,
    capsys
):
    model, tokenizer = byte_model_and_tokenizer
    model.save_pretrained(tmp_path / "model")
    tokenizer.save_pretrained(tmp_path / "model")
    args = [
        "--model",
        str(tmp_path / "model"),
        "--device",
        "cpu",
        "--out",
        str(tmp_path / "runs")
    ]
    for name in TOURED:
        args += ["--only", name]
    results = load_main("paradigm_tour")(args)
    assert set(results) == set(TOURED)
    printed = capsys.readouterr().out
    for name in TOURED:
        assert f"{name}: " in printed
    assert not results["category_learning"].score().empty
    assert not results["iterated_learning"].metrics("accuracy").empty



def test_function_learning_reports_the_median_correlation_per_generation(tmp_path):
    args = [
        "--model",
        "exemplar",
        "--chains",
        "2",
        "--generations",
        "2",
        "--conditions",
        "positive",
        "negative",
        "--out",
        str(tmp_path / "runs")
    ]
    medians = load_main("function_learning")(args)
    assert len(medians) == 2 * 3
    first = medians[medians["generation"] == 0].set_index("condition")["correlation"]
    assert first.tolist() == pytest.approx([1.0, -1.0])
    answers = pd.read_csv(tmp_path / "runs" / "answers.csv")
    assert len(answers) == 2 * 2 * 2 * 100
    assert answers["y"].between(1, 100).all()


def test_function_learning_runs_on_a_language_model(byte_model_and_tokenizer, tmp_path):
    model, tokenizer = byte_model_and_tokenizer
    model.save_pretrained(tmp_path / "model")
    tokenizer.save_pretrained(tmp_path / "model")
    args = [
        "--model",
        str(tmp_path / "model"),
        "--device",
        "cpu",
        "--chains",
        "1",
        "--generations",
        "1",
        "--examples",
        "5",
        "--conditions",
        "random",
        "--out",
        str(tmp_path / "runs")
    ]
    medians = load_main("function_learning")(args)
    assert medians["generation"].tolist() == [0, 1]


def test_shepard_types_order_the_rational_learner_like_people(tmp_path):
    args = [
        "--model",
        "bayesian",
        "--participants",
        "10",
        "--blocks",
        "2",
        "--out",
        str(tmp_path / "runs")
    ]
    curves = load_main("shepard_types")(args)
    assert len(curves) == 6 * 2
    means = curves.groupby("type")["error"].mean()
    assert means["I"] < means["II"] < min(means["III"], means["IV"], means["V"])
    assert max(means["III"], means["IV"], means["V"]) < means["VI"]
    assert (tmp_path / "runs" / "errors.csv").exists()


def test_shepard_types_run_on_a_language_model(byte_model_and_tokenizer, tmp_path, capsys):
    model, tokenizer = byte_model_and_tokenizer
    model.save_pretrained(tmp_path / "model")
    tokenizer.save_pretrained(tmp_path / "model")
    args = [
        "--model",
        str(tmp_path / "model"),
        "--device",
        "cpu",
        "--participants",
        "1",
        "--blocks",
        "2",
        "--out",
        str(tmp_path / "runs")
    ]
    curves = load_main("shepard_types")(args)
    assert curves["error"].between(0, 1).all()
    assert "rank correlation with the human order" in capsys.readouterr().out


def test_grammar_learning_runs_both_tests_on_a_language_model(byte_model_and_tokenizer, tmp_path, capsys):
    model, tokenizer = byte_model_and_tokenizer
    model.save_pretrained(tmp_path / "model")
    tokenizer.save_pretrained(tmp_path / "model")
    args = [
        "--model",
        str(tmp_path / "model"),
        "--device",
        "cpu",
        "--participants",
        "2",
        "--n-train",
        "6",
        "--n-test",
        "4",
        "--out",
        str(tmp_path / "runs")
    ]
    scores = load_main("grammar_learning")(args)
    assert list(scores) == ["judgment", "probability"]
    assert scores["judgment"]["effect"].tolist() == ["grammaticality", "chunk_strength"]
    assert scores["probability"]["effect"].tolist() == ["preference"]
    assert "probability of answering grammatical" in capsys.readouterr().out


def test_grammar_learning_gives_a_feature_learner_only_the_judgment_test(tmp_path):
    args = [
        "--model",
        "exemplar",
        "--participants",
        "2",
        "--out",
        str(tmp_path / "runs")
    ]
    assert list(load_main("grammar_learning")(args)) == ["judgment"]
