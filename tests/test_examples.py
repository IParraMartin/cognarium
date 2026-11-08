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


def write_chips(folder):
    rows = [
        f"{number}\t{'ABCDEFGHIJ'[number % 10]}\t{number}\t4\t5R\t5.0\t{lightness}\t{a}\t{b}"
        for number, (lightness, a, b) in enumerate(
            [(lightness, a, b) for lightness in (20, 40, 60, 80, 95) for a, b in ((-20, 0), (20, 0), (0, -20), (0, 20))],
            start=1
        )
    ]
    path = folder / "chips.txt"
    path.write_text("#cnum\tV\tH\tC\tMunH\tMunV\tL*\ta*\tb*\n" + "\n".join(rows) + "\n")
    return path


def color_args(folder, model, *extra):
    return [
        "--model",
        model,
        "--chips",
        str(write_chips(folder)),
        "--chains",
        "2",
        "--generations",
        "2",
        "--memory",
        "2",
        "--out",
        str(folder / "runs"),
        *extra
    ]


def test_color_naming_measures_each_generation_from_the_random_start(tmp_path, capsys):
    measures = load_main("color_naming")(color_args(tmp_path, "exemplar", "--words", "2", "3"))
    assert set(measures["metric"]) == {"ib_complexity", "ib_accuracy", "efficiency_loss", "ib_alignment", "terms"}
    assert sorted(measures["generation"].unique()) == [0, 1, 2]
    assert len(measures) == 2 * 3 * 2 * 5
    assert measures["value"].notna().all()
    terms = measures[measures["metric"] == "terms"]
    assert (terms["value"] <= terms["words"]).all()
    assert (tmp_path / "runs" / "measures.csv").exists()
    assert "efficiency_loss" in capsys.readouterr().out


def test_color_naming_runs_on_a_language_model(byte_model_and_tokenizer, tmp_path):
    model, tokenizer = byte_model_and_tokenizer
    model.save_pretrained(tmp_path / "model")
    tokenizer.save_pretrained(tmp_path / "model")
    args = color_args(tmp_path, str(tmp_path / "model"), "--device", "cpu", "--words", "2", "--generations", "1")
    measures = load_main("color_naming")(args)
    assert sorted(measures["generation"].unique()) == [0, 1]


def test_color_naming_converts_cielab_to_srgb():
    spec = importlib.util.spec_from_file_location("color_naming_example", EXAMPLES / "color_naming.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lab = pd.DataFrame([[100, 0, 0], [0, 0, 0], [53.24, 80.09, 67.20], [50, 0, 0]]).to_numpy(dtype=float)
    rgb = module.lab_to_srgb(lab)
    assert rgb[0] == pytest.approx([1, 1, 1], abs=1e-3)
    assert rgb[1] == pytest.approx([0, 0, 0], abs=1e-6)
    assert rgb[2] == pytest.approx([1, 0, 0], abs=5e-3)
    assert rgb[3] == pytest.approx([0.466] * 3, abs=2e-3)
