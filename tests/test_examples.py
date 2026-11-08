from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from cognarium.paradigms import PARADIGMS

EXAMPLES = Path(__file__).parent.parent / "examples"
TOURED = (
    "stroop",
    "n_back",
    "category_learning",
    "minimal_pairs",
    "framing"
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


def test_tour_runs_on_a_tiny_model(byte_model_and_tokenizer, tmp_path):
    model, tokenizer = byte_model_and_tokenizer
    model.save_pretrained(tmp_path / "model")
    tokenizer.save_pretrained(tmp_path / "model")
    args = [
        "--model",
        str(tmp_path / "model"),
        "--device",
        "cpu",
        "--out",
        str(tmp_path / "runs"),
        "--docs",
        str(tmp_path / "results"),
        "--screen",
        str(tmp_path / "runs.js")
    ]
    for name in TOURED:
        args += ["--only", name]
    results = load_main("paradigm_tour")(args)
    assert set(results) == set(TOURED)
    for name in TOURED:
        page = (tmp_path / "results" / f"{name}.md").read_text(encoding="utf-8")
        assert str(tmp_path / "model") in page
        assert "Sample prompt" in page
        assert "| effect |" in page
        assert not results[name].score().empty
    screen = (tmp_path / "runs.js").read_text(encoding="utf-8")
    runs = json.loads(screen[screen.index("["):screen.rindex("]") + 1])
    assert [run["name"] for run in runs] == [name for name in PARADIGMS if name in TOURED]
    stroop = runs[0]
    assert stroop["page"] == "control" and stroop["anchor"] == "stroop"
    assert stroop["options"] == ["red", "blue", "green", "yellow"]
    assert abs(sum(stroop["probabilities"]) - 1) < 1e-6
    assert [effect["effect"] for effect in stroop["effects"]] == ["interference", "errors"]


def test_tour_keeps_the_screen_entries_it_did_not_rerun(byte_model_and_tokenizer, tmp_path):
    model, tokenizer = byte_model_and_tokenizer
    model.save_pretrained(tmp_path / "model")
    tokenizer.save_pretrained(tmp_path / "model")
    args = [
        "--model",
        str(tmp_path / "model"),
        "--device",
        "cpu",
        "--out",
        str(tmp_path / "runs"),
        "--docs",
        str(tmp_path / "results"),
        "--screen",
        str(tmp_path / "runs.js")
    ]
    tour = load_main("paradigm_tour")
    tour([*args, "--only", "framing"])
    tour([*args, "--only", "stroop"])
    screen = (tmp_path / "runs.js").read_text(encoding="utf-8")
    runs = json.loads(screen[screen.index("["):screen.rindex("]") + 1])
    assert [run["name"] for run in runs] == ["stroop", "framing"]
