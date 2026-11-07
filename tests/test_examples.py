from __future__ import annotations

import importlib.util
from pathlib import Path

EXAMPLE = Path(__file__).parent.parent / "examples" / "iterated_learning.py"


def load_main():
    spec = importlib.util.spec_from_file_location("iterated_learning_example", EXAMPLE)
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
