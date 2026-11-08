from __future__ import annotations

import subprocess
import sys

import pytest

import cognarium as cg


def test_import_does_not_load_transformers():
    code = "import sys, cognarium; assert 'transformers' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_import_does_not_load_torchmetrics():
    code = "import sys, cognarium; assert 'torchmetrics' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_every_public_name_is_exposed():
    names = {
        cg: [
            "load",
            "run",
            "Results",
            "Paradigm",
            "Step",
            "Stimulus",
            "Trial",
            "Observation",
            "Query",
            "Responses",
            "PromptTemplate",
            "HiddenStates",
            "LayerMap",
            "Readout",
            "MissingAbilityError",
            "RunConflictError",
            "Effect",
            "IdenticalParticipantsWarning",
            "context",
            "data"
        ],
        cg.paradigms: [paradigm.__name__ for paradigm in cg.paradigms.PARADIGMS.values()],
        cg.data: ["available", "load", "from_csv", "from_json", "Dataset", "Entry"],
        cg.tasks: ["ShepardCategories"],
        cg.subjects: [
            "Subject",
            "HFSubject",
            "CustomSubject",
            "LSTM",
            "ExemplarLearner",
            "BayesianLearner"
        ],
        cg.models: ["LSTMLanguageModel", "SimpleTokenizer"],
        cg.metrics: [
            "entropy",
            "kl_divergence",
            "surprisal",
            "item_correlation",
            "split_half_ceiling",
            "effect_comparison"
        ]
    }
    for module, exposed in names.items():
        for name in exposed:
            assert hasattr(module, name), f"{module.__name__}.{name}"
    for name in names[cg]:
        assert name in cg.__all__, name
    assert len(cg.paradigms.PARADIGMS) == 3
    assert set(names[cg.paradigms]) <= set(cg.paradigms.__all__)


def test_the_version_is_the_first_release():
    assert cg.__version__ == "0.1.0"


def test_the_approved_shape_runs(tmp_path):
    task = cg.tasks.ShepardCategories(type="VI")
    bayes = cg.subjects.BayesianLearner(prior=task.type_prior({"I": 1.0, "VI": 1.0}))
    results = cg.run(
        cg.paradigms.IteratedLearning(task=task, generations=2, chains=2),
        bayes,
        learning="posterior",
        out=tmp_path / "bayes",
        progress=False
    )
    assert set(results.metrics("entropy", "kl")["metric"]) == {"entropy", "kl"}


def test_the_approved_example_runs_in_full(tiny_model_and_tokenizer, tmp_path):
    model, tok = tiny_model_and_tokenizer
    task = cg.tasks.ShepardCategories(type="VI")
    results = cg.run(
        cg.paradigms.IteratedLearning(
            task=task,
            generations=2,
            chains=2,
            instructions="Sort each object into category A or B.",
            examples=[("small red circle", "A")],
            temperature=0.7,
            top_p=0.9
        ),
        cg.load(model, tokenizer=tok, device="cpu"),
        learning="in_context",
        record=["logits", "hidden_states"],
        out=tmp_path / "run",
        progress=False
    )
    measures = set(results.metrics("accuracy", "entropy", "surprisal")["metric"])
    assert measures == {"accuracy", "entropy", "surprisal"}
    assert not results.lens().empty
    results.hidden.save(tmp_path / "il_gpt2")
    reopened = cg.HiddenStates.open(tmp_path / "il_gpt2")
    first, last = reopened.layers[0], reopened.layers[-1]
    saved = cg.metrics.linear_cka(reopened.to_tensor(first), reopened.to_tensor(last))
    live = cg.metrics.linear_cka(results.hidden.to_tensor(first), results.hidden.to_tensor(last))
    assert saved == live


def test_load_gives_a_custom_module_its_recording_paths(tmp_path):
    tok = cg.models.SimpleTokenizer.from_texts(
        ["Object: small large red blue circle square Category: A B"]
    )
    subject = cg.load(
        cg.models.LSTMLanguageModel(len(tok)),
        tokenizer=tok,
        device="cpu",
        layers={"lstm": "lstm"},
        readout="readout"
    )
    results = cg.run(
        cg.paradigms.IteratedLearning(
            task=cg.tasks.ShepardCategories(type="VI"),
            generations=1,
            chains=1
        ),
        subject,
        learning="in_context",
        record=["hidden_states"],
        out=tmp_path / "run",
        progress=False
    )
    assert results.hidden.layers == ("lstm",)
    assert not results.lens().empty


def test_load_gives_a_hugging_face_model_its_layer_map(tiny_model_and_tokenizer, tmp_path):
    model, tok = tiny_model_and_tokenizer
    model.save_pretrained(tmp_path / "model")
    tok.save_pretrained(tmp_path / "model")
    layer_map = cg.LayerMap(layers="transformer.h", norm=None, head="lm_head")
    from_object = cg.load(
        model,
        tokenizer=tok,
        device="cpu",
        layer_map=layer_map
    )
    from_name = cg.load(str(tmp_path / "model"), device="cpu", layer_map=layer_map)
    assert from_object.layer_map == layer_map
    assert from_name.layer_map == layer_map


def test_load_refuses_a_setting_the_model_can_not_use(tiny_model_and_tokenizer):
    model, tok = tiny_model_and_tokenizer
    with pytest.raises(TypeError, match="layers"):
        cg.load(
            model,
            tokenizer=tok,
            device="cpu",
            layers={"a": "transformer.h.0"}
        )
    with pytest.raises(TypeError, match="readout"):
        cg.load(
            model,
            tokenizer=tok,
            device="cpu",
            readout="lm_head"
        )
    with pytest.raises(TypeError, match="layers"):
        cg.load("not-read-because-the-check-comes-first", layers={"a": "b"})
    custom = cg.models.LSTMLanguageModel(len(tok))
    with pytest.raises(TypeError, match="layer_map"):
        cg.load(
            custom,
            tokenizer=tok,
            device="cpu",
            layer_map=cg.LayerMap(layers="a", norm=None, head="b")
        )
