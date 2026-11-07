from __future__ import annotations

import subprocess
import sys

import cognarium as cg


def test_import_does_not_load_transformers():
    code = "import sys, cognarium; assert 'transformers' not in sys.modules"
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
            "MissingAbilityError",
            "RunConflictError"
        ],
        cg.paradigms: ["IteratedLearning"],
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
        cg.metrics: ["entropy", "kl_divergence", "surprisal"]
    }
    for module, exposed in names.items():
        for name in exposed:
            assert hasattr(module, name), f"{module.__name__}.{name}"


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
