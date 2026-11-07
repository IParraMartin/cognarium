from __future__ import annotations

import cognarium as cg
from cognarium.paradigms.learning import IteratedLearning
from cognarium.runner import run
from cognarium.subjects.bayesian import BayesianLearner
from cognarium.tasks.shepard import ShepardCategories

WEIGHTS = {
    "I": 0.4,
    "II": 0.1,
    "III": 0.1,
    "IV": 0.1,
    "V": 0.1,
    "VI": 0.1,
    "unbalanced": 0.1
}


def final_kl(decision, tmp_path, noise=0.0):
    task = ShepardCategories(type="VI")
    prior = task.type_prior(WEIGHTS)
    made = IteratedLearning(
        task=task,
        generations=15,
        chains=1000,
        noise=noise
    )
    results = run(
        made,
        BayesianLearner(prior=prior, decision=decision),
        out=tmp_path / decision,
        progress=False
    )
    table = results.metrics("kl", smoothing=0.0)
    return table.set_index("generation")["value"].to_dict()


def test_sampling_chains_converge_to_the_prior(tmp_path):
    kl = final_kl("sample", tmp_path)
    assert kl[0] > 0.5
    assert kl[14] < 0.02


def test_noisy_sampling_chains_also_converge(tmp_path):
    assert final_kl("sample", tmp_path, noise=0.1)[14] < 0.02


def test_map_chains_do_not_converge_to_the_prior(tmp_path):
    assert final_kl("map", tmp_path)[14] > 0.5


def test_every_metric_family_is_exposed():
    families = [
        [
            "conditional_entropy",
            "cross_entropy",
            "entropy",
            "information_bottleneck",
            "joint_entropy",
            "js_divergence",
            "kl_divergence",
            "ksg_mutual_information",
            "mutual_information",
            "perplexity",
            "surprisal"
        ],
        [
            "accuracy",
            "auprc",
            "auroc",
            "balanced_accuracy",
            "brier",
            "ece",
            "f1",
            "mcc",
            "precision",
            "recall"
        ],
        ["a_prime", "criterion", "d_prime"],
        [
            "bootstrap_ci",
            "cohens_d",
            "fdr",
            "hedges_g",
            "permutation_test"
        ],
        [
            "GEOMETRY",
            "intrinsic_dimension",
            "layer_matrix",
            "linear_cka",
            "mse",
            "participation_ratio",
            "procrustes",
            "pwcca",
            "rbf_cka",
            "rsa",
            "svcca"
        ]
    ]
    for family in families:
        for name in family:
            assert hasattr(cg.metrics, name), name
