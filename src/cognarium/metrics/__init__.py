"""Metrics, usable on plain tensors or through `Results.metrics`."""

from __future__ import annotations

from cognarium.metrics.classification import (
    accuracy,
    auprc,
    auroc,
    balanced_accuracy,
    brier,
    ece,
    f1,
    mcc,
    precision,
    recall
)
from cognarium.metrics.detection import a_prime, criterion, d_prime
from cognarium.metrics.information import (
    conditional_entropy,
    cross_entropy,
    entropy,
    information_bottleneck,
    joint_entropy,
    js_divergence,
    kl_divergence,
    ksg_mutual_information,
    mutual_information,
    perplexity,
    surprisal
)
from cognarium.metrics.statistics import (
    bootstrap_ci,
    cohens_d,
    fdr,
    hedges_g,
    permutation_test
)

__all__ = [
    "a_prime",
    "accuracy",
    "auprc",
    "auroc",
    "balanced_accuracy",
    "bootstrap_ci",
    "brier",
    "cohens_d",
    "conditional_entropy",
    "criterion",
    "cross_entropy",
    "d_prime",
    "ece",
    "entropy",
    "f1",
    "fdr",
    "hedges_g",
    "information_bottleneck",
    "joint_entropy",
    "js_divergence",
    "kl_divergence",
    "ksg_mutual_information",
    "mcc",
    "mutual_information",
    "permutation_test",
    "perplexity",
    "precision",
    "recall",
    "surprisal"
]
