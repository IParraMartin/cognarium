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
from cognarium.metrics.comparison import effect_comparison, item_correlation, split_half_ceiling
from cognarium.metrics.detection import a_prime, criterion, d_prime
from cognarium.metrics.geometry import (
    GEOMETRY,
    intrinsic_dimension,
    layer_matrix,
    linear_cka,
    mse,
    participation_ratio,
    procrustes,
    pwcca,
    rbf_cka,
    rsa,
    svcca
)
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
from cognarium.metrics.naming import (
    Frontier,
    co_naming_mse,
    efficiency_loss,
    gnid,
    ib_accuracy,
    ib_complexity,
    ib_frontier,
    meanings
)
from cognarium.metrics.statistics import (
    bootstrap_ci,
    cohens_d,
    contrast_table,
    fdr,
    hedges_g,
    permutation_test
)

__all__ = [
    "GEOMETRY",
    "Frontier",
    "a_prime",
    "accuracy",
    "auprc",
    "auroc",
    "balanced_accuracy",
    "bootstrap_ci",
    "brier",
    "co_naming_mse",
    "cohens_d",
    "conditional_entropy",
    "contrast_table",
    "criterion",
    "cross_entropy",
    "d_prime",
    "ece",
    "effect_comparison",
    "efficiency_loss",
    "entropy",
    "f1",
    "fdr",
    "gnid",
    "hedges_g",
    "ib_accuracy",
    "ib_complexity",
    "ib_frontier",
    "information_bottleneck",
    "intrinsic_dimension",
    "item_correlation",
    "joint_entropy",
    "js_divergence",
    "kl_divergence",
    "ksg_mutual_information",
    "layer_matrix",
    "linear_cka",
    "mcc",
    "meanings",
    "mse",
    "mutual_information",
    "participation_ratio",
    "permutation_test",
    "perplexity",
    "precision",
    "procrustes",
    "pwcca",
    "rbf_cka",
    "recall",
    "rsa",
    "split_half_ceiling",
    "surprisal",
    "svcca"
]
