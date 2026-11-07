from __future__ import annotations

import math

import dit
import numpy as np
import pytest
import scipy.spatial.distance
import scipy.stats
import sklearn.metrics
import torch

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


def test_entropy_matches_scipy():
    p = torch.tensor([0.5, 0.25, 0.25], dtype=torch.float64)
    assert entropy(p.log()).item() == pytest.approx(scipy.stats.entropy(p.numpy(), base=2))
    assert entropy(p.log()).item() == pytest.approx(1.5)


def test_kl_matches_scipy_and_is_never_negative():
    p = torch.tensor([0.7, 0.3], dtype=torch.float64)
    q = torch.tensor([0.4, 0.6], dtype=torch.float64)
    expected = scipy.stats.entropy(p.numpy(), q.numpy(), base=2)
    assert kl_divergence(p.log(), q.log()).item() == pytest.approx(expected)
    assert kl_divergence(p.log(), p.log()).item() == pytest.approx(0.0)
    assert kl_divergence(p.log(), q.log()) >= 0


def test_impossible_outcomes_are_handled():
    certain = torch.tensor([0.0, -math.inf], dtype=torch.float64)
    assert entropy(certain).item() == 0.0
    even = torch.log(torch.tensor([0.5, 0.5], dtype=torch.float64))
    assert kl_divergence(certain, even).item() == pytest.approx(1.0)
    assert kl_divergence(even, certain).item() == math.inf


def test_surprisal_in_bits():
    assert surprisal(torch.log(torch.tensor(0.25))).item() == pytest.approx(2.0)


def test_base_changes_the_unit():
    even = torch.log(torch.tensor([0.5, 0.5], dtype=torch.float64))
    assert entropy(even, base=math.e).item() == pytest.approx(math.log(2))


def test_entropy_of_a_certain_outcome_is_positive_zero():
    certain = torch.tensor([0.0, -math.inf], dtype=torch.float64)
    assert math.copysign(1.0, entropy(certain).item()) == 1.0


def double_log(values):
    return torch.tensor(values, dtype=torch.float64).log()


def test_cross_entropy_is_entropy_plus_kl():
    p, q = double_log([0.7, 0.3]), double_log([0.4, 0.6])
    assert cross_entropy(p, q).item() == pytest.approx((entropy(p) + kl_divergence(p, q)).item())


def test_js_matches_scipy_and_is_symmetric():
    p, q = [0.7, 0.3], [0.4, 0.6]
    expected = scipy.spatial.distance.jensenshannon(p, q, base=2) ** 2
    assert js_divergence(double_log(p), double_log(q)).item() == pytest.approx(expected)
    assert js_divergence(double_log(q), double_log(p)).item() == pytest.approx(expected)


def test_js_stays_finite_when_supports_differ():
    a = torch.tensor([0.0, -math.inf], dtype=torch.float64)
    b = torch.tensor([-math.inf, 0.0], dtype=torch.float64)
    assert js_divergence(a, b).item() == pytest.approx(1.0)


def test_mi_of_a_variable_with_itself_is_its_entropy():
    joint = torch.diag(torch.tensor([0.5, 0.25, 0.25], dtype=torch.float64)).log()
    assert mutual_information(joint).item() == pytest.approx(1.5)


def test_independent_variables_share_nothing():
    joint = torch.full((2, 2), 0.25, dtype=torch.float64).log()
    assert joint_entropy(joint).item() == pytest.approx(2.0)
    assert conditional_entropy(joint).item() == pytest.approx(1.0)
    assert mutual_information(joint).item() == pytest.approx(0.0)


def test_mi_matches_scikit_learn():
    counts = np.array([[10, 2], [3, 15]])
    joint = torch.tensor(counts / counts.sum(), dtype=torch.float64).log()
    expected = sklearn.metrics.mutual_info_score(None, None, contingency=counts)
    assert mutual_information(joint, base=math.e).item() == pytest.approx(expected)


def test_joint_measures_match_dit():
    outcomes = ["00", "01", "10", "11"]
    probabilities = [0.4, 0.1, 0.2, 0.3]
    distribution = dit.Distribution(outcomes, probabilities)
    joint = torch.tensor(probabilities, dtype=torch.float64).reshape(2, 2).log()
    assert joint_entropy(joint).item() == pytest.approx(dit.shannon.entropy(distribution))
    assert conditional_entropy(joint).item() == pytest.approx(
        dit.shannon.conditional_entropy(distribution, [1], [0])
    )
    assert mutual_information(joint).item() == pytest.approx(
        dit.shannon.mutual_information(distribution, [0], [1])
    )


def test_perplexity_is_the_inverse_geometric_mean():
    assert perplexity(double_log([0.5, 0.25])).item() == pytest.approx(8**0.5)


def test_ksg_recovers_a_gaussian():
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(2000, 1, generator=generator, dtype=torch.float64)
    y = 0.8 * x + 0.6 * torch.randn(2000, 1, generator=generator, dtype=torch.float64)
    unrelated = torch.randn(2000, 1, generator=generator, dtype=torch.float64)
    assert ksg_mutual_information(x, y) == pytest.approx(-0.5 * math.log2(1 - 0.64), abs=0.05)
    assert abs(ksg_mutual_information(x, unrelated)) < 0.05


def test_ksg_rejects_too_few_samples():
    with pytest.raises(ValueError, match="k"):
        ksg_mutual_information(torch.zeros(3, 1), torch.zeros(3, 1), k=3)


def test_ksg_rejects_mismatched_rows():
    with pytest.raises(ValueError, match="rows"):
        ksg_mutual_information(torch.zeros(10, 1), torch.zeros(11, 1))


def test_bottleneck_hits_its_limits():
    joint = double_log([[0.4, 0.1], [0.1, 0.4]])
    table = information_bottleneck(joint, [1e-3, 200.0]).set_index("beta")
    assert table.loc[1e-3, "complexity"] < 1e-3
    assert table.loc[1e-3, "relevance"] < 1e-3
    assert table.loc[200.0, "relevance"] == pytest.approx(mutual_information(joint).item(), abs=1e-3)


def test_bottleneck_obeys_data_processing():
    joint = double_log([[0.3, 0.05, 0.05], [0.05, 0.3, 0.05], [0.05, 0.05, 0.1]])
    table = information_bottleneck(joint, [0.5, 2.0, 8.0], n_clusters=2)
    assert (table["relevance"] <= table["complexity"] + 1e-9).all()
    assert (table["relevance"] <= mutual_information(joint).item() + 1e-9).all()


def test_bottleneck_is_reproducible():
    joint = double_log([[0.4, 0.1], [0.1, 0.4]])
    assert information_bottleneck(joint, [2.0], seed=3).equals(information_bottleneck(joint, [2.0], seed=3))


def test_bottleneck_handles_an_impossible_outcome():
    joint = double_log([[0.5, 0.0], [0.0, 0.5]])
    table = information_bottleneck(joint, [50.0])
    assert table.loc[0, "relevance"] == pytest.approx(1.0, abs=1e-3)


def test_ksg_rejects_duplicate_samples():
    with pytest.raises(ValueError, match="duplicate"):
        ksg_mutual_information(torch.zeros(10, 1), torch.zeros(10, 1), k=3)


@pytest.mark.parametrize("n_clusters", [2, 3, 6])
def test_bottleneck_stays_finite_when_clusters_die(n_clusters):
    joint = double_log([[0.5, 0.0], [0.0, 0.5]])
    table = information_bottleneck(joint, [0.01, 1.0, 1000.0], n_clusters=n_clusters)
    assert table[["complexity", "relevance"]].notna().all().all()
