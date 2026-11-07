from __future__ import annotations

import math

import pytest
import scipy.spatial
import scipy.stats
import torch

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


def matrix(n=50, d=10, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=generator, dtype=torch.float64)


def rotation(d, seed=1):
    return torch.linalg.qr(matrix(d, d, seed))[0]


def test_cka_of_a_matrix_with_itself_is_one():
    x = matrix()
    for score in [linear_cka(x, x), linear_cka(x, x, debiased=True), rbf_cka(x, x)]:
        assert score == pytest.approx(1.0)


def test_cka_ignores_rotation_and_scale():
    x = matrix()
    y = 3.0 * x @ rotation(10)
    assert linear_cka(x, y) == pytest.approx(1.0)
    assert rbf_cka(x, y) == pytest.approx(1.0)


def test_cka_gives_the_same_score_from_features_and_from_grams():
    x, y = matrix(n=20, d=5), matrix(n=20, d=5, seed=2)
    centered_x, centered_y = x - x.mean(dim=0), y - y.mean(dim=0)
    gram_x, gram_y = centered_x @ centered_x.T, centered_y @ centered_y.T
    expected = (gram_x * gram_y).sum() / (gram_x.norm() * gram_y.norm())
    assert linear_cka(x, y) == pytest.approx(expected.item())
    wide_x, wide_y = matrix(n=5, d=20), matrix(n=5, d=20, seed=2)
    centered_x, centered_y = wide_x - wide_x.mean(dim=0), wide_y - wide_y.mean(dim=0)
    gram_x, gram_y = centered_x @ centered_x.T, centered_y @ centered_y.T
    expected = (gram_x * gram_y).sum() / (gram_x.norm() * gram_y.norm())
    assert linear_cka(wide_x, wide_y) == pytest.approx(expected.item())


def test_debiased_cka_of_unrelated_matrices_is_near_zero():
    assert abs(linear_cka(matrix(seed=2), matrix(seed=3), debiased=True)) < 0.1


def test_debiased_cka_needs_four_rows():
    with pytest.raises(ValueError, match="4 rows"):
        linear_cka(matrix(n=3), matrix(n=3, seed=2), debiased=True)


def test_rbf_cka_rejects_a_bad_sigma():
    with pytest.raises(ValueError, match="sigma_fraction"):
        rbf_cka(matrix(), matrix(), sigma_fraction=0.0)


def test_rsa_matches_scipy():
    x, y = matrix(), matrix(seed=4)
    expected = scipy.stats.spearmanr(
        scipy.spatial.distance.pdist(x.numpy(), "correlation"),
        scipy.spatial.distance.pdist(y.numpy(), "correlation")
    ).statistic
    assert rsa(x, y) == pytest.approx(expected)
    assert rsa(x, x) == pytest.approx(1.0)


def test_euclidean_pearson_rsa_matches_scipy():
    x, y = matrix(), matrix(seed=4)
    expected = scipy.stats.pearsonr(
        scipy.spatial.distance.pdist(x.numpy(), "euclidean"),
        scipy.spatial.distance.pdist(y.numpy(), "euclidean")
    ).statistic
    assert rsa(x, y, metric="euclidean", method="pearson") == pytest.approx(expected)


def test_rsa_averages_tied_ranks_like_scipy():
    x = torch.tensor([[0.0], [0.0], [1.0], [1.0], [2.0]], dtype=torch.float64)
    y = matrix(n=5, d=3, seed=4)
    expected = scipy.stats.spearmanr(
        scipy.spatial.distance.pdist(x.numpy(), "euclidean"),
        scipy.spatial.distance.pdist(y.numpy(), "euclidean")
    ).statistic
    assert rsa(x, y, metric="euclidean") == pytest.approx(expected)


@pytest.mark.parametrize("metric", ["correlation", "euclidean"])
def test_rsa_matches_rsatoolbox(metric):
    rsatoolbox = pytest.importorskip("rsatoolbox")
    x, y = matrix(), matrix(seed=4)
    rdms = [
        rsatoolbox.rdm.calc_rdm(rsatoolbox.data.Dataset(values.numpy()), method=metric)
        for values in (x, y)
    ]
    expected = rsatoolbox.rdm.compare(rdms[0], rdms[1], method="spearman")
    assert rsa(x, y, metric=metric) == pytest.approx(expected.item())


def test_rsa_rejects_unknown_options():
    with pytest.raises(ValueError, match="euclidean"):
        rsa(matrix(), matrix(seed=4), metric="nope")
    with pytest.raises(ValueError, match="spearman"):
        rsa(matrix(), matrix(seed=4), method="nope")


def test_rsa_of_equal_distances_raises():
    corners = torch.eye(5, dtype=torch.float64)
    with pytest.raises(ValueError, match="variance"):
        rsa(corners, matrix(n=5, d=3), metric="euclidean")


def test_euclidean_rsa_of_a_constant_matrix_raises():
    with pytest.raises(ValueError, match="variance"):
        rsa(torch.full((20, 4), 0.1, dtype=torch.float64), matrix(n=20, d=4), metric="euclidean")


def test_procrustes_is_zero_after_a_rotation_and_matches_scipy():
    x, y = matrix(), matrix(seed=5)
    assert procrustes(x, x @ rotation(10)) == pytest.approx(0.0, abs=1e-10)
    assert procrustes(x, y) == pytest.approx(scipy.spatial.procrustes(x.numpy(), y.numpy())[2])


def test_procrustes_handles_different_widths():
    assert procrustes(matrix(d=10), matrix(d=4, seed=6)) >= 0
    x, y = matrix(d=10), matrix(d=4, seed=6)
    padded = torch.nn.functional.pad(y, (0, 6))
    assert procrustes(x, y) == pytest.approx(scipy.spatial.procrustes(x.numpy(), padded.numpy())[2])


def test_procrustes_without_scale_keeps_the_scale_difference():
    x = matrix()
    assert procrustes(x, x @ rotation(10), scale=False) == pytest.approx(0.0, abs=1e-8)
    centered = x - x.mean(dim=0)
    expected = centered.square().sum().item()
    assert procrustes(x, 2 * x, scale=False) == pytest.approx(expected)
    assert procrustes(x, 2 * x) == pytest.approx(0.0, abs=1e-10)


def test_procrustes_with_more_columns_than_rows_matches_the_wide_form():
    x, y = matrix(n=5, d=20), matrix(n=5, d=20, seed=5)
    assert procrustes(x, y) == pytest.approx(scipy.spatial.procrustes(x.numpy(), y.numpy())[2])


def test_mse_hand_value():
    assert mse(torch.zeros(2, 2), torch.ones(2, 2)) == pytest.approx(1.0)


def test_mse_needs_matching_columns():
    with pytest.raises(ValueError, match="columns"):
        mse(matrix(d=3), matrix(d=4))


def test_cca_scores_ignore_invertible_maps():
    x = matrix(n=200)
    mixed = x @ (rotation(10) + 2 * torch.eye(10, dtype=torch.float64))
    assert svcca(x, mixed, variance=1.0) == pytest.approx(1.0)
    assert pwcca(x, mixed) == pytest.approx(1.0)


def test_cca_scores_are_bounded():
    x, y = matrix(n=200), matrix(n=200, seed=7)
    assert 0 <= svcca(x, y) <= 1
    assert 0 <= pwcca(x, y) <= 1


def test_cca_scores_of_a_shared_subspace_fall_between_the_extremes():
    shared, x_rest, y_rest = matrix(n=500, d=1), matrix(n=500, d=1, seed=2), matrix(n=500, d=1, seed=3)
    x, y = torch.cat([shared, x_rest], dim=1), torch.cat([shared, y_rest], dim=1)
    assert svcca(x, y, variance=1.0) == pytest.approx(0.5, abs=0.1)


def test_pwcca_handles_different_widths_in_either_order():
    x, y = matrix(n=200, d=10), matrix(n=200, d=4, seed=7)
    assert 0 <= pwcca(x, y) <= 1
    assert 0 <= pwcca(y, x) <= 1


def test_pwcca_weights_directions_by_how_much_of_x_they_explain():
    shared, rest, other = matrix(n=500, d=1), matrix(n=500, d=1, seed=2), matrix(n=500, d=1, seed=3)
    y = torch.cat([shared, other], dim=1)
    loud_shared = torch.cat([100 * shared, rest], dim=1)
    quiet_shared = torch.cat([shared, 100 * rest], dim=1)
    assert pwcca(loud_shared, y) > pwcca(quiet_shared, y)


def test_svcca_rejects_a_bad_variance():
    with pytest.raises(ValueError, match="variance"):
        svcca(matrix(), matrix(), variance=0.0)


@pytest.mark.parametrize("metric", [svcca, pwcca])
def test_cca_warns_when_the_rows_force_a_high_score(metric):
    with pytest.warns(UserWarning, match="rows"):
        score = metric(matrix(n=10, d=50), matrix(n=10, d=50, seed=10))
    assert 0 <= score <= 1 + 1e-9


def test_participation_ratio_limits():
    assert participation_ratio(matrix(n=5000)) == pytest.approx(10, rel=0.05)
    steps = torch.linspace(0, 1, 100, dtype=torch.float64)[:, None]
    line = steps @ torch.ones(1, 5, dtype=torch.float64)
    assert participation_ratio(line) == pytest.approx(1.0)


def test_participation_ratio_of_a_constant_matrix_raises():
    with pytest.raises(ValueError, match="variance"):
        participation_ratio(torch.ones(5, 3))


def test_intrinsic_dimension_finds_an_embedded_space():
    embedded = matrix(n=2000, d=3) @ matrix(n=3, d=20, seed=8)
    assert intrinsic_dimension(embedded) == pytest.approx(3, abs=0.5)


def test_intrinsic_dimension_drops_duplicates():
    points = matrix(n=500, d=2)
    assert intrinsic_dimension(torch.cat([points, points[:50]])) == pytest.approx(2, abs=0.5)


def test_intrinsic_dimension_needs_three_distinct_points():
    with pytest.raises(ValueError, match="distinct"):
        intrinsic_dimension(torch.tensor([[0.0], [1.0], [1.0]]))


def test_intrinsic_dimension_raises_when_neighbours_are_equally_far():
    with pytest.raises(ValueError, match="neighbour"):
        intrinsic_dimension(torch.arange(20.0)[:, None])


def test_layer_matrix_is_square_and_labelled():
    table = layer_matrix({"a": matrix(), "b": matrix(seed=9)})
    assert list(table.index) == ["a", "b"] == list(table.columns)
    assert table.loc["a", "a"] == pytest.approx(1.0)
    assert table.loc["a", "b"] == pytest.approx(table.loc["b", "a"])


def test_layer_matrix_accepts_a_callable():
    assert layer_matrix({"a": matrix()}, lambda x, y: 7.0).loc["a", "a"] == 7.0


def test_unknown_metric_lists_the_known_ones():
    with pytest.raises(ValueError, match="linear_cka"):
        layer_matrix({"a": matrix()}, "nope")


def test_debiased_cka_is_a_named_metric():
    assert sorted(GEOMETRY) == sorted(
        [
            "linear_cka",
            "debiased_cka",
            "rbf_cka",
            "rsa",
            "procrustes",
            "mse",
            "svcca",
            "pwcca"
        ]
    )
    x, y = matrix(seed=2), matrix(seed=3)
    assert layer_matrix({"a": x, "b": y}, "debiased_cka").loc["a", "b"] == pytest.approx(linear_cka(x, y, debiased=True))


@pytest.mark.parametrize("metric", [linear_cka, rbf_cka, rsa, procrustes, svcca, pwcca])
def test_constant_input_raises(metric):
    with pytest.raises(ValueError, match="variance"):
        metric(torch.ones(20, 4, dtype=torch.float64), matrix(n=20, d=4))


def test_a_constant_matrix_of_an_inexact_value_raises():
    with pytest.raises(ValueError, match="variance"):
        linear_cka(torch.full((7, 4), 0.1, dtype=torch.float64), matrix(n=7, d=4))


def test_one_dead_column_is_not_a_constant_matrix():
    x = matrix()
    x[:, 3] = 1.0
    assert linear_cka(x, x) == pytest.approx(1.0)


def test_constant_rows_raise_for_correlation_rsa():
    x = torch.arange(20.0, dtype=torch.float64)[:, None].repeat(1, 4)
    with pytest.raises(ValueError, match="variance"):
        rsa(x, matrix(n=20, d=4))


@pytest.mark.parametrize("metric", [linear_cka, rsa, procrustes, mse, svcca, pwcca])
def test_mismatched_rows_raise(metric):
    with pytest.raises(ValueError, match="rows"):
        metric(matrix(n=20), matrix(n=21))


def test_non_finite_input_raises():
    x = matrix()
    x[0, 0] = math.inf
    with pytest.raises(ValueError, match="finite"):
        linear_cka(x, matrix())


def test_too_few_rows_raise():
    with pytest.raises(ValueError, match="2 rows"):
        linear_cka(matrix(n=1), matrix(n=1))
    with pytest.raises(ValueError, match="2 rows"):
        participation_ratio(torch.tensor(1.0))


def test_trailing_dimensions_are_flattened():
    x = matrix(n=20, d=12)
    assert linear_cka(x.reshape(20, 3, 4), x) == pytest.approx(1.0)


def test_fewer_samples_than_features_still_works():
    x, y = matrix(n=10, d=50), matrix(n=10, d=50, seed=10)
    assert 0 <= linear_cka(x, y) <= 1


def test_geometry_runs_on_every_device(device):
    x = matrix().float().to(device)
    assert linear_cka(x, x) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("scale", [1e200, 1e120, 1e-200])
@pytest.mark.parametrize("name", sorted(set(GEOMETRY) - {"mse"}))
def test_scale_free_metrics_survive_extreme_values(name, scale):
    x, y = matrix(), matrix(seed=11)
    assert GEOMETRY[name](scale * x, y) == pytest.approx(GEOMETRY[name](x, y))


@pytest.mark.parametrize("scale", [1e200, 1e120, 1e-200])
def test_single_matrix_measures_survive_extreme_values(scale):
    x = matrix()
    assert participation_ratio(scale * x) == pytest.approx(participation_ratio(x))
    assert intrinsic_dimension(scale * x) == pytest.approx(intrinsic_dimension(x))


def test_columns_of_very_different_scales_stay_finite():
    x = matrix() * torch.logspace(-150, 150, 10, dtype=torch.float64)
    for score in [linear_cka(x, x), participation_ratio(x)]:
        assert math.isfinite(score)


def test_scale_dependent_metrics_raise_on_overflow():
    x, y = 1e200 * matrix(), 1e200 * matrix(seed=11)
    with pytest.raises(ValueError, match="too large"):
        mse(x, y)
    with pytest.raises(ValueError, match="too large"):
        procrustes(x, y, scale=False)
