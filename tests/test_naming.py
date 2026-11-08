from __future__ import annotations

import math

import pytest
import torch

from helpers import ScriptedSubject

import cognarium as cg
from cognarium.errors import MissingAbilityError
from cognarium.metrics import naming

POINTS = torch.arange(10, dtype=torch.float64)[:, None]
PRIOR = torch.full((10,), 0.1, dtype=torch.float64)


def one_hot(labels):
    words = sorted(set(labels))
    return torch.tensor([[float(label == word) for word in words] for label in labels], dtype=torch.float64)


@pytest.fixture(scope="module")
def space():
    meanings = naming.meanings(POINTS, sigma=1.0)
    return meanings, naming.ib_frontier(PRIOR, meanings)


def test_meanings_are_distributions_that_peak_on_their_item():
    meanings = naming.meanings(POINTS, sigma=1.0)
    assert torch.allclose(meanings.sum(dim=1), torch.ones(10, dtype=torch.float64))
    assert (meanings.argmax(dim=1) == torch.arange(10)).all()
    wider = naming.meanings(POINTS, sigma=3.0)
    assert wider[0, 0] < meanings[0, 0]


def test_complexity_is_the_bits_a_system_spends_on_its_items():
    assert naming.ib_complexity(torch.eye(10, dtype=torch.float64), PRIOR) == pytest.approx(math.log2(10))
    assert naming.ib_complexity(torch.ones(10, 1, dtype=torch.float64), PRIOR) == pytest.approx(0.0, abs=1e-12)
    halves = one_hot(["a"] * 5 + ["b"] * 5)
    assert naming.ib_complexity(halves, PRIOR) == pytest.approx(1.0)


def test_accuracy_runs_from_zero_to_what_the_items_carry(space):
    meanings, _ = space
    assert naming.ib_accuracy(torch.ones(10, 1, dtype=torch.float64), PRIOR, meanings) == pytest.approx(0.0, abs=1e-12)
    joint = (PRIOR[:, None] * meanings).log()
    ceiling = cg.metrics.mutual_information(joint).item()
    assert naming.ib_accuracy(torch.eye(10, dtype=torch.float64), PRIOR, meanings) == pytest.approx(ceiling)


def test_the_frontier_rises_with_beta(space):
    _, frontier = space
    assert (frontier.betas.diff() > 0).all()
    assert (frontier.complexity.diff() > -1e-6).all()
    assert (frontier.accuracy.diff() > -1e-6).all()
    assert len(frontier.encoders) == len(frontier.betas)


def test_no_system_beats_the_frontier(space):
    meanings, frontier = space
    generator = torch.Generator().manual_seed(0)
    for _ in range(20):
        labels = torch.randint(3, (10,), generator=generator).tolist()
        epsilon, _ = naming.efficiency_loss(one_hot(labels), PRIOR, meanings, frontier)
        assert epsilon > -1e-6


def test_a_system_on_the_frontier_loses_nothing_and_a_scrambled_one_does(space):
    meanings, frontier = space
    optimal = frontier.encoders[len(frontier.betas) // 2]
    epsilon, _ = naming.efficiency_loss(optimal, PRIOR, meanings, frontier)
    assert epsilon == pytest.approx(0.0, abs=1e-6)
    contiguous = one_hot(["a"] * 5 + ["b"] * 5)
    scrambled = one_hot(["a", "b"] * 5)
    assert (
        naming.efficiency_loss(scrambled, PRIOR, meanings, frontier)[0]
        > naming.efficiency_loss(contiguous, PRIOR, meanings, frontier)[0]
    )


def test_the_fitted_beta_picks_the_nearest_optimal_system(space):
    meanings, frontier = space
    contiguous = one_hot(["a"] * 5 + ["b"] * 5)
    _, beta = naming.efficiency_loss(contiguous, PRIOR, meanings, frontier)
    assert naming.gnid(contiguous, frontier.encoder_at(beta), PRIOR) < 0.5


def test_gnid_ignores_label_names_and_measures_disagreement():
    system = one_hot(["a"] * 5 + ["b"] * 5)
    renamed = one_hot(["z"] * 5 + ["y"] * 5)
    assert naming.gnid(system, renamed, PRIOR) == pytest.approx(0.0, abs=1e-12)
    assert naming.gnid(system, torch.ones(10, 1, dtype=torch.float64), PRIOR) == pytest.approx(1.0)
    other = one_hot(["a", "b"] * 5)
    assert naming.gnid(system, other, PRIOR) == pytest.approx(naming.gnid(other, system, PRIOR))
    assert naming.gnid(torch.ones(10, 1), torch.ones(10, 1), PRIOR) == 0.0


def test_co_naming_mse_compares_which_items_share_a_label():
    system = one_hot(["a"] * 5 + ["b"] * 5)
    assert naming.co_naming_mse(system, one_hot(["y"] * 5 + ["z"] * 5)) == pytest.approx(0.0)
    single = torch.ones(10, 1, dtype=torch.float64)
    # 40 of the 90 ordered pairs of different items share a label in the halves
    assert naming.co_naming_mse(system, single) == pytest.approx(50 / 90)


def run_mapping(tmp_path, task, **settings):
    paradigm = cg.paradigms.IteratedLearning(task, generations=2, chains=3, n_prev_examples=4, **settings)
    return cg.run(paradigm, cg.subjects.ExemplarLearner(), out=tmp_path / "run", progress=False)


def test_results_report_the_bottleneck_measures_by_generation(tmp_path):
    task = cg.tasks.Mapping([[value] for value in range(8)], ("wug", "dax", "fep"))
    results = run_mapping(tmp_path, task)
    table = results.metrics("ib_complexity", "ib_accuracy", "efficiency_loss", "ib_alignment")
    assert set(table["metric"]) == {"ib_complexity", "ib_accuracy", "efficiency_loss", "ib_alignment"}
    assert sorted(table["generation"].unique()) == [0, 1]
    assert table["value"].notna().all()
    assert (table.loc[table["metric"] == "ib_alignment", "value"].between(0, 1)).all()


def test_results_compare_each_chain_with_a_reference_labelling(tmp_path):
    start = ["wug"] * 4 + ["dax"] * 4
    task = cg.tasks.Mapping([[value] for value in range(8)], ("wug", "dax"), start=start)
    results = run_mapping(tmp_path, task)
    renamed = ["warm"] * 4 + ["cool"] * 4
    table = results.metrics("alignment", "co_naming_mse", reference=renamed, by=["generation", "chain"])
    assert len(table) == 2 * 2 * 3
    assert set(table.columns) >= {"generation", "chain", "metric", "value"}


def test_results_score_a_function_against_a_reference(tmp_path):
    task = cg.tasks.Mapping.from_function(range(1, 11), lambda x: x, scale=range(1, 11))
    results = run_mapping(tmp_path, task)
    table = results.metrics("mse", "correlation", reference=lambda x: x)
    first = table[table["generation"] == 0].set_index("metric")["value"]
    assert first["correlation"] > 0.5
    assert first["mse"] >= 0
    listed = results.metrics("mse", reference=list(range(1, 11)))
    assert listed["value"].tolist() == table.loc[table["metric"] == "mse", "value"].tolist()


def test_bottleneck_measures_need_numeric_items(tmp_path):
    task = cg.tasks.Mapping([["red", "big"], ["blue", "small"], ["red", "small"]], ("wug", "dax"))
    paradigm = cg.paradigms.IteratedLearning(task, generations=1, chains=1, n_prev_examples=2)
    results = cg.run(paradigm, ScriptedSubject(), out=tmp_path / "run", progress=False)
    with pytest.raises(MissingAbilityError, match="numbers"):
        results.metrics("ib_complexity")
