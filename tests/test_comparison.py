"""Comparing a model with people: item correlations, noise ceilings and effect sizes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from helpers import ScriptedSubject
from scipy import stats

import cognarium as cg
from cognarium.metrics import effect_comparison, item_correlation, split_half_ceiling


def lexical_run(folder):
    return cg.run(
        cg.paradigms.LexicalDecision(),
        ScriptedSubject(probabilities=(0.7, 0.3)),
        out=folder,
        progress=False
    )


def human_table(
    results,
    value=None,
    drop=0,
    extra=()
):
    items = sorted(results.to_pandas()["item"].unique())[drop:]
    values = np.arange(len(items), dtype=float) if value is None else value(items)
    return pd.DataFrame({
        "item": [*items, *extra],
        "value": [*values, *range(len(extra))]
    })


def test_item_correlation_matches_scipy():
    rng = np.random.default_rng(0)
    model = rng.normal(size=30)
    human = model + rng.normal(size=30)
    pearson = item_correlation(model, human)
    spearman = item_correlation(model, human, method="spearman")
    assert pearson["correlation"] == pytest.approx(stats.pearsonr(model, human).statistic)
    assert spearman["correlation"] == pytest.approx(stats.spearmanr(model, human).statistic)
    assert pearson["n"] == 30


def test_item_correlation_interval_covers_the_estimate():
    rng = np.random.default_rng(1)
    model = rng.normal(size=40)
    found = item_correlation(model, model + rng.normal(size=40), n_resamples=500)
    assert found["ci_low"] < found["correlation"] < found["ci_high"]


def test_item_correlation_is_seeded():
    rng = np.random.default_rng(1)
    model = rng.normal(size=40)
    human = model + rng.normal(size=40)
    assert item_correlation(model, human, seed=3) == item_correlation(model, human, seed=3)
    assert item_correlation(model, human, seed=3)["ci_low"] != item_correlation(model, human, seed=4)["ci_low"]


def test_item_correlation_refuses_unknown_methods_and_mismatched_lengths():
    with pytest.raises(ValueError, match="method"):
        item_correlation([1, 2, 3], [1, 2, 4], method="kendall")
    with pytest.raises(ValueError, match="same number"):
        item_correlation([1, 2, 3], [1, 2])


def test_split_half_hits_its_limits():
    rng = np.random.default_rng(2)
    pattern = rng.normal(size=50)
    identical = np.tile(pattern, (10, 1))
    assert split_half_ceiling(identical, n_splits=50)["ceiling"] == pytest.approx(1.0)
    noise = rng.normal(size=(40, 300))
    assert abs(split_half_ceiling(noise, n_splits=200)["ceiling"]) < 0.15


def test_split_half_matches_a_hand_value_with_one_split_of_four():
    ratings = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [2.0, 1.0, 4.0, 3.0, 6.0],
        [1.0, 3.0, 2.0, 5.0, 4.0],
        [3.0, 2.0, 1.0, 4.0, 6.0]
    ])
    # the documented split: participants shuffled by a generator seeded with seed, then halved
    order = np.random.default_rng([7, 0]).permutation(4)
    first, second = ratings[order[:2]].mean(axis=0), ratings[order[2:]].mean(axis=0)
    r = np.corrcoef(first, second)[0, 1]
    found = split_half_ceiling(ratings, n_splits=1, seed=7)
    assert found["ceiling"] == pytest.approx(2 * r / (1 + r))
    assert found["ci_low"] == found["ci_high"] == pytest.approx(found["ceiling"])


def test_split_half_handles_missing_ratings():
    rng = np.random.default_rng(3)
    pattern = rng.normal(size=30)
    ratings = pattern + 0.5 * rng.normal(size=(12, 30))
    ratings[rng.random(ratings.shape) < 0.2] = np.nan
    found = split_half_ceiling(ratings, n_splits=100)
    assert 0.5 < found["ceiling"] <= 1
    assert found["ci_low"] <= found["ceiling"] <= found["ci_high"]


def test_split_half_needs_two_participants():
    with pytest.raises(ValueError, match="participants"):
        split_half_ceiling(np.ones((1, 5)))


def test_effect_comparison_is_near_zero_for_the_same_effect():
    rng = np.random.default_rng(4)
    a, b = rng.normal(1.0, 1.0, size=40), rng.normal(0.0, 1.0, size=40)
    found = effect_comparison(
        a,
        b,
        a,
        b
    )
    assert found["difference"] == pytest.approx(0.0)
    assert found["model_d"] == pytest.approx(found["human_d"])
    assert found["ci_low"] < 0 < found["ci_high"]


def test_effect_comparison_sees_a_real_gap():
    rng = np.random.default_rng(5)
    model_a, model_b = rng.normal(2.0, 1.0, size=60), rng.normal(0.0, 1.0, size=60)
    human_a, human_b = rng.normal(0.2, 1.0, size=60), rng.normal(0.0, 1.0, size=60)
    found = effect_comparison(
        model_a,
        model_b,
        human_a,
        human_b
    )
    assert found["difference"] > 1
    assert found["ci_low"] > 0


def test_effect_comparison_paired_uses_the_differences():
    rng = np.random.default_rng(6)
    a = rng.normal(size=30)
    b = a - 1 + 0.5 * rng.normal(size=30)
    found = effect_comparison(
        a,
        b,
        a,
        b,
        paired=True
    )
    differences = a - b
    assert found["model_d"] == pytest.approx(differences.mean() / differences.std(ddof=1))


def test_compare_to_human_joins_on_item_and_warns_about_dropped_items(tmp_path):
    results = lexical_run(tmp_path / "run")
    human = human_table(results, drop=2, extra=("zzz", "yyy", "xxx"))
    with pytest.warns(UserWarning, match="2 items .* only in the results.*3 .* only in the human"):
        table = results.compare_to_human(human, "answer_probability")
    assert len(table) == 1
    assert table.iloc[0]["n"] == results.to_pandas()["item"].nunique() - 2


def test_compare_to_human_correlates_the_item_means(tmp_path):
    results = lexical_run(tmp_path / "run")
    frame = results.to_pandas()
    model = frame.assign(value=np.exp([row[0] for row in frame["log_probs"]])).groupby("item")["value"].mean()
    human = pd.DataFrame({"item": model.index, "value": model.to_numpy() * 3 + np.arange(len(model))})
    table = results.compare_to_human(human, "option_probability", option="word")
    assert table.iloc[0]["correlation"] == pytest.approx(np.corrcoef(model.to_numpy(), human["value"])[0, 1])


def test_compare_to_human_labels_proxies(tmp_path):
    results = lexical_run(tmp_path / "run")
    human = human_table(results)
    surprisal = results.compare_to_human(human, "answer_surprisal", ceiling=0.8).iloc[0]
    accuracy = results.compare_to_human(human, "answer_probability").iloc[0]
    assert surprisal["proxy_for"] == "reaction time"
    assert surprisal["ceiling"] == 0.8
    assert accuracy["proxy_for"] is None
    assert list(results.compare_to_human(human, "answer_probability").columns) == [
        "measure",
        "correlation",
        "ci_low",
        "ci_high",
        "n",
        "ceiling",
        "proxy_for"
    ]


def test_constant_input_raises_rather_than_nan():
    with pytest.raises(ValueError, match="constant"):
        item_correlation([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="constant"):
        split_half_ceiling(np.ones((4, 6)))
    with pytest.raises(ValueError, match="variance"):
        effect_comparison(
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [2.0, 3.0]
        )


def test_too_few_shared_items_raises(tmp_path):
    results = lexical_run(tmp_path / "run")
    items = sorted(results.to_pandas()["item"].unique())[:2]
    human = pd.DataFrame({"item": items, "value": [1.0, 2.0]})
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="at least 3"):
        results.compare_to_human(human, "answer_probability")


def test_compare_to_human_needs_the_columns(tmp_path):
    results = lexical_run(tmp_path / "run")
    with pytest.raises(ValueError, match="'value'"):
        results.compare_to_human(pd.DataFrame({"item": ["a"], "rating": [1.0]}), "answer_probability")
