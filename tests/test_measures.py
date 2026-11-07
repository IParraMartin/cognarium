from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest
import torch
from safetensors.torch import save_file

from cognarium import metrics
from cognarium.errors import MissingAbilityError
from cognarium.results import Results, logits_path, step_path


def chain_frame():
    half = math.log(0.5)
    rows = []
    for generation, chain, group in [
        (0, 0, "I"),
        (0, 1, "I"),
        (1, 0, "I"),
        (1, 1, "VI")
    ]:
        rows.append({
            "generation": generation,
            "chain": chain,
            "group": group,
            "options": ["A", "B"],
            "response": "A",
            "answer": "A" if chain == 0 else "B",
            "parent_label": "A",
            "log_probs": [half, half],
            "shaped_log_probs": [0.0, -math.inf]
        })
    return pd.DataFrame(rows)


MANIFEST = {
    "group_by": ["generation"],
    "groups": ["I", "VI"],
    "reference": None
}


def results(manifest=None, frame=None):
    return Results(Path("."), chain_frame() if frame is None else frame, manifest or MANIFEST)


def values(table, metric):
    return table[table["metric"] == metric].set_index("generation")["value"].to_dict()


def test_accuracy_and_fidelity():
    table = results().metrics("accuracy", "fidelity")
    assert values(table, "accuracy") == {0: 0.5, 1: 0.5}
    assert values(table, "fidelity") == {0: 1.0, 1: 1.0}


def test_rows_without_an_answer_do_not_count_towards_accuracy():
    frame = chain_frame()
    frame.loc[frame["chain"] == 1, "answer"] = None
    assert values(results(frame=frame).metrics("accuracy"), "accuracy") == {0: 1.0, 1: 1.0}


def test_entropy_uses_shaped_by_default_and_raw_on_request():
    assert values(results().metrics("entropy"), "entropy") == {0: 0.0, 1: 0.0}
    assert values(results().metrics("entropy", raw=True), "entropy") == pytest.approx({0: 1.0, 1: 1.0})


def test_surprisal_stays_finite_on_a_fixated_chain():
    assert values(results().metrics("surprisal"), "surprisal") == {0: 0.0, 1: 0.0}


def test_surprisal_uses_raw_on_request():
    assert values(results().metrics("surprisal", raw=True), "surprisal") == pytest.approx({0: 1.0, 1: 1.0})


def test_chain_entropy():
    assert values(results().metrics("chain_entropy"), "chain_entropy") == {0: 0.0, 1: 1.0}


def test_kl_prefers_passed_reference_then_prior_then_final_third():
    passed = results().metrics("kl", reference={"I": 0.5, "VI": 0.5}, smoothing=0.0)
    assert set(passed["reference"]) == {"passed"}
    assert values(passed, "kl") == {0: pytest.approx(1.0), 1: pytest.approx(0.0)}
    prior = results({**MANIFEST, "reference": {"I": 0.5, "VI": 0.5}}).metrics("kl")
    assert set(prior["reference"]) == {"subject_prior"}
    assert set(results().metrics("kl")["reference"]) == {"final_third"}


def test_kl_final_third_pools_the_last_generations():
    table = results().metrics("kl", smoothing=0.0)
    assert values(table, "kl")[1] == pytest.approx(0.0)
    assert values(table, "kl")[0] == pytest.approx(1.0)


def test_kl_reports_infinity_outside_the_reference_support():
    table = results().metrics("kl", reference={"I": 1.0}, smoothing=0.0)
    assert values(table, "kl")[1] == math.inf


def test_unknown_group_in_reference_raises():
    with pytest.raises(ValueError, match="nope"):
        results().metrics("kl", reference={"nope": 1.0})


def test_unknown_metric_and_option_raise():
    with pytest.raises(ValueError, match="accuracy"):
        results().metrics("nope")
    with pytest.raises(TypeError, match="nope"):
        results().metrics("accuracy", nope=1)


def test_missing_columns_name_the_measure():
    frame = chain_frame().drop(columns=["group"])
    with pytest.raises(MissingAbilityError, match="chain_entropy"):
        results(frame=frame).metrics("chain_entropy")


def test_table_has_a_stable_shape():
    table = results().metrics("accuracy", "kl")
    assert list(table.columns) == ["generation", "metric", "value", "reference"]


def test_to_pandas_returns_a_copy():
    made = results()
    copied = made.to_pandas()
    copied.loc[0, "response"] = "Z"
    assert made.to_pandas().loc[0, "response"] == "A"


def test_asking_for_no_metrics_lists_the_known_ones():
    with pytest.raises(ValueError, match="accuracy"):
        results().metrics()


def test_frame_and_manifest_are_not_options():
    with pytest.raises(TypeError, match="No requested metric accepts"):
        results().metrics("accuracy", manifest={})


def write_run(root, frames):
    (root / "steps").mkdir()
    (root / "manifest.json").write_text(json.dumps(MANIFEST))
    for index, frame in enumerate(frames):
        frame.to_parquet(step_path(root, index))


def test_load_reads_the_manifest_and_every_step(tmp_path):
    write_run(tmp_path, [chain_frame(), chain_frame()])
    loaded = Results.load(tmp_path)
    assert loaded.manifest == MANIFEST
    assert len(loaded.to_pandas()) == 8
    assert values(loaded.metrics("accuracy"), "accuracy") == {0: 0.5, 1: 0.5}


def test_load_gives_an_empty_table_before_any_step_finishes(tmp_path):
    write_run(tmp_path, [])
    assert Results.load(tmp_path).to_pandas().empty


def test_logits_not_recorded_raises(tmp_path):
    with pytest.raises(MissingAbilityError, match="step 3"):
        Results(tmp_path, chain_frame(), MANIFEST).logits(3)


def test_logits_are_read_back(tmp_path):
    expected = torch.arange(6.0).reshape(2, 3)
    logits_path(tmp_path, 1).parent.mkdir()
    save_file({"logits": expected}, logits_path(tmp_path, 1))
    assert torch.equal(Results(tmp_path, chain_frame(), MANIFEST).logits(1), expected)


def log_pairs(first_option):
    return [[math.log(p), math.log(1 - p)] for p in first_option]


def scored_frame():
    return pd.DataFrame(
        {
            "generation": 0,
            "options": [["A", "B"] for _ in range(6)],
            "answer": ["A", "A", "A", "B", "B", "B"],
            "response": ["A", "B", "A", "B", "B", "A"],
            "log_probs": log_pairs([0.9, 0.4, 0.7, 0.2, 0.3, 0.6]),
            "shaped_log_probs": log_pairs([0.99, 0.1, 0.8, 0.3, 0.2, 0.7])
        }
    )


def predictions():
    return torch.tensor([0, 1, 0, 1, 1, 0])


def targets():
    return torch.tensor([0, 0, 0, 1, 1, 1])


def probabilities(column):
    return torch.tensor(scored_frame()[column].tolist(), dtype=torch.float64).exp()


@pytest.mark.parametrize("name", ["balanced_accuracy", "precision", "recall", "f1", "mcc"])
def test_label_measures_use_response_and_answer(name):
    table = results(frame=scored_frame()).metrics(name)
    expected = getattr(metrics, name)(predictions(), targets())
    assert values(table, name)[0] == pytest.approx(expected)


@pytest.mark.parametrize("name", ["precision", "recall", "f1"])
def test_positive_names_an_option(name):
    table = results(frame=scored_frame()).metrics(name, positive="B")
    expected = getattr(metrics, name)(predictions(), targets(), positive=1)
    assert values(table, name)[0] == pytest.approx(expected)


def test_label_measures_skip_rows_without_an_answer():
    frame = pd.concat([scored_frame(), scored_frame().head(1).assign(answer=None, response="B")], ignore_index=True)
    table = results(frame=frame).metrics("balanced_accuracy")
    assert values(table, "balanced_accuracy")[0] == pytest.approx(metrics.balanced_accuracy(predictions(), targets()))


@pytest.mark.parametrize("name", ["auroc", "auprc", "brier", "ece"])
def test_probability_measures_use_shaped_unless_raw(name):
    plain = getattr(metrics, name)
    shaped = values(results(frame=scored_frame()).metrics(name), name)[0]
    raw = values(results(frame=scored_frame()).metrics(name, raw=True), name)[0]
    assert shaped == pytest.approx(plain(probabilities("shaped_log_probs"), targets()))
    assert raw == pytest.approx(plain(probabilities("log_probs"), targets()))
    assert shaped != pytest.approx(raw)


def test_ece_takes_its_bin_count():
    table = results(frame=scored_frame()).metrics("ece", n_bins=2)
    expected = metrics.ece(probabilities("shaped_log_probs"), targets(), n_bins=2)
    assert values(table, "ece")[0] == pytest.approx(expected)


def test_probability_measures_accept_a_rounded_up_probability():
    frame = scored_frame()
    frame["shaped_log_probs"] = [[1e-16, -math.inf]] * 3 + [[-math.inf, 1e-16]] * 3
    assert values(results(frame=frame).metrics("auroc"), "auroc")[0] == pytest.approx(1.0)


def test_cross_entropy_and_perplexity():
    table = results(frame=scored_frame()).metrics("cross_entropy", "perplexity", raw=True)
    answer_probabilities = torch.tensor([0.9, 0.4, 0.7, 0.8, 0.7, 0.4], dtype=torch.float64)
    expected = -answer_probabilities.log2().mean().item()
    assert values(table, "cross_entropy")[0] == pytest.approx(expected)
    assert values(table, "perplexity")[0] == pytest.approx(2**expected)


def test_cross_entropy_is_infinite_for_an_impossible_answer():
    frame = scored_frame()
    frame["shaped_log_probs"] = [[-math.inf, 0.0]] * 6
    table = results(frame=frame).metrics("cross_entropy", "perplexity")
    assert values(table, "cross_entropy")[0] == math.inf
    assert values(table, "perplexity")[0] == math.inf


def test_detection_measures_use_the_first_option_by_default():
    frame = scored_frame()
    assert values(results(frame=frame).metrics("d_prime"), "d_prime")[0] == pytest.approx(metrics.d_prime(
        2,
        1,
        1,
        2
    ))
    for name in ["d_prime", "criterion", "a_prime"]:
        default = values(results(frame=frame).metrics(name), name)[0]
        assert default == pytest.approx(values(results(frame=frame).metrics(name, positive="A"), name)[0])


def test_swapping_the_positive_option_flips_only_the_bias():
    frame = pd.concat([scored_frame(), scored_frame().head(3).assign(response="A")], ignore_index=True)
    measured = {
        (name, positive): values(results(frame=frame).metrics(name, positive=positive), name)[0]
        for name in ["d_prime", "criterion", "a_prime"]
        for positive in ["A", "B"]
    }
    assert measured["criterion", "A"] != pytest.approx(0.0)
    assert measured["criterion", "B"] == pytest.approx(-measured["criterion", "A"])
    assert measured["d_prime", "B"] == pytest.approx(measured["d_prime", "A"])
    assert measured["a_prime", "B"] == pytest.approx(measured["a_prime", "A"])


@pytest.mark.parametrize("name", ["precision", "d_prime"])
def test_unknown_positive_option_raises(name):
    with pytest.raises(ValueError, match="'C'"):
        results(frame=scored_frame()).metrics(name, positive="C")


@pytest.mark.parametrize("name", ["logit_margin", "logit_difference", "target_rank", "vocab_entropy"])
def test_logit_measures_average_their_column_and_drop_empty_rows(name):
    frame = scored_frame().assign(**{name: [1.0, 2.0, 3.0, None, None, 6.0]})
    assert values(results(frame=frame).metrics(name), name)[0] == pytest.approx(3.0)


def test_logit_measures_need_recorded_logits():
    with pytest.raises(MissingAbilityError, match="logit_margin"):
        results().metrics("logit_margin")


def test_logit_measure_with_no_values_raises():
    frame = scored_frame().assign(target_rank=None)
    with pytest.raises(MissingAbilityError, match="target_rank"):
        results(frame=frame).metrics("target_rank")


def test_mixed_option_lists_raise():
    frame = scored_frame()
    frame["options"] = [["A", "C"]] + [["A", "B"] for _ in range(5)]
    for name in ["f1", "auroc", "cross_entropy", "d_prime"]:
        with pytest.raises(ValueError, match=name):
            results(frame=frame).metrics(name)


def test_answer_outside_the_options_raises():
    frame = scored_frame().assign(answer=["C", "A", "A", "B", "B", "B"])
    with pytest.raises(ValueError, match="'C'"):
        results(frame=frame).metrics("f1")


def test_one_class_group_names_the_group():
    frame = pd.concat([scored_frame(), scored_frame().assign(generation=1, answer="A")], ignore_index=True)
    with pytest.raises(ValueError, match="generation=1"):
        results(frame=frame).metrics("auroc")


@pytest.mark.parametrize("name", ["f1", "auroc", "cross_entropy", "d_prime"])
def test_measures_need_an_answer(name):
    with pytest.raises(MissingAbilityError, match=name):
        results(frame=scored_frame().assign(answer=None)).metrics(name)


def test_measures_read_the_arrays_parquet_gives_back(tmp_path):
    scored_frame().to_parquet(tmp_path / "frame.parquet")
    frame = pd.read_parquet(tmp_path / "frame.parquet")
    table = results(frame=frame).metrics(
        "f1",
        "auroc",
        "cross_entropy",
        "d_prime"
    )
    assert table["value"].notna().all()
