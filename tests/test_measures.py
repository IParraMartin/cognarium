from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest
import torch
from safetensors.torch import save_file

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
