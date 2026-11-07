from __future__ import annotations

import json
import threading
import time

import numpy as np
import pandas as pd
import pytest
import torch

from cognarium.errors import MissingAbilityError, RunConflictError
from cognarium.metrics.geometry import linear_cka
from cognarium.recording import store
from cognarium.recording.store import HiddenStates, StateWriter


def fill(rows, width=4, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, width, generator=generator)


def index_columns(index, rows):
    return {
        "step": 0,
        "query": list(range(rows)),
        "state_shard": [index[query][0] for query in range(rows)],
        "state_offset": [index[query][1] for query in range(rows)]
    }


def describe(root, index, rows, *, layers, pool, dtype):
    root.mkdir(parents=True, exist_ok=True)
    (root / "steps").mkdir(exist_ok=True)
    frame = pd.DataFrame(index_columns(index, rows))
    frame.to_parquet(root / "steps" / "step-00000.parquet")
    manifest = {
        "states": {
            "layers": list(layers),
            "pool": pool,
            "dtype": dtype
        },
        "paradigm": {"name": "test"},
        "subject": {"class": "Test"},
        "learning": None,
        "versions": {"cognarium": "test"},
        "sessions": []
    }
    (root / "manifest.json").write_text(json.dumps(manifest))


def written(
    root,
    *,
    layers=("a", "b"),
    dtype="float32",
    rows=6,
    rows_per_shard=4096
):
    with StateWriter(root, 0, dtype=dtype, rows_per_shard=rows_per_shard) as writer:
        for seed, layer in enumerate(layers):
            writer.write(layer, fill(rows, seed=seed), range(rows))
        index = writer.finish(rows)
    describe(
        root,
        index,
        rows,
        layers=layers,
        pool="last",
        dtype=dtype
    )
    return HiddenStates.open(root)


def within(seconds, function):
    """Runs a function on another thread, and fails if it hasn't returned in time."""
    outcome = {}

    def target():
        try:
            outcome["value"] = function()
        except BaseException as error:
            outcome["error"] = error

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(seconds)
    assert not thread.is_alive(), "this hung"
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")


def test_round_trip_restores_query_order_whatever_order_writes_arrive(tmp_path):
    rows = fill(3)
    with StateWriter(tmp_path, 0, dtype="float32") as writer:
        writer.write("a", rows[[2, 0]], [2, 0])
        writer.write("a", rows[[1]], [1])
        index = writer.finish(3)
    describe(
        tmp_path,
        index,
        3,
        layers=["a"],
        pool="last",
        dtype="float32"
    )
    assert torch.equal(HiddenStates.open(tmp_path).to_tensor("a"), rows)


def test_shards_roll_over_by_row_count(tmp_path):
    states = written(tmp_path, layers=("a",), rows=7, rows_per_shard=3)
    assert len(list((tmp_path / "states" / "a").glob("*.safetensors"))) == 3
    assert torch.equal(states.to_tensor("a"), fill(7))


def test_shard_names_follow_the_layout(tmp_path):
    written(tmp_path, layers=("a",), rows=4, rows_per_shard=3)
    names = sorted(path.name for path in (tmp_path / "states" / "a").iterdir())
    assert names == ["step-00000-00000.safetensors", "step-00000-00001.safetensors"]


def test_a_rerun_overwrites_the_same_shards(tmp_path):
    written(tmp_path, layers=("a",), rows=4, rows_per_shard=3)
    first = sorted((tmp_path / "states" / "a").iterdir())
    written(tmp_path, layers=("a",), rows=4, rows_per_shard=3)
    assert sorted((tmp_path / "states" / "a").iterdir()) == first


def test_float16_overflow_names_the_layer(tmp_path):
    with StateWriter(tmp_path, 0, dtype="float16") as writer:
        with pytest.raises(ValueError, match="layer_03.*float32"):
            writer.write("layer_03", torch.full((1, 2), 1e6), [0])


def test_non_finite_activations_are_not_blamed_on_the_dtype(tmp_path):
    with StateWriter(tmp_path, 0, dtype="float16") as writer:
        with pytest.raises(ValueError, match="layer_03") as raised:
            writer.write("layer_03", torch.tensor([[float("nan"), 1.0]]), [0])
    assert "float32" not in str(raised.value)


def test_unknown_dtype_lists_the_allowed_ones(tmp_path):
    with pytest.raises(ValueError, match="bfloat16"):
        StateWriter(tmp_path, 0, dtype="int8")


def test_a_failing_disk_surfaces_and_does_not_hang(tmp_path, monkeypatch):
    def full(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(store, "save_file", full)
    made = []

    def attempt():
        with StateWriter(tmp_path, 0, dtype="float32", rows_per_shard=1, max_pending=1) as writer:
            made.append(writer)
            for query in range(50):
                writer.write("a", fill(1), [query])
            writer.finish(50)

    with pytest.raises(OSError, match="disk full"):
        within(10, attempt)
    assert not made[0]._thread.is_alive()
    assert not list(tmp_path.rglob("*.safetensors"))


def test_exiting_on_an_exception_stops_the_thread(tmp_path, monkeypatch):
    real = store.save_file

    def slow(*args, **kwargs):
        time.sleep(0.05)
        real(*args, **kwargs)

    monkeypatch.setattr(store, "save_file", slow)
    made = []

    def attempt():
        with StateWriter(tmp_path, 0, dtype="float32", rows_per_shard=1, max_pending=1) as writer:
            made.append(writer)
            for query in range(3):
                writer.write("a", fill(1), [query])
            raise KeyError("stopped")

    with pytest.raises(KeyError, match="stopped"):
        within(10, attempt)
    assert not made[0]._thread.is_alive()


def test_the_queue_stays_bounded(tmp_path, monkeypatch):
    real = store.save_file
    sizes = []
    made = []

    def slow(*args, **kwargs):
        sizes.append(made[0]._queue.qsize())
        time.sleep(0.005)
        real(*args, **kwargs)

    monkeypatch.setattr(store, "save_file", slow)
    with StateWriter(tmp_path, 0, dtype="float32", rows_per_shard=1, max_pending=2) as writer:
        made.append(writer)
        for query in range(20):
            writer.write("a", fill(1), [query])
        writer.finish(20)
    assert len(sizes) == 20
    assert max(sizes) <= 2


def test_finish_reports_missing_queries(tmp_path):
    with StateWriter(tmp_path, 0, dtype="float32") as writer:
        writer.write("a", torch.zeros(1, 2), [0])
        with pytest.raises(RuntimeError, match="1"):
            writer.finish(2)


def test_finish_with_nothing_written_raises(tmp_path):
    with StateWriter(tmp_path, 0, dtype="float32") as writer:
        with pytest.raises(RuntimeError, match="No states"):
            writer.finish(2)


def test_finish_rejects_a_query_written_twice(tmp_path):
    with StateWriter(tmp_path, 0, dtype="float32") as writer:
        writer.write("a", torch.zeros(2, 2), [0, 0])
        with pytest.raises(RuntimeError, match="'a'"):
            writer.finish(2)


def test_layers_must_agree_on_their_queries(tmp_path):
    with StateWriter(tmp_path, 0, dtype="float32") as writer:
        writer.write("a", torch.zeros(2, 2), [0, 1])
        writer.write("b", torch.zeros(1, 2), [0])
        with pytest.raises(RuntimeError, match="'b'"):
            writer.finish(2)


def test_layers_must_agree_on_their_order(tmp_path):
    with StateWriter(tmp_path, 0, dtype="float32") as writer:
        writer.write("a", torch.zeros(2, 2), [0, 1])
        writer.write("b", torch.zeros(2, 2), [1, 0])
        with pytest.raises(RuntimeError, match="'b'"):
            writer.finish(2)


def test_width_changes_raise(tmp_path):
    with StateWriter(tmp_path, 0, dtype="float32") as writer:
        writer.write("a", torch.zeros(1, 2), [0])
        with pytest.raises(ValueError, match="'a'"):
            writer.write("a", torch.zeros(1, 3), [1])


def test_zero_width_states_explain(tmp_path):
    with StateWriter(tmp_path, 0, dtype="float32") as writer:
        with pytest.raises(ValueError, match="similarity"):
            writer.write("similarity", torch.empty(2, 0), [0, 1])


def test_rows_and_queries_must_match(tmp_path):
    with StateWriter(tmp_path, 0, dtype="float32") as writer:
        with pytest.raises(ValueError, match="queries"):
            writer.write("a", torch.zeros(2, 2), [0])


def test_writing_after_finish_raises(tmp_path):
    with StateWriter(tmp_path, 0, dtype="float32") as writer:
        writer.write("a", torch.zeros(1, 2), [0])
        writer.finish(1)
        with pytest.raises(RuntimeError, match="finish"):
            writer.write("a", torch.zeros(1, 2), [1])


def test_writer_names_its_layers(tmp_path):
    with StateWriter(tmp_path, 0, dtype="float32") as writer:
        writer.write("a", torch.zeros(1, 2), [0])
        assert writer.layers == ("a",)
        writer.finish(1)


def test_states_from_inference_mode_can_be_written(tmp_path):
    with torch.inference_mode():
        rows = fill(3)
    with StateWriter(tmp_path, 0, dtype="float16", rows_per_shard=2) as writer:
        writer.write("a", rows, range(3))
        index = writer.finish(3)
    assert index[2] == ("step-00000-00001", 0)


def test_no_pool_round_trips_through_sequences(tmp_path):
    first = torch.arange(8.0).reshape(2, 2, 2)
    second = torch.arange(100.0, 106.0).reshape(1, 3, 2)
    with StateWriter(tmp_path, 0, dtype="float32", rows_per_shard=2) as writer:
        writer.write("a", first, [0, 1], torch.tensor([1, 2]))
        writer.write("a", second, [2], torch.tensor([3]))
        index = writer.finish(3)
    describe(
        tmp_path,
        index,
        3,
        layers=["a"],
        pool=None,
        dtype="float32"
    )
    opened = HiddenStates.open(tmp_path)
    sequences = opened.sequences("a")
    assert [row.shape[0] for row in sequences] == [1, 2, 3]
    assert torch.equal(sequences[0], first[0, :1])
    assert torch.equal(sequences[1], first[1, :2])
    assert torch.equal(sequences[2], second[0])
    with pytest.raises(ValueError, match="sequences"):
        opened.to_tensor("a")


def test_no_pool_pads_rows_that_share_a_shard(tmp_path):
    first = torch.ones(1, 2, 2)
    second = 2 * torch.ones(1, 3, 2)
    with StateWriter(tmp_path, 0, dtype="float32") as writer:
        writer.write("a", first, [0], torch.tensor([2]))
        writer.write("a", second, [1], torch.tensor([3]))
        index = writer.finish(2)
    describe(
        tmp_path,
        index,
        2,
        layers=["a"],
        pool=None,
        dtype="float32"
    )
    sequences = HiddenStates.open(tmp_path).sequences("a")
    assert torch.equal(sequences[0], first[0])
    assert torch.equal(sequences[1], second[0])


def test_sequences_on_a_pooled_store_points_to_to_tensor(tmp_path):
    with pytest.raises(ValueError, match="to_tensor"):
        written(tmp_path).sequences("a")


def test_open_reads_no_shard_until_asked(tmp_path, monkeypatch):
    states = written(tmp_path)
    real = store.safe_open
    calls = []

    def counting(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(store, "safe_open", counting)
    opened = HiddenStates.open(tmp_path)
    assert len(opened.trials) == 6
    assert calls == []
    assert torch.equal(opened.to_tensor("a"), states.to_tensor("a"))
    assert calls


def test_where_selects_rows(tmp_path):
    states = written(tmp_path, layers=("a",), rows=6, rows_per_shard=4)
    mask = states.trials["query"] > 3
    assert torch.equal(states.to_tensor("a", where=mask), fill(6)[4:])
    assert torch.equal(states.to_tensor("a", where=mask.to_numpy()), fill(6)[4:])


@pytest.mark.parametrize("where", [[1, 2], np.array([True])])
def test_a_bad_where_explains(tmp_path, where):
    with pytest.raises(ValueError, match="where"):
        written(tmp_path).to_tensor("a", where=where)


def test_selecting_nothing_explains(tmp_path):
    states = written(tmp_path)
    with pytest.raises(ValueError, match="No trials"):
        states.to_tensor("a", where=np.zeros(6, dtype=bool))


def test_steps_of_different_widths_explain(tmp_path):
    states = written(tmp_path, layers=("a",), rows=2, rows_per_shard=1)
    with StateWriter(tmp_path, 0, dtype="float32", rows_per_shard=1) as writer:
        writer.write("a", fill(1, width=4), [0])
        writer.write("a", fill(1, width=4), [1])
        writer.finish(2)
    shard = tmp_path / "states" / "a" / "step-00000-00001.safetensors"
    with StateWriter(tmp_path / "other", 0, dtype="float32") as writer:
        writer.write("a", fill(1, width=5), [0])
        writer.finish(1)
    (tmp_path / "other" / "states" / "a" / "step-00000-00000.safetensors").replace(shard)
    with pytest.raises(ValueError, match="'a'"):
        states.to_tensor("a")


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        ("float16", np.float16),
        ("bfloat16", np.float32),
        ("float32", np.float32),
        ("float64", np.float64)
    ]
)
def test_to_numpy_handles_every_dtype(tmp_path, dtype, expected):
    states = written(tmp_path, dtype=dtype)
    assert states.dtype == dtype
    assert states.to_tensor("a").dtype == getattr(torch, dtype)
    assert states.to_numpy("a").dtype == expected


def test_save_then_open_gives_the_same_tensors_and_cka(tmp_path):
    states = written(tmp_path / "run")
    saved = states.save(tmp_path / "shared")
    reopened = HiddenStates.open(tmp_path / "shared")
    assert torch.equal(saved.to_tensor("a"), states.to_tensor("a"))
    assert torch.equal(reopened.to_tensor("a"), states.to_tensor("a"))
    assert linear_cka(reopened.to_tensor("a"), reopened.to_tensor("b")) == linear_cka(
        states.to_tensor("a"),
        states.to_tensor("b")
    )
    columns = set(pd.read_parquet(tmp_path / "shared" / "trials.parquet").columns)
    assert {"state_shard", "state_offset"} <= columns


def test_a_saved_store_describes_the_run(tmp_path):
    written(tmp_path / "run").save(tmp_path / "shared")
    manifest = json.loads((tmp_path / "shared" / "manifest.json").read_text())
    assert manifest["layers"] == ["a", "b"]
    assert manifest["pool"] == "last"
    assert manifest["dtype"] == "float32"
    assert manifest["paradigm"] == {"name": "test"}
    assert manifest["subject"] == {"class": "Test"}
    assert manifest["versions"] == {"cognarium": "test"}
    assert "sessions" not in manifest


def test_a_saved_store_can_be_saved_again(tmp_path):
    states = written(tmp_path / "run")
    again = states.save(tmp_path / "first").save(tmp_path / "second")
    assert torch.equal(again.to_tensor("b"), states.to_tensor("b"))


def test_save_into_an_empty_folder_works(tmp_path):
    (tmp_path / "shared").mkdir()
    saved = written(tmp_path / "run").save(tmp_path / "shared")
    assert saved.layers == ("a", "b")
    assert [path.name for path in tmp_path.iterdir() if path.name.startswith(".")] == []


def test_save_refuses_a_non_empty_folder(tmp_path):
    (tmp_path / "shared").mkdir()
    (tmp_path / "shared" / "keep.txt").write_text("keep")
    with pytest.raises(RunConflictError, match="shared"):
        written(tmp_path / "run").save(tmp_path / "shared")
    assert (tmp_path / "shared" / "keep.txt").read_text() == "keep"


def test_a_failed_save_leaves_nothing_behind(tmp_path, monkeypatch):
    states = written(tmp_path / "run")

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(store.shutil, "copy2", fail)
    with pytest.raises(OSError, match="disk full"):
        states.save(tmp_path / "shared")
    assert sorted(path.name for path in tmp_path.iterdir()) == ["run"]


def test_compare_gives_a_layer_matrix(tmp_path):
    states = written(tmp_path)
    table = states.compare()
    assert list(table.index) == ["a", "b"] == list(table.columns)
    assert table.loc["a", "a"] == pytest.approx(1.0)
    assert table.loc["a", "b"] == pytest.approx(linear_cka(states.to_tensor("a"), states.to_tensor("b")))


def test_compare_takes_layers_and_where(tmp_path):
    states = written(tmp_path)
    mask = states.trials["query"] < 4
    table = states.compare(lambda x, y: float(len(x)), layers=["b"], where=mask)
    assert table.loc["b", "b"] == 4.0
    assert list(table.index) == ["b"]


@pytest.mark.parametrize("method", ["to_tensor", "to_numpy", "sequences"])
def test_unknown_layer_lists_the_known_ones(tmp_path, method):
    with pytest.raises(ValueError, match="'a', 'b'"):
        getattr(written(tmp_path), method)("nope")


def test_a_run_without_states_explains(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"paradigm": {}}))
    with pytest.raises(MissingAbilityError, match="hidden_states"):
        HiddenStates.open(tmp_path)


class FakeApi:
    def __init__(self, token=None, public=False):
        self.token = token
        self.public = public
        self.created = None
        self.uploaded = None

    def create_repo(self, repo_id, **kwargs):
        self.created = {"repo_id": repo_id, **kwargs}

    def repo_info(self, repo_id, **kwargs):
        return type("Info", (), {"private": not self.public})

    def upload_folder(self, **kwargs):
        folder = store.Path(kwargs["folder_path"])
        self.uploaded = {
            **kwargs,
            "files": sorted(str(path.relative_to(folder)) for path in folder.rglob("*") if path.is_file())
        }


def fake_hub(monkeypatch, **kwargs):
    made = []

    def build(token=None):
        made.append(FakeApi(token, **kwargs))
        return made[-1]

    monkeypatch.setattr("huggingface_hub.HfApi", build)
    return made


def test_push_to_hub_uses_a_private_dataset_repo_by_default(tmp_path, monkeypatch):
    made = fake_hub(monkeypatch)
    written(tmp_path / "run").push_to_hub("someone/states", token="secret")
    api = made[0]
    assert api.token == "secret"
    assert api.created == {
        "repo_id": "someone/states",
        "repo_type": "dataset",
        "exist_ok": True,
        "private": True
    }
    assert api.uploaded["repo_id"] == "someone/states"
    assert api.uploaded["repo_type"] == "dataset"
    assert "trials.parquet" in api.uploaded["files"]
    assert "manifest.json" in api.uploaded["files"]
    assert not any(name.startswith("steps") for name in api.uploaded["files"])


def test_push_to_hub_can_be_public(tmp_path, monkeypatch):
    made = fake_hub(monkeypatch, public=True)
    written(tmp_path / "run").push_to_hub("someone/states", private=False)
    assert made[0].created["private"] is False


def test_pushing_a_saved_store_uploads_its_folder(tmp_path, monkeypatch):
    made = fake_hub(monkeypatch)
    saved = written(tmp_path / "run").save(tmp_path / "shared")
    saved.push_to_hub("someone/states")
    assert made[0].uploaded["folder_path"] == str(tmp_path / "shared")


def test_push_refuses_a_public_repo_when_asked_for_a_private_one(tmp_path, monkeypatch):
    made = fake_hub(monkeypatch, public=True)
    with pytest.raises(ValueError, match="public"):
        written(tmp_path / "run").push_to_hub("someone/states")
    assert made[0].uploaded is None
