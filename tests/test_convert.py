from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

import cognarium as cg
from cognarium.models.lstm import SimpleTokenizer


def write_csv(folder, content, name="trials.csv"):
    path = folder / name
    path.write_text(content)
    return path


def test_from_csv_as_in_the_spec(tmp_path):
    path = write_csv(tmp_path, "prompt,answer\nhello,A\nbye,B\n")
    dataset = cg.data.from_csv(path, text="prompt", label="answer")
    assert dataset.to_numpy()["label"].tolist() == [0, 1]
    assert dataset.vocabularies["label"] == ["A", "B"]
    assert dataset.to_tensor()["label"].tolist() == [0, 1]
    assert [trial.answer for trial in dataset.to_trials()] == ["A", "B"]
    assert [trial.options for trial in dataset.to_trials()] == [("A", "B"), ("A", "B")]
    assert dataset.to_numpy()["text"].tolist() == ["hello", "bye"]
    assert len(dataset) == 2 and dataset.columns == ["prompt", "answer"]


def test_words_that_look_missing_stay_text(tmp_path):
    path = write_csv(tmp_path, "prompt,answer\nnull,A\nNA,B\nNone,A\n")
    dataset = cg.data.from_csv(path, text="prompt", label="answer")
    assert dataset.to_numpy()["text"].tolist() == ["null", "NA", "None"]


def test_to_tensor_tokenizes_and_pads(tmp_path):
    path = write_csv(tmp_path, "prompt,answer\na b c,A\na,B\n")
    tokenizer = SimpleTokenizer(["a", "b", "c"])
    tensors = cg.data.from_csv(path, text="prompt", label="answer").to_tensor(tokenizer=tokenizer)
    assert tensors["input_ids"].tolist() == [[2, 3, 4], [2, 0, 0]]
    assert tensors["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]


def test_to_tensor_tokenizes_with_a_hugging_face_tokenizer(tiny_tokenizer, tmp_path):
    path = write_csv(tmp_path, "prompt,answer\nsmall red,A\nblue,B\n")
    tensors = cg.data.from_csv(path, text="prompt", label="answer").to_tensor(tokenizer=tiny_tokenizer)
    assert tensors["input_ids"].shape == (2, 2)
    assert tensors["attention_mask"].tolist() == [[1, 1], [1, 0]]


def test_to_tensor_refuses_text_over_the_limit_and_a_missing_text_role(tmp_path):
    path = write_csv(tmp_path, "prompt,answer\na b c,A\na,B\n")
    tokenizer = SimpleTokenizer(["a", "b", "c"])
    dataset = cg.data.from_csv(path, text="prompt", label="answer")
    with pytest.raises(ValueError, match="Row 0"):
        dataset.to_tensor(tokenizer=tokenizer, max_length=2)
    with pytest.raises(ValueError, match="text"):
        cg.data.from_csv(path, label="answer").to_tensor(tokenizer=tokenizer)


def test_from_json_reads_arrays(tmp_path):
    path = tmp_path / "rows.json"
    path.write_text(json.dumps([{"s": "a", "y": "A"}, {"s": "b", "y": "B"}]))
    assert cg.data.from_json(path, text="s", label="y").to_numpy()["label"].tolist() == [0, 1]


def test_from_json_reads_jsonl(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"s": "a", "y": "A"}\n\n{"s": "b", "y": "B"}\n')
    assert cg.data.from_json(path, text="s", label="y").to_numpy()["text"].tolist() == ["a", "b"]


def test_from_json_reads_nested_paths(tmp_path):
    path = tmp_path / "rows.json"
    rows = [{"stim": {"sentence": "a"}, "y": "A"}, {"stim": {"sentence": "b"}, "y": "B"}]
    path.write_text(json.dumps({"data": {"items": rows}}))
    dataset = cg.data.from_json(
        path,
        records="data.items",
        text="stim.sentence",
        label="y"
    )
    assert dataset.to_numpy()["text"].tolist() == ["a", "b"]
    assert "sentence" in dataset.columns


def test_from_json_keeps_a_colliding_path_whole(tmp_path):
    path = tmp_path / "rows.json"
    rows = [{"a": {"x": "one"}, "x": "two", "y": "A"}, {"a": {"x": "three"}, "x": "four", "y": "B"}]
    path.write_text(json.dumps(rows))
    dataset = cg.data.from_json(path, text="a.x", label="y")
    assert dataset.to_numpy()["text"].tolist() == ["one", "three"]
    assert "a.x" in dataset.columns and "x" in dataset.columns


def test_from_json_reads_an_object_of_columns(tmp_path):
    path = tmp_path / "rows.json"
    path.write_text(json.dumps({"s": ["a", "b"], "y": ["A", "B"]}))
    assert len(cg.data.from_json(path, text="s", label="y")) == 2


def test_from_json_explains_an_object_without_records(tmp_path):
    path = tmp_path / "rows.json"
    path.write_text(json.dumps({"meta": 1, "data": [{"s": "a"}]}))
    with pytest.raises(ValueError, match="records="):
        cg.data.from_json(path, text="s")


def test_from_json_names_a_wrong_records_path(tmp_path):
    path = tmp_path / "rows.json"
    path.write_text(json.dumps({"data": [{"s": "a"}]}))
    with pytest.raises(ValueError, match="'nothing'"):
        cg.data.from_json(path, records="nothing", text="s")
    with pytest.raises(ValueError, match="list"):
        cg.data.from_json(path, records="data.0", text="s")


def test_from_json_names_a_malformed_file(tmp_path):
    path = tmp_path / "rows.json"
    path.write_text("{not json")
    with pytest.raises(ValueError, match="rows.json"):
        cg.data.from_json(path, text="s")
    lines = tmp_path / "rows.jsonl"
    lines.write_text('{"s": "a"}\n{broken\n')
    with pytest.raises(ValueError, match="line 2"):
        cg.data.from_json(lines, text="s")


def test_parquet_round_trip_keeps_roles_and_vocabularies(tmp_path):
    path = write_csv(tmp_path, "prompt,answer,kind,x\nhello,A,p,1\nbye,B,q,2\n")
    dataset = cg.data.from_csv(
        path,
        text="prompt",
        label="answer",
        condition=["kind"],
        features=["x"],
        options=["A", "B", "C"]
    )
    dataset.to_parquet(tmp_path / "out.parquet")
    back = cg.data.Dataset.from_parquet(tmp_path / "out.parquet")
    assert back.vocabularies == dataset.vocabularies
    assert back.to_trials()[0].options == ("A", "B", "C")
    assert back.to_trials()[1].condition == {"kind": "q"}
    assert back.to_trials()[1].stimulus.features.tolist() == [2.0]
    assert not list(tmp_path.glob("*.tmp"))


def test_a_plain_parquet_file_opens_with_no_roles(tmp_path):
    pd.DataFrame({"word": ["a"]}).to_parquet(tmp_path / "plain.parquet")
    assert cg.data.Dataset.from_parquet(tmp_path / "plain.parquet").columns == ["word"]


def test_vocabulary_is_stable_when_rows_are_reordered(tmp_path):
    first = cg.data.from_csv(write_csv(tmp_path, "p,y\na,B\nb,A\n", "a.csv"), text="p", label="y")
    second = cg.data.from_csv(write_csv(tmp_path, "p,y\nb,A\na,B\n", "b.csv"), text="p", label="y")
    assert first.vocabularies == second.vocabularies == {"label": ["A", "B"]}


def test_a_given_vocabulary_is_kept_and_checked():
    table = pd.DataFrame({"p": ["a"], "y": ["A"]})
    dataset = cg.data.Dataset(
        table,
        text="p",
        label="y",
        vocabularies={"label": ["Z", "A"]}
    )
    assert dataset.to_numpy()["label"].tolist() == [1]
    with pytest.raises(ValueError, match="'A'"):
        cg.data.Dataset(
            table,
            text="p",
            label="y",
            vocabularies={"label": ["Z"]}
        )


def test_features_stack_into_a_float_matrix(tmp_path):
    path = write_csv(tmp_path, "x,z,y\n1,2.5,A\n3,4,B\n")
    dataset = cg.data.from_csv(path, features=["x", "z"], label="y")
    assert dataset.to_numpy()["features"].tolist() == [[1.0, 2.5], [3.0, 4.0]]
    assert dataset.to_tensor()["features"].dtype == torch.float32
    assert dataset.to_trials()[0].stimulus.text is None


def test_a_feature_column_of_lists_stacks(tmp_path):
    path = tmp_path / "rows.json"
    path.write_text(json.dumps([{"v": [1, 2], "y": "A"}, {"v": [3, 4], "y": "B"}]))
    dataset = cg.data.from_json(path, features="v", label="y")
    assert np.array_equal(dataset.to_numpy()["features"], [[1.0, 2.0], [3.0, 4.0]])


def test_uneven_feature_lists_name_the_row(tmp_path):
    path = tmp_path / "rows.json"
    path.write_text(json.dumps([{"v": [1, 2], "y": "A"}, {"v": [3], "y": "B"}]))
    with pytest.raises(ValueError, match="Row 1"):
        cg.data.from_json(path, features="v", label="y")


def test_conditions_and_items_reach_the_trials(tmp_path):
    path = write_csv(tmp_path, "p,y,kind,id\na,A,x,10\nb,B,z,11\n")
    dataset = cg.data.from_csv(
        path,
        text="p",
        label="y",
        condition=["kind"],
        item="id"
    )
    trial = dataset.to_trials()[1]
    assert trial.condition == {"kind": "z"} and trial.item == 11
    assert dataset.to_numpy()["kind"].tolist() == [0, 1]
    assert dataset.vocabularies["kind"] == ["x", "z"]


def test_missing_column_lists_what_exists(tmp_path):
    path = write_csv(tmp_path, "prompt,answer\na,A\n")
    with pytest.raises(ValueError, match=r"'nothing'.*prompt"):
        cg.data.from_csv(path, text="nothing", label="answer")


def test_blank_label_names_its_row(tmp_path):
    path = write_csv(tmp_path, "prompt,answer\na,A\nb,\n")
    with pytest.raises(ValueError, match=r"Row 1.*answer"):
        cg.data.from_csv(path, text="prompt", label="answer")


def test_blank_text_names_its_row(tmp_path):
    path = write_csv(tmp_path, "prompt,answer\n,A\nb,B\n")
    with pytest.raises(ValueError, match=r"Row 0.*prompt"):
        cg.data.from_csv(path, text="prompt", label="answer")


def test_non_numeric_feature_names_row_and_column(tmp_path):
    path = write_csv(tmp_path, "x,y\n1,A\nabc,B\n")
    with pytest.raises(ValueError, match=r"Row 1.*'x'"):
        cg.data.from_csv(path, features=["x"], label="y")


def test_non_utf8_file_names_the_file(tmp_path):
    path = tmp_path / "latin.csv"
    path.write_bytes("prompt,answer\ncafé,A\n".encode("latin-1"))
    with pytest.raises(ValueError, match=r"latin\.csv.*UTF-8"):
        cg.data.from_csv(path, text="prompt", label="answer")
    path = tmp_path / "latin.json"
    path.write_bytes('[{"s": "café"}]'.encode("latin-1"))
    with pytest.raises(ValueError, match=r"latin\.json.*UTF-8"):
        cg.data.from_json(path, text="s")


def test_utf8_text_is_read_as_utf8(tmp_path):
    path = tmp_path / "words.csv"
    path.write_bytes("prompt,answer\ncafé,A\n".encode())
    assert cg.data.from_csv(path, text="prompt", label="answer").to_numpy()["text"].tolist() == ["café"]


def test_missing_file_names_the_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="nothing.csv"):
        cg.data.from_csv(tmp_path / "nothing.csv", text="p")
    with pytest.raises(FileNotFoundError, match="nothing.json"):
        cg.data.from_json(tmp_path / "nothing.json", text="p")


def test_empty_file_fails_clearly(tmp_path):
    empty = write_csv(tmp_path, "", "empty.csv")
    with pytest.raises(ValueError, match="empty"):
        cg.data.from_csv(empty, text="p")
    headers = write_csv(tmp_path, "p,y\n", "headers.csv")
    with pytest.raises(ValueError, match="no rows"):
        cg.data.from_csv(headers, text="p")
    array = tmp_path / "empty.json"
    array.write_text("[]")
    with pytest.raises(ValueError, match="no rows"):
        cg.data.from_json(array, text="p")


def test_label_outside_the_options_fails(tmp_path):
    path = write_csv(tmp_path, "p,y\na,A\nb,C\n")
    dataset = cg.data.from_csv(path, text="p", label="y")
    with pytest.raises(ValueError, match=r"Row 1.*'C'"):
        dataset.to_trials(options=["A", "B"])
    with pytest.raises(ValueError, match=r"Row 1.*'C'"):
        cg.data.from_csv(
            path,
            text="p",
            label="y",
            options=["A", "B"]
        )


def test_trials_need_a_stimulus_and_options(tmp_path):
    path = write_csv(tmp_path, "p,y\na,A\nb,B\n")
    with pytest.raises(ValueError, match="text or features"):
        cg.data.from_csv(path, label="y").to_trials()
    with pytest.raises(ValueError, match="options"):
        cg.data.from_csv(path, text="p").to_trials()
    assert cg.data.from_csv(path, text="p").to_trials(options=["x", "y"])[0].answer is None


def test_to_pandas_gives_a_copy(tmp_path):
    dataset = cg.data.from_csv(write_csv(tmp_path, "p,y\na,A\n"), text="p")
    table = dataset.to_pandas()
    table.loc[0, "p"] = "changed"
    assert dataset.to_pandas().loc[0, "p"] == "a"


def test_trials_run_in_a_paradigm(tmp_path):
    from helpers import ScriptedSubject

    dataset = cg.data.from_csv(write_csv(tmp_path, "p,y\na,A\nb,B\n"), text="p", label="y")

    class FromFile(cg.Paradigm):
        def trials(self, rng):
            return dataset.to_trials()

    frame = cg.run(
        FromFile(),
        ScriptedSubject(),
        out=tmp_path / "run",
        progress=False
    ).to_pandas()
    assert sorted(frame["answer"]) == ["A", "B"]
