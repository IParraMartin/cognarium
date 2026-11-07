from __future__ import annotations

import json

import pandas as pd
import pytest
from helpers import ScriptedSubject

import cognarium as cg


class Words(cg.Paradigm):
    item_columns = ("word", "kind")

    def default_items(self, rng):
        return [{"word": "built", "kind": "in"}]

    def trials(self, rng):
        for row in self.item_rows(rng):
            yield cg.Trial(
                cg.Stimulus(text=row["word"]),
                options=["yes", "no"],
                answer="yes",
                condition={"kind": row["kind"]}
            )


ROWS = [{"word": "cup", "kind": "a"}, {"word": "dog", "kind": "b"}]


def texts(paradigm):
    return sorted(trial.stimulus.text for row in paradigm.sequences() for trial in row)


def test_built_in_rows_are_the_default_and_are_labelled():
    paradigm = Words()
    assert texts(paradigm) == ["built"]
    assert paradigm.config()["stimuli"] == "built-in, not human-normed"


def test_stimuli_accepts_a_list_of_rows():
    assert texts(Words(stimuli=ROWS)) == ["cup", "dog"]


def test_stimuli_accepts_a_data_frame():
    assert texts(Words(stimuli=pd.DataFrame(ROWS))) == ["cup", "dog"]


def test_stimuli_accepts_a_dataset():
    dataset = cg.data.Dataset(pd.DataFrame(ROWS), text="word", source="my words")
    paradigm = Words(stimuli=dataset)
    assert texts(paradigm) == ["cup", "dog"]
    assert paradigm.config()["stimuli"] == "my words"


@pytest.mark.parametrize("suffix", ["csv", "json", "jsonl", "parquet"])
def test_stimuli_accepts_a_file(suffix, tmp_path):
    path = tmp_path / f"words.{suffix}"
    if suffix == "csv":
        pd.DataFrame(ROWS).to_csv(path, index=False)
    elif suffix == "json":
        path.write_text(json.dumps(ROWS))
    elif suffix == "jsonl":
        path.write_text("\n".join(json.dumps(row) for row in ROWS))
    else:
        pd.DataFrame(ROWS).to_parquet(path)
    paradigm = Words(stimuli=path)
    assert texts(paradigm) == ["cup", "dog"]
    assert paradigm.config()["stimuli"] == path.name
    assert texts(Words(stimuli=str(path))) == ["cup", "dog"]


def test_a_row_missing_a_column_is_named():
    with pytest.raises(ValueError, match=r"Row 1.*'kind'"):
        Words(stimuli=[{"word": "a", "kind": "x"}, {"word": "b"}])


def test_a_frame_missing_a_column_names_it():
    with pytest.raises(ValueError, match="'kind'"):
        Words(stimuli=pd.DataFrame({"word": ["a"]}))


def test_a_blank_cell_is_named(tmp_path):
    path = tmp_path / "words.csv"
    path.write_text("word,kind\na,x\nb,\n")
    with pytest.raises(ValueError, match=r"Row 1.*'kind'"):
        Words(stimuli=path)


def test_an_unknown_suffix_fails(tmp_path):
    path = tmp_path / "words.txt"
    path.write_text("a")
    with pytest.raises(ValueError, match=r"\.txt"):
        Words(stimuli=path)


def test_a_missing_file_fails(tmp_path):
    with pytest.raises(FileNotFoundError, match="nothing.csv"):
        Words(stimuli=tmp_path / "nothing.csv")


def test_no_rows_fail():
    with pytest.raises(ValueError, match="no rows"):
        Words(stimuli=[])


def test_stimuli_on_a_paradigm_without_item_columns_fails():
    class Plain(cg.Paradigm):
        def trials(self, rng):
            yield cg.Trial(cg.Stimulus(text="a"), options=["a", "b"])

    with pytest.raises(ValueError, match="item_columns"):
        Plain(stimuli=ROWS)


def test_unknown_stimuli_type_fails():
    with pytest.raises(ValueError, match="stimuli"):
        Words(stimuli=3)
    with pytest.raises(ValueError, match="stimuli"):
        Words(stimuli=[1, 2])


def test_a_paradigm_with_no_built_in_items_says_so():
    class Bare(Words):
        def default_items(self, rng):
            return super(Words, self).default_items(rng)

    with pytest.raises(NotImplementedError, match="Bare"):
        Bare().sequences()


def test_changing_the_stimuli_changes_the_run_key(tmp_path):
    key = lambda rows, name: cg.run(
        Words(stimuli=rows),
        ScriptedSubject(),
        out=tmp_path / name,
        progress=False
    ).manifest["key"]
    assert key(ROWS, "a") == key(ROWS, "b")
    assert key(ROWS, "a") != key(ROWS[:1], "c")
