"""Priming, lexical decision, garden paths, minimal pairs and verbal fluency."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from importlib import resources

import pandas as pd
import pytest
import torch
from helpers import ScriptedSubject
from paradigm_cases import EffectSubject

import cognarium as cg
from cognarium.paradigms import language
from cognarium.paradigms.base import load_items
from cognarium.responses import Query
from cognarium.subjects.hf import HFSubject

ITEM_KEYS = {
    "semantic_priming": "target",
    "structural_priming": "target_po",
    "lexical_decision": "letters",
    "garden_path": "ambiguous",
    "minimal_pairs": "good"
}


def run_quietly(paradigm, subject, folder):
    return cg.run(
        paradigm,
        subject,
        out=folder,
        progress=False
    )


def trials_of(paradigm, participant=0):
    return paradigm.sequences()[participant]


def writing(text):
    """A subject that writes the same text for every trial."""
    return EffectSubject(None, text_for=lambda trial, effect: text)


def tokens(text):
    return re.findall(r"\w+|[^\w\s]", text)


def is_subsequence(short, long):
    remaining = iter(long)
    return all(token in remaining for token in short)


def test_priming_targets_appear_in_both_conditions():
    levels = defaultdict(list)
    for trial in trials_of(language.SemanticPriming()):
        if trial.condition["relatedness"] != "filler":
            levels[trial.item].append(trial.condition["relatedness"])
    assert len(levels) >= 24
    assert all(sorted(found) == ["related", "unrelated"] for found in levels.values())


def test_priming_filler_targets_are_answered_nonword():
    for trial in trials_of(language.SemanticPriming()):
        assert trial.options == ("word", "nonword")
        assert (trial.answer == "nonword") == (trial.condition["relatedness"] == "filler")


def test_priming_shows_the_prime_then_the_target():
    trial = trials_of(language.SemanticPriming())[0]
    prime, target = trial.condition["prime"], trial.item
    assert trial.stimulus.text == f"Prime: {prime}. Target: {target}. Is the target a word?"


def test_priming_primes_are_unrelated_to_the_targets_they_are_not_paired_with():
    rows = load_items("semantic_priming")
    related = {(row["prime"], row["target"]) for row in rows if row["relatedness"] == "related"}
    unrelated = {(row["prime"], row["target"]) for row in rows if row["relatedness"] == "unrelated"}
    assert not related & unrelated
    # every prime appears once in each condition, so primes are matched across them
    assert Counter(prime for prime, _ in related) == Counter(prime for prime, _ in unrelated)


def test_structural_prime_and_target_use_the_same_verb():
    for row in load_items("structural_priming"):
        for column in ("prime_po", "prime_do", "target_po", "target_do"):
            assert re.search(rf"\b{row['verb']}\b", row[column]), (column, row)


def test_structural_options_are_the_two_completions_po_first():
    paradigm = language.StructuralPriming()
    rows = {row["target_po"]: row for row in load_items("structural_priming")}
    seen = Counter()
    for trial in trials_of(paradigm):
        row = rows[trial.item]
        structure = trial.condition["prime_structure"]
        seen[trial.item] += 1
        assert trial.options == (row["target_po"], row["target_do"])
        assert trial.stimulus.text == row[f"prime_{structure}"]
        prompt = Query(trial=trial, template=paradigm.template).prompt()
        assert prompt == f"Sentence: {row[f'prime_{structure}']}\nSentence:"
    assert set(seen.values()) == {2}


def test_structural_dative_forms_differ_as_expected():
    for row in load_items("structural_priming"):
        assert " to " in row["prime_po"] and " to " in row["target_po"]
        assert sorted(tokens(row["prime_po"].replace(" to ", " "))) == sorted(tokens(row["prime_do"]))
        assert sorted(tokens(row["target_po"].replace(" to ", " "))) == sorted(tokens(row["target_do"]))


def test_lexical_decision_answer_matches_lexicality():
    for trial in trials_of(language.LexicalDecision()):
        lexicality = trial.condition["lexicality"]
        assert trial.answer == lexicality
        assert trial.stimulus.text == f"Is this a word? {trial.item}"
        expected = {"high", "low"} if lexicality == "word" else {None}
        assert trial.condition["frequency"] in expected


def test_nonwords_and_words_do_not_overlap():
    lexical = load_items("lexical_decision")
    priming = load_items("semantic_priming")
    nonwords = {row["letters"] for row in lexical if row["lexicality"] == "nonword"}
    nonwords |= {row["target"] for row in priming if row["relatedness"] == "filler"}
    words = {row["letters"] for row in lexical if row["lexicality"] == "word"}
    words |= {row["prime"] for row in priming}
    words |= {row["target"] for row in priming if row["relatedness"] != "filler"}
    assert not words & nonwords


def test_lexical_decision_reports_d_prime_with_words_positive(tmp_path):
    results = run_quietly(language.LexicalDecision(), ScriptedSubject(probabilities=(0.8, 0.2)), tmp_path / "run")
    table = results.metrics("d_prime", positive="word")
    assert len(table) == 1


def test_garden_path_pair_differs_only_by_the_disambiguating_material():
    for row in load_items("garden_path"):
        ambiguous, unambiguous = tokens(row["ambiguous"]), tokens(row["unambiguous"])
        assert len(unambiguous) > len(ambiguous)
        assert ambiguous[-1] == unambiguous[-1]
        assert is_subsequence(ambiguous, unambiguous), row


def test_garden_path_options_are_scored_after_the_prefix(byte_model_and_tokenizer, tmp_path):
    model, tokenizer = byte_model_and_tokenizer
    paradigm = language.GardenPath(stimuli=load_items("garden_path")[:2])
    frame = run_quietly(paradigm, HFSubject(model, tokenizer, device="cpu"), tmp_path / "run").to_pandas()
    assert len(frame) == 4
    for _, row in frame.iterrows():
        trial = trials_of(paradigm)[row["trial"]]
        assert trial.options == (trial.answer, trial.condition["foil"])
        prompt = tokenizer(trial.stimulus.text)["input_ids"]
        for position, option in enumerate(trial.options):
            ids = tokenizer(" " + option, add_special_tokens=False)["input_ids"]
            with torch.inference_mode():
                log_probs = model(torch.tensor([prompt + ids])).logits.double().log_softmax(dim=-1)[0]
            expected = sum(log_probs[len(prompt) + offset - 1, token].item() for offset, token in enumerate(ids))
            assert row["scores"][position] == pytest.approx(expected, abs=1e-4)


def test_minimal_pairs_answer_is_the_good_sentence_wherever_it_lands():
    paradigm = language.MinimalPairs(n_participants=3)
    rows = {row["good"]: row for row in load_items("minimal_pairs")}
    positions = Counter()
    for participant in range(3):
        for trial in trials_of(paradigm, participant):
            row = rows[trial.answer]
            assert sorted(trial.options) == sorted((row["good"], row["bad"]))
            positions[trial.options.index(trial.answer)] += 1
    assert positions[0] > 0 and positions[1] > 0


def test_minimal_pairs_are_the_same_length_in_words():
    for row in load_items("minimal_pairs"):
        assert len(row["good"].split()) == len(row["bad"].split()), row


def test_minimal_pairs_reports_by_phenomenon(tmp_path):
    results = run_quietly(language.MinimalPairs(), ScriptedSubject(), tmp_path / "run")
    table = results.metrics("accuracy")
    phenomena = {row["phenomenon"] for row in load_items("minimal_pairs")}
    assert set(table["phenomenon"]) == phenomena


def test_minimal_pairs_takes_stimuli_rows(tmp_path):
    rows = [
        {
            "good": "The cat sleeps.",
            "bad": "The cat sleep.",
            "phenomenon": "agreement"
        },
        {
            "good": "The cats sleep.",
            "bad": "The cats sleeps.",
            "phenomenon": "agreement"
        }
    ]
    paradigm = language.MinimalPairs(stimuli=rows)
    assert len(trials_of(paradigm)) == 2
    assert paradigm.config()["stimuli"] == "your own rows"
    results = run_quietly(paradigm, ScriptedSubject(probabilities=(0.7, 0.3)), tmp_path / "run")
    assert len(results.to_pandas()) == 2


def test_fluency_prompts_name_the_category_or_letter():
    texts = {trial.condition["cue"]: trial.stimulus.text for trial in trials_of(language.Fluency())}
    assert texts["animals"] == "List as many animals as you can, separated by commas:"
    assert texts["f"] == "List as many words starting with f as you can, separated by commas:"


def test_fluency_counts_unique_valid_and_repeats():
    paradigm = language.Fluency(semantic=("animals",), letters=("f",))
    frame = pd.DataFrame({
        "response": ["Cat, dog, cat, polar bear.", "fish, Fox, apple, fish"],
        "kind": ["semantic", "letter"],
        "cue": ["animals", "f"]
    })
    annotated = paradigm.annotate(frame)
    assert annotated["produced"].tolist() == [["cat", "dog", "polar bear"], ["fish", "fox", "apple"]]
    assert annotated["n_produced"].tolist() == [3, 3]
    assert annotated["n_valid"].tolist() == [3, 2]
    assert annotated["n_repeats"].tolist() == [1, 1]


def test_fluency_handles_output_with_no_separators(tmp_path):
    frame = run_quietly(language.Fluency(), writing("cat dog horse"), tmp_path / "words").to_pandas()
    assert frame["n_produced"].tolist() == [3] * len(frame)
    frame = run_quietly(language.Fluency(), writing("..."), tmp_path / "nothing").to_pandas()
    assert frame["n_produced"].tolist() == [0] * len(frame)


def test_fluency_counts_only_chunks_of_one_or_two_words():
    paradigm = language.Fluency(semantic=("animals",), letters=("f",))
    frame = pd.DataFrame({
        "response": ["1500:273291 [A0:4C: E8:2T, cat, sea lion, the big brown dog, café", "f00, creation:tld, (fig)"],
        "kind": ["semantic", "letter"],
        "cue": ["animals", "f"]
    })
    annotated = paradigm.annotate(frame)
    assert annotated["produced"].tolist() == [["cat", "sea lion", "café"], ["fig"]]
    assert annotated["n_valid"].tolist() == [3, 1]


def test_fluency_needs_a_cue_of_each_kind():
    with pytest.raises(ValueError, match="semantic"):
        language.Fluency(semantic=(), letters=("f",))


def test_every_item_file_has_enough_items_and_no_duplicates():
    for name, key in ITEM_KEYS.items():
        rows = load_items(name)
        assert len({row[key] for row in rows}) >= 24, name
        assert len({json.dumps(row, sort_keys=True) for row in rows}) == len(rows), name


def test_item_files_reach_the_wheel_config():
    folder = resources.files("cognarium.paradigms") / "items"
    for name in ITEM_KEYS:
        loaded = json.loads((folder / f"{name}.json").read_text(encoding="utf-8"))
        assert "not human-normed" in loaded["note"] and loaded["rows"], name
