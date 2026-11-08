"""Downloads every verified registry dataset and reads it: pytest -m network."""

from __future__ import annotations

import pytest

import cognarium as cg
from cognarium.data import check
from cognarium.data.registry import REGISTRY

pytestmark = pytest.mark.network

READY = sorted(name for name, entry in REGISTRY.items() if entry.status == "ready")
EXPECTED = {
    "wordsim353": (353, ("Word 1", "Word 2", "Human (mean)")),
    "blimp": (67000, ("good", "bad", "phenomenon")),
    "syntaxgym": (3304, ("suite", "item", "condition", "sentence")),
    "provo": (230412, ("Participant_ID", "Text_ID", "Word", "IA_FIRST_FIXATION_DURATION")),
    "things_triplets": (4699160, ("image1", "image2", "image3", "choice", "RT")),
    "thingsplus": (1854, ("Word", "uniqueID", "property_manmade_mean")),
    "bigtom": (200, ("story", "belief_aware", "belief_not_aware")),
    "navarro2018": (38, ("ppt", "Gender", "Age")),
    "many_labs_2": (7263, (".id", "ResponseID")),
    "many_labs_2_slate_2": (8042, (".id", "ResponseID")),
    "tomi": (5994, ("story", "question", "answer", "question_type", "story_type")),
    "cifar10h": (10000, ("image", "airplane", "truck")),
    "natural_stories": (848875, ("WorkerId", "item", "zone", "RT", "word")),
    "fantom": (870, ("set_id", "full_context", "beliefQAs")),
    "simlex999": (999, ("word1", "word2", "SimLex999")),
    "wcs_chips": (330, ("chip", "row", "column", "L", "a", "b")),
    "semantic_priming_project": (847723, ("Subject", "prime", "target", "target.RT", "rel")),
    "elp_trials": (2745951, ("subject", "session", "item", "rt", "accuracy")),
    "brysbaert_concreteness": (39954, ("Word", "Conc.M", "Conc.SD")),
    "warriner": (13915, ("Word", "V.Mean.Sum", "A.Mean.Sum", "D.Mean.Sum"))
}


def test_every_ready_dataset_is_expected():
    assert set(READY) == set(EXPECTED)


@pytest.mark.parametrize("name", READY)
def test_dataset_downloads_and_loads(name):
    table = cg.data.load(name, accept_terms=True).to_pandas()
    rows, columns = EXPECTED[name]
    assert len(table) == rows
    assert set(columns) <= set(table.columns)


def test_every_link_answers():
    assert check.check_links() == {}


def test_a_blimp_phenomenon_holds_its_thousand_pairs():
    pairs = cg.data.load("blimp", phenomena=["causative"])
    assert len(pairs) == 1000
