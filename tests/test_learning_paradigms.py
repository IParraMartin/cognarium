"""Artificial grammar learning."""

from __future__ import annotations

import math
from collections import Counter

import pytest
import torch
from helpers import ScriptedSubject, TrainableScriptedSubject

import cognarium as cg
from cognarium.errors import MissingAbilityError
from cognarium.paradigms import learning
from cognarium.responses import Query
from cognarium.stimuli import Observation, Stimulus, Trial
from cognarium.subjects.custom import CustomSubject
from cognarium.tasks.shepard import ShepardCategories

# Reber (1967), written out again here so the tests don't trust the paradigm's copy
REBER = {
    0: {
        "T": 1,
        "P": 2
    },
    1: {
        "S": 1,
        "X": 3
    },
    2: {
        "T": 2,
        "V": 4
    },
    3: {
        "X": 2,
        "S": None
    },
    4: {
        "P": 3,
        "V": None
    }
}


def GRAMMAR(**options):
    return learning.ArtificialGrammar(n_train=12, n_test=6, **options)


def run_quietly(
    paradigm,
    subject,
    folder,
    **options
):
    return cg.run(
        paradigm,
        subject,
        out=folder,
        progress=False,
        **options
    )


def reber_accepts(text):
    state = 0
    for letter in text:
        if state is None or letter not in REBER[state]:
            return False
        state = REBER[state][letter]
    return state is None


def reber_strings(longest=8):
    found = []
    pending = [(0, "")]
    while pending:
        state, prefix = pending.pop()
        for letter, target in REBER[state].items():
            text = prefix + letter
            if len(text) > longest:
                continue
            if target is None:
                found.append(text)
            else:
                pending.append((target, text))
    return set(found)


def trials_of(paradigm, participant=0):
    return paradigm.sequences()[participant]


def posterior_group(paradigm):
    learner = cg.subjects.BayesianLearner(decision="average").bind(paradigm, "posterior")
    log_posterior = learner.log_posterior(paradigm.study(0, 0))
    return paradigm.hypothesis_space().groups[int(log_posterior.argmax())]


def test_runs_on_exemplar_and_bayesian_learners(tmp_path):
    paradigm = GRAMMAR()
    for subject in [cg.subjects.ExemplarLearner(criterion=0.5), cg.subjects.BayesianLearner(decision="average")]:
        results = run_quietly(paradigm, subject, tmp_path / type(subject).__name__)
        assert results.score().iloc[0]["mean_a"] >= 0


def test_exemplar_criterion_zero_keeps_scores_and_run_key(tmp_path):
    paradigm = cg.paradigms.IteratedLearning(ShepardCategories(type="II"), generations=2, chains=2)
    results = run_quietly(paradigm, cg.subjects.ExemplarLearner(criterion=0.0), tmp_path / "run")
    assert results.manifest["key"] == "d200a201942e"
    assert results.to_pandas()["scores"].iloc[3].tolist() == [-1.3333333333333333, 0.5000188820728624]


def test_exemplar_criterion_hand_value():
    learner = cg.subjects.ExemplarLearner(c=2.0, criterion=0.5)
    seen = Observation(Stimulus(features=torch.tensor([1.0, 0.0])), "A")
    query = Query(
        trial=Trial(Stimulus(features=torch.tensor([0.0, 0.0])), options=("A", "B")),
        context=(seen,)
    )
    # the attention weights are 1/2 each, so the distance is 1/2 and the similarity exp(-1)
    scores = learner.respond([query]).scores[0].tolist()
    assert scores == pytest.approx([math.log(0.5 + math.exp(-1)), math.log(0.5)])


def test_exemplar_criterion_is_in_the_config_only_when_set():
    assert "criterion" not in cg.subjects.ExemplarLearner().config()
    assert cg.subjects.ExemplarLearner(criterion=0.5).config()["criterion"] == 0.5


def test_exemplar_criterion_must_not_be_negative():
    with pytest.raises(ValueError, match="criterion"):
        cg.subjects.ExemplarLearner(criterion=-1.0)


def test_exemplar_without_a_criterion_is_refused_on_single_label_training(tmp_path):
    with pytest.raises(MissingAbilityError, match="criterion="):
        run_quietly(GRAMMAR(), cg.subjects.ExemplarLearner(), tmp_path / "run")


def test_bayesian_recovers_the_grammar(tmp_path):
    paradigm = GRAMMAR()
    assert posterior_group(paradigm) == "true"
    results = run_quietly(paradigm, cg.subjects.BayesianLearner(decision="average"), tmp_path / "run")
    above_chance = results.contrast("answer_probability", against=0.5).iloc[0]
    assert above_chance["detected"] and above_chance["difference"] > 0
    scores = results.score().set_index("effect")
    assert scores.loc["grammaticality", "detected"]
    assert not scores.loc["chunk_strength", "detected"]


def test_grammar_hypotheses_are_the_letter_permutations():
    space = GRAMMAR().hypothesis_space()
    assert len(space.hypotheses) == 120
    assert Counter(space.groups) == {"true": 1, "other": 119}


def test_exemplar_matches_hand_values_and_tracks_chunk_strength(tmp_path):
    paradigm = GRAMMAR(n_participants=8)
    c, criterion = 4.0, 1.0
    results = run_quietly(paradigm, cg.subjects.ExemplarLearner(c=c, criterion=criterion), tmp_path / "run")
    frame = results.to_pandas().query("participant == 0")
    exemplars = torch.stack([observation.stimulus.features for observation in paradigm.study(0, 0)]).double()
    for _, row in frame.iterrows():
        features = trials_of(paradigm)[row["trial"]].stimulus.features.double()
        summed = torch.exp(-c * (features - exemplars).abs().mean(dim=1)).sum().item()
        expected = (criterion + summed) / (2 * criterion + summed)
        assert math.exp(row["log_probs"][0]) == pytest.approx(expected)
    chunk = results.score().set_index("effect").loc["chunk_strength"]
    assert chunk["detected"] and chunk["difference"] > 0


def test_grammar_strings_obey_the_transitions():
    paradigm = GRAMMAR(n_participants=3)
    for participant in range(3):
        for observation in paradigm.study(participant, 0):
            assert reber_accepts(observation.stimulus.text.replace(" ", ""))
        for trial in trials_of(paradigm, participant):
            assert reber_accepts(trial.stimulus.text.replace(" ", "")) == (trial.condition["status"] == "grammatical")


def test_ungrammatical_strings_leave_the_grammar_by_one_letter():
    grammatical = reber_strings()
    for trial in trials_of(GRAMMAR(n_participants=2), 1):
        if trial.condition["status"] != "ungrammatical":
            continue
        text = trial.stimulus.text.replace(" ", "")
        assert any(
            len(source) == len(text) and sum(a != b for a, b in zip(source, text, strict=True)) == 1
            for source in grammatical
        )


def test_test_strings_are_not_training_strings():
    paradigm = GRAMMAR()
    trained = {observation.stimulus.text for observation in paradigm.study(0, 0)}
    shown = [trial.stimulus.text for trial in trials_of(paradigm)]
    assert len(set(shown)) == len(shown) == 12
    assert not trained & set(shown)


def test_grammar_strings_have_distinct_features():
    paradigm = GRAMMAR()
    stimuli = [observation.stimulus for observation in paradigm.study(0, 0)]
    stimuli += [trial.stimulus for trial in trials_of(paradigm)]
    assert len({tuple(stimulus.features.tolist()) for stimulus in stimuli}) == len(stimuli)


def test_each_change_is_the_one_nearest_its_original_in_chunk_strength():
    paradigm = GRAMMAR(n_participants=3)
    _, variants = learning._grammar_pool()
    for participant in range(3):
        studied, tested, changed = paradigm._material(paradigm._participant_rng(participant))
        counts = learning._chunk_counts(studied)
        used = []
        for text, change in zip(tested, changed, strict=True):
            target = learning._chunk_strength(text, counts)
            free = [variant for variant in variants[text] if variant not in used] or [change]
            gaps = [abs(learning._chunk_strength(variant, counts) - target) for variant in free]
            assert abs(learning._chunk_strength(change, counts) - target) == min(gaps)
            used.append(change)


def test_chunk_strength_is_the_mean_count_of_letter_pairs_and_triples():
    counts = learning._chunk_counts(["TXS", "TXX"])
    # TX twice, XS once, TXS once, out of the pairs TX and XS and the triple TXS
    assert learning._chunk_strength("TXS", counts) == pytest.approx((2 + 1 + 1) / 3)
    assert learning._chunk_strength("PVV", counts) == 0


def test_each_status_splits_into_high_and_low_chunk_strength(tmp_path):
    paradigm = GRAMMAR(n_participants=2)
    results = run_quietly(paradigm, cg.subjects.ExemplarLearner(criterion=0.5), tmp_path / "run")
    frame = results.to_pandas()
    for (_, status), part in frame.groupby(["participant", "status"]):
        assert (part["chunk"] == "high").sum() == (part["chunk"] == "low").sum() == 3
        assert part.loc[part["chunk"] == "high", "chunk_strength"].min() >= (
            part.loc[part["chunk"] == "low", "chunk_strength"].max()
        )
    assert results.manifest["paradigm"]["parameters"]["test"] == "judgment"


def test_the_probability_test_compares_each_string_with_its_change():
    paradigm = GRAMMAR(test="probability")
    trials = trials_of(paradigm)
    assert len(trials) == 6
    for trial in trials:
        assert trial.answer in trial.options and len(trial.options) == 2
        assert trial.answer.replace(" ", "") == trial.item
        assert learning._accepts(trial.item)
        change = next(option for option in trial.options if option != trial.answer)
        assert not learning._accepts(change.replace(" ", ""))
    assert paradigm.template.complete
    assert paradigm.modalities == frozenset({"text"})
    studied = paradigm.study(0, 0)
    step = paradigm.step(0, None)
    prompt = step.queries[0].prompt()
    assert prompt.startswith(f"Letters: {studied[0].label}\n")
    assert prompt.endswith("\nLetters:")
    assert [effect.name for effect in paradigm.effects] == ["preference"]


def test_the_probability_test_runs_on_a_reader_and_not_on_a_feature_learner(tmp_path):
    results = run_quietly(GRAMMAR(test="probability"), ScriptedSubject(), tmp_path / "run")
    frame = results.to_pandas()
    assert {"chunk_strength", "chunk_difference"} <= set(frame.columns)
    assert results.score().iloc[0]["effect"] == "preference"
    with pytest.raises(MissingAbilityError, match="text"):
        run_quietly(GRAMMAR(test="probability"), cg.subjects.ExemplarLearner(), tmp_path / "exemplar")


def test_grammar_refuses_an_unknown_test():
    with pytest.raises(ValueError, match="test"):
        GRAMMAR(test="recall")


def test_grammar_refuses_more_strings_than_it_has():
    with pytest.raises(ValueError, match="n_train"):
        learning.ArtificialGrammar(n_train=30, n_test=20)


def test_lstm_in_weights_mode_trains_once_on_the_studied_strings(tmp_path, monkeypatch):
    learned = []
    original = CustomSubject.learn

    def recording(self, observations, **options):
        learned.append(len(observations))
        return original(self, observations, **options)

    monkeypatch.setattr(CustomSubject, "learn", recording)
    subject = cg.subjects.LSTM(
        device="cpu",
        train_epochs=1,
        embedding_dim=4,
        hidden_dim=4
    )
    results = run_quietly(GRAMMAR(), subject, tmp_path / "run")
    assert results.manifest["learning"] == "weights"
    assert learned == [12]


def test_a_weight_learner_starts_fresh_for_every_participant(tmp_path):
    subject = TrainableScriptedSubject()
    run_quietly(GRAMMAR(n_participants=2), subject, tmp_path / "run")
    assert len(subject.fresh_seeds) == 2
    assert [len(observations) for observations in subject.learned] == [12, 12]
    # the context lives in the weights, so the prompt holds none of it
    assert subject.max_context_seen == 0
