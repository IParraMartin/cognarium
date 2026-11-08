"""Category learning, artificial grammar learning and statistical learning."""

from __future__ import annotations

import math
from collections import Counter

import pytest
import torch
from helpers import TrainableScriptedSubject

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


def CATEGORY(**options):
    return learning.CategoryLearning(ShepardCategories(type="II"), blocks=3, **options)


def GRAMMAR(**options):
    return learning.ArtificialGrammar(n_train=12, n_test=6, **options)


def STREAM(**options):
    return learning.StatisticalLearning(n_words=4, stream_words=12, **options)


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


def windows(syllables):
    return [tuple(syllables[start : start + 3]) for start in range(len(syllables) - 2)]


def posterior_group(paradigm):
    learner = cg.subjects.BayesianLearner(decision="average").bind(paradigm, "posterior")
    log_posterior = learner.log_posterior(paradigm.study(0, 0))
    return paradigm.hypothesis_space().groups[int(log_posterior.argmax())]


@pytest.mark.parametrize("build", [CATEGORY, GRAMMAR, STREAM])
def test_runs_on_exemplar_and_bayesian_learners(build, tmp_path):
    paradigm = build()
    for subject in [cg.subjects.ExemplarLearner(criterion=0.5), cg.subjects.BayesianLearner(decision="average")]:
        results = run_quietly(paradigm, subject, tmp_path / type(subject).__name__)
        assert results.score().iloc[0]["mean_a"] >= 0


def _final_accuracy(type_name, subject, folder):
    paradigm = learning.CategoryLearning(ShepardCategories(type=type_name), blocks=4)
    return run_quietly(paradigm, subject, folder).score().iloc[0]["mean_a"]


def _assumed_noise(labels, label, noise):
    # feedback is always correct, but a learner that allows for errors learns gradually
    return torch.where(labels == label, math.log(0.75), math.log(0.25))


def test_every_learner_finds_shepard_type_one_easier_than_type_six(tmp_path):
    simple_first = ShepardCategories().type_prior({
        "I": 32,
        "II": 16,
        "III": 8,
        "IV": 8,
        "V": 4,
        "VI": 1
    })
    learners = {
        "exemplar": lambda: cg.subjects.ExemplarLearner(),
        "bayesian": lambda: cg.subjects.BayesianLearner(
            prior=simple_first,
            likelihood=_assumed_noise,
            decision="average"
        )
    }
    for name, make in learners.items():
        easy = _final_accuracy("I", make(), tmp_path / f"{name}-I")
        hard = _final_accuracy("VI", make(), tmp_path / f"{name}-VI")
        assert easy > hard, name


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


def test_grammar_hypotheses_are_the_letter_permutations():
    space = GRAMMAR().hypothesis_space()
    assert len(space.hypotheses) == 120
    assert Counter(space.groups) == {"true": 1, "other": 119}


def test_exemplar_prefers_grammatical_strings(tmp_path):
    paradigm = GRAMMAR()
    c, criterion = 4.0, 1.0
    results = run_quietly(paradigm, cg.subjects.ExemplarLearner(c=c, criterion=criterion), tmp_path / "run")
    frame = results.to_pandas()
    exemplars = torch.stack([observation.stimulus.features for observation in paradigm.study(0, 0)]).double()
    for _, row in frame.iterrows():
        features = trials_of(paradigm)[row["trial"]].stimulus.features.double()
        summed = torch.exp(-c * (features - exemplars).abs().mean(dim=1)).sum().item()
        expected = (criterion + summed) / (2 * criterion + summed)
        assert math.exp(row["log_probs"][0]) == pytest.approx(expected)
    primary = results.score().iloc[0]
    assert primary["detected"] and primary["difference"] > 0


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


def test_grammar_refuses_more_strings_than_it_has():
    with pytest.raises(ValueError, match="n_train"):
        learning.ArtificialGrammar(n_train=30, n_test=20)


def test_stream_has_no_immediate_repeats():
    paradigm = STREAM(n_participants=3)
    for participant in range(3):
        stream = paradigm.stream(participant)
        words = [tuple(stream[start : start + 3]) for start in range(0, len(stream), 3)]
        assert all(word in paradigm.words for word in words)
        assert all(first != second for first, second in zip(words, words[1:], strict=False))


def test_part_words_occur_in_the_stream_less_often_than_words():
    paradigm = STREAM()
    counts = Counter(windows(paradigm.stream(0)))
    by_type = {"word": [], "part_word": []}
    for trial in trials_of(paradigm):
        by_type[trial.condition["test_type"]].append(counts[tuple(trial.stimulus.text.split())])
    assert min(by_type["part_word"]) >= 1
    assert max(by_type["part_word"]) < min(by_type["word"])


def test_transitional_probabilities_are_higher_inside_words():
    paradigm = STREAM()
    stream = paradigm.stream(0)
    pairs = Counter(zip(stream, stream[1:], strict=False))
    firsts = Counter(stream[:-1])
    inside = {(word[0], word[1]) for word in paradigm.words} | {(word[1], word[2]) for word in paradigm.words}
    probabilities = {pair: count / firsts[pair[0]] for pair, count in pairs.items()}
    assert min(p for pair, p in probabilities.items() if pair in inside) > max(
        p for pair, p in probabilities.items() if pair not in inside
    )


def test_stream_refuses_too_few_words():
    with pytest.raises(ValueError, match="n_words"):
        learning.StatisticalLearning(n_words=2, stream_words=12)


def test_stream_refuses_a_length_that_isnt_a_multiple_of_the_words():
    with pytest.raises(ValueError, match="stream_words"):
        learning.StatisticalLearning(n_words=4, stream_words=10)


def test_bayesian_finds_the_words_in_the_stream(tmp_path):
    paradigm = STREAM()
    assert posterior_group(paradigm) == "true"
    results = run_quietly(paradigm, cg.subjects.BayesianLearner(decision="average"), tmp_path / "run")
    primary = results.score().iloc[0]
    assert primary["detected"] and primary["difference"] > 0


def test_exemplar_prefers_words_to_part_words(tmp_path):
    results = run_quietly(STREAM(), cg.subjects.ExemplarLearner(criterion=0.5), tmp_path / "run")
    primary = results.score().iloc[0]
    assert primary["detected"] and primary["difference"] > 0


def test_category_learning_feedback_holds_every_earlier_block_with_correct_labels():
    paradigm = CATEGORY(n_participants=2)
    task = paradigm.task
    for participant in range(2):
        row = trials_of(paradigm, participant)
        for block in range(3):
            studied = paradigm.study(participant, block)
            earlier = [trial for trial in row if trial.block < block]
            assert [observation.stimulus for observation in studied] == [trial.stimulus for trial in earlier]
            assert [observation.label for observation in studied] == [
                task.seed_labels[trial.item] for trial in earlier
            ]


def test_category_learning_asks_every_object_once_per_block():
    row = trials_of(CATEGORY())
    for block in range(3):
        assert sorted(trial.item for trial in row if trial.block == block) == list(range(8))


def test_category_learning_has_one_step_per_block():
    paradigm = CATEGORY(n_participants=2)
    steps = [paradigm.step(index, None) for index in range(4)]
    assert steps[3] is None
    assert [len(step.queries) for step in steps[:3]] == [16, 16, 16]
    shared = {id(query.context) for query in steps[2].queries if query.group == 0}
    assert len(shared) == 1


def test_category_learning_records_the_task():
    config = CATEGORY().config()
    assert config["task"] == {"name": "shepard", "type": "II"}


def test_lstm_in_weights_mode_trains_on_each_block(tmp_path, monkeypatch):
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
    results = run_quietly(CATEGORY(), subject, tmp_path / "run")
    assert results.manifest["learning"] == "weights"
    assert learned == [8, 16]


def test_a_weight_learner_starts_fresh_for_every_block(tmp_path):
    subject = TrainableScriptedSubject()
    run_quietly(CATEGORY(n_participants=2), subject, tmp_path / "run")
    assert len(subject.fresh_seeds) == 6
    assert sorted(len(observations) for observations in subject.learned) == [8, 8, 16, 16]
    # the context lives in the weights, so the prompt holds none of it
    assert subject.max_context_seen == 0
