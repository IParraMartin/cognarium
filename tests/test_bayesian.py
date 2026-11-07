from __future__ import annotations

import itertools

import pytest
import torch

from cognarium.paradigms.learning import IteratedLearning
from cognarium.responses import Query
from cognarium.stimuli import Observation, Stimulus, Trial
from cognarium.subjects.bayesian import BayesianLearner
from cognarium.tasks.shepard import ShepardCategories

WEIGHTS = {
    "I": 0.4,
    "II": 0.1,
    "III": 0.1,
    "IV": 0.1,
    "V": 0.1,
    "VI": 0.1,
    "unbalanced": 0.1
}


def bound(
    task,
    *,
    noise=0.0,
    prior=None,
    decision="sample",
    device="cpu",
    **kwargs
):
    made = IteratedLearning(task=task, noise=noise, **kwargs)
    return BayesianLearner(prior=prior, decision=decision, device=device).bind(made, "posterior")


def query_for(
    task,
    item,
    context,
    group=0,
    group_seed=(0,)
):
    return Query(
        trial=Trial(task.stimuli[item], options=task.options, item=item),
        context=tuple(context),
        group=group,
        group_seed=group_seed
    )


def test_log_posterior_by_hand():
    task = ShepardCategories()
    learner = bound(task, noise=0.1)
    posterior = learner.log_posterior([Observation(task.stimuli[0], "A")]).exp()
    hypotheses = task.hypothesis_space().hypotheses
    assert posterior[hypotheses[:, 0] == 0][0].item() == pytest.approx(0.9 / 128)
    assert posterior[hypotheses[:, 0] == 1][0].item() == pytest.approx(0.1 / 128)


def test_predictive_is_certain_when_seen_and_even_when_unseen():
    task = ShepardCategories()
    learner = bound(task, decision="average")
    context = [Observation(task.stimuli[0], "B")]
    queries = [query_for(task, 0, context), query_for(task, 1, context)]
    probs = learner.respond(queries).log_probs.exp()
    assert probs.flatten().tolist() == pytest.approx([0.0, 1.0, 0.5, 0.5])


def test_sample_commits_to_one_consistent_hypothesis_per_group():
    task = ShepardCategories()
    learner = bound(task)
    context = [Observation(task.stimuli[0], "B")]
    queries = [query_for(
        task,
        i,
        context,
        group_seed=(7,)
    ) for i in range(8)]
    choices = learner.respond(queries).choices
    assert choices[0].item() == 1
    assert (task.hypothesis_space().hypotheses == choices).all(dim=1).sum() == 1
    assert learner.respond(queries).choices.tolist() == choices.tolist()


def test_groups_draw_their_own_hypotheses():
    task = ShepardCategories()
    learner = bound(task)
    context = (Observation(task.stimuli[0], "B"),)
    drawn = set()
    for group in range(20):
        queries = [query_for(
            task,
            i,
            context,
            group=group,
            group_seed=(group,)
        ) for i in range(8)]
        drawn.add(tuple(learner.respond(queries).choices.tolist()))
    assert len(drawn) > 1


def test_average_leaves_choice_to_the_runner():
    task = ShepardCategories()
    assert bound(task, decision="average").respond([query_for(task, 0, [])]).choices is None


def test_map_picks_the_prior_mode():
    task = ShepardCategories()
    prior = task.type_prior({"I": 1.0})
    learner = bound(task, prior=prior, decision="map")
    choices = learner.respond([query_for(task, i, []) for i in range(8)]).choices
    labels = [task.options[c] for c in choices.tolist()]
    assert task.group_of(labels) == "I"


def test_conflicting_evidence_raises_clearly():
    task = ShepardCategories()
    context = [Observation(task.stimuli[0], "A"), Observation(task.stimuli[0], "B")]
    with pytest.raises(ValueError, match="rule out every hypothesis"):
        bound(task).log_posterior(context)


def test_zero_prior_evidence_raises_clearly():
    task = ShepardCategories()
    prior = task.type_prior({"VI": 1.0})
    context = [Observation(task.stimuli[i], "A") for i in range(8)]
    with pytest.raises(ValueError, match="rule out every hypothesis"):
        bound(task, prior=prior).log_posterior(context)


def test_shaping_with_joint_choice_fails_at_bind():
    with pytest.raises(ValueError, match="average"):
        bound(ShepardCategories(), temperature=0.5)


def test_shaping_with_average_is_allowed():
    assert bound(ShepardCategories(), decision="average", temperature=0.5) is not None


def test_bad_prior_fails_at_bind():
    with pytest.raises(ValueError, match="prior"):
        bound(ShepardCategories(), prior=torch.ones(3))


def test_example_outside_the_items_fails_at_bind():
    example = Observation(Stimulus(text="x", features=torch.ones(3) * 5), "A")
    with pytest.raises(ValueError, match="items"):
        bound(ShepardCategories(), examples=[example])


def test_unknown_decision_raises():
    with pytest.raises(ValueError, match="decision"):
        BayesianLearner(decision="vote")


def test_unbound_learner_says_so():
    with pytest.raises(RuntimeError, match="bind"):
        BayesianLearner().log_posterior([])


def test_custom_likelihood_is_used():
    task = ShepardCategories()
    made = IteratedLearning(task=task)

    def indifferent(hypothesis_labels, label_index, noise):
        return torch.zeros_like(hypothesis_labels, dtype=torch.float64)

    learner = BayesianLearner(likelihood=indifferent).bind(made, "posterior")
    posterior = learner.log_posterior([Observation(task.stimuli[0], "A")]).exp()
    assert posterior[0].item() == pytest.approx(1 / 256)


def test_group_prior_pushes_the_prior_through_groups():
    task = ShepardCategories()
    assert bound(task, prior=task.type_prior(WEIGHTS)).group_prior()["I"] == pytest.approx(0.4)


def test_config_changes_with_the_prior():
    task = ShepardCategories()
    uniform = bound(task).config()
    assert bound(task, prior=task.type_prior(WEIGHTS)).config() != uniform


def test_runs_on_every_device(device):
    task = ShepardCategories()
    context = [Observation(task.stimuli[0], "B")]
    learner = bound(task, decision="average", device=device)
    probs = learner.respond([query_for(task, 1, context)]).log_probs.exp()
    assert probs[0].tolist() == pytest.approx([0.5, 0.5])


@pytest.mark.parametrize("noise", [0.0, 0.1])
def test_prior_is_stationary_under_posterior_sampling(noise):
    task = ShepardCategories()
    prior = task.type_prior(WEIGHTS)
    learner = bound(task, noise=noise, prior=prior)
    hypotheses = task.hypothesis_space().hypotheses
    subsets = list(itertools.combinations(range(8), 4))
    transition = torch.zeros(256, 256, dtype=torch.float64)
    for subset in subsets:
        for data in itertools.product([0, 1], repeat=4):
            context = [Observation(task.stimuli[i], task.options[d]) for i, d in zip(subset, data)]
            match = hypotheses[:, list(subset)] == torch.tensor(data)
            like = (match.double() * (1 - noise) + (~match).double() * noise).prod(dim=1)
            posterior = learner.log_posterior(context).exp().double()
            transition += torch.outer(like, posterior) / len(subsets)
    assert torch.allclose(transition.sum(dim=1), torch.ones(256, dtype=torch.float64), atol=1e-10)
    assert torch.allclose(prior @ transition, prior, atol=1e-10)
