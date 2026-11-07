from __future__ import annotations

import pytest
import sklearn.metrics
import torch

from cognarium.metrics.classification import (
    accuracy,
    auprc,
    auroc,
    balanced_accuracy,
    brier,
    ece,
    f1,
    mcc,
    precision,
    recall
)

LABELS = [
    ([0, 1, 1, 0, 1, 2, 2, 0], [0, 1, 0, 0, 1, 2, 1, 2]),
    ([0, 0, 0, 0], [0, 1, 2, 1]),
    ([1, 2, 1, 2], [0, 0, 1, 1])
]
ONE_CLASS_SCORES = [
    (precision, sklearn.metrics.precision_score),
    (recall, sklearn.metrics.recall_score),
    (f1, sklearn.metrics.f1_score)
]


@pytest.mark.parametrize("predicted, target", LABELS)
def test_label_metrics_match_scikit_learn(predicted, target):
    p, t = torch.tensor(predicted), torch.tensor(target)
    assert accuracy(p, t) == pytest.approx(sklearn.metrics.accuracy_score(target, predicted))
    assert mcc(p, t) == pytest.approx(sklearn.metrics.matthews_corrcoef(target, predicted))
    for ours, theirs in ONE_CLASS_SCORES:
        assert ours(p, t) == pytest.approx(theirs(
            target,
            predicted,
            average="macro",
            zero_division=0
        ))


def test_balanced_accuracy_matches_scikit_learn():
    predicted, target = [0, 1, 1, 0, 1], [0, 1, 0, 0, 1]
    expected = sklearn.metrics.balanced_accuracy_score(target, predicted)
    assert balanced_accuracy(torch.tensor(predicted), torch.tensor(target)) == pytest.approx(expected)


def test_balanced_accuracy_ignores_classes_missing_from_targets():
    predicted, target = [0, 1, 2], [0, 0, 1]
    with pytest.warns(UserWarning):
        expected = sklearn.metrics.balanced_accuracy_score(target, predicted)
    assert balanced_accuracy(torch.tensor(predicted), torch.tensor(target)) == pytest.approx(expected)


def test_positive_class_gives_one_class_scores():
    p, t = torch.tensor([1, 1, 0, 0]), torch.tensor([1, 0, 0, 0])
    assert precision(p, t, positive=1) == pytest.approx(0.5)
    assert recall(p, t, positive=1) == pytest.approx(1.0)
    assert f1(p, t, positive=1) == pytest.approx(2 / 3)


def test_positive_class_that_never_occurs_scores_zero():
    p, t = torch.tensor([0, 0]), torch.tensor([0, 0])
    assert precision(p, t, positive=1) == 0.0


def test_one_class_everywhere_is_defined():
    p, t = torch.tensor([0, 0]), torch.tensor([0, 0])
    assert accuracy(p, t) == 1.0
    assert mcc(p, t) == 0.0


def test_binary_curves_match_scikit_learn_with_ties():
    probabilities, target = [0.9, 0.9, 0.2, 0.4, 0.4, 0.1], [1, 0, 0, 1, 1, 0]
    p, t = torch.tensor(probabilities), torch.tensor(target)
    assert auroc(p, t) == pytest.approx(sklearn.metrics.roc_auc_score(target, probabilities))
    assert auprc(p, t) == pytest.approx(sklearn.metrics.average_precision_score(target, probabilities))
    assert brier(p, t) == pytest.approx(sklearn.metrics.brier_score_loss(target, probabilities))


def test_multiclass_auroc_matches_scikit_learn():
    probabilities = [
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.2, 0.6],
        [0.5, 0.3, 0.2],
        [0.3, 0.4, 0.3],
        [0.1, 0.3, 0.6]
    ]
    target = [0, 1, 2, 1, 0, 2]
    p, t = torch.tensor(probabilities), torch.tensor(target)
    assert auroc(p, t) == pytest.approx(sklearn.metrics.roc_auc_score(target, probabilities, multi_class="ovr"))
    assert brier(p, t) == pytest.approx(sklearn.metrics.brier_score_loss(target, probabilities))


def test_two_column_brier_equals_one_column():
    t = torch.tensor([1, 0, 1])
    one = torch.tensor([0.8, 0.3, 0.6])
    assert brier(torch.stack([1 - one, one], dim=1), t) == pytest.approx(brier(one, t))


@pytest.mark.parametrize("metric", [auroc, auprc])
def test_one_class_targets_raise_rather_than_return_a_score(metric):
    with pytest.raises(ValueError, match="class"):
        metric(torch.tensor([0.2, 0.8]), torch.tensor([1, 1]))


@pytest.mark.parametrize("metric", [auroc, auprc])
def test_multiclass_with_a_missing_class_raises(metric):
    probabilities = torch.tensor([[0.5, 0.3, 0.2], [0.2, 0.6, 0.2]])
    with pytest.raises(ValueError, match="class"):
        metric(probabilities, torch.tensor([0, 1]))


@pytest.mark.parametrize("metric", [
    auroc,
    auprc,
    brier,
    ece
])
def test_values_outside_zero_and_one_raise(metric):
    with pytest.raises(ValueError, match="probabilities"):
        metric(torch.tensor([1.5, -0.5]), torch.tensor([1, 0]))


def test_ece_of_perfect_calibration_is_zero_and_of_confident_errors_is_large():
    assert ece(torch.tensor([1.0, 0.0, 1.0, 0.0]), torch.tensor([1, 0, 1, 0])) == pytest.approx(0.0)
    assert ece(torch.tensor([1.0, 1.0]), torch.tensor([0, 0])) == pytest.approx(1.0)


def test_empty_input_raises():
    with pytest.raises(ValueError, match="empty"):
        accuracy(torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="length"):
        accuracy(torch.tensor([0, 1]), torch.tensor([0]))


def test_metrics_accept_tensors_on_any_device(device):
    p, t = torch.tensor([0, 1, 1]).to(device), torch.tensor([0, 1, 0]).to(device)
    assert accuracy(p, t) == pytest.approx(2 / 3)
    assert auroc(torch.tensor([0.1, 0.9, 0.4]).to(device), t) == pytest.approx(1.0)
