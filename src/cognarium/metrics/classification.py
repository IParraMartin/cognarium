"""Classification metrics, always computed on CPU.

Label metrics share one confusion matrix. A macro average covers the classes present
in the predictions or the targets, as in scikit-learn.
"""

from __future__ import annotations

import torch


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Computes the share of correct predictions.

    Args:
        predictions: Predicted class indices, shape (n,).
        targets: True class indices, shape (n,).

    Returns:
        The accuracy.

    Raises:
        ValueError: If the inputs are empty or differ in length.
    """
    matrix = _confusion(predictions, targets)
    return (matrix.trace() / matrix.sum()).item()


def balanced_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Computes recall averaged over the classes that occur in the targets.

    Args:
        predictions: Predicted class indices, shape (n,).
        targets: True class indices, shape (n,).

    Returns:
        The balanced accuracy.

    Raises:
        ValueError: If the inputs are empty or differ in length.
    """
    matrix = _confusion(predictions, targets)
    occurs = matrix.sum(dim=1) > 0
    return (matrix.diagonal()[occurs] / matrix.sum(dim=1)[occurs]).mean().item()


def precision(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    positive: int | None = None
) -> float:
    """Computes how often a predicted class is right.

    Args:
        predictions: Predicted class indices, shape (n,).
        targets: True class indices, shape (n,).
        positive: The class to score. By default, the macro average.

    Returns:
        The precision. A class that is never predicted scores 0.

    Raises:
        ValueError: If the inputs are empty or differ in length.
    """
    matrix = _confusion(predictions, targets, positive)
    return _average(_safe_ratio(matrix.diagonal(), matrix.sum(dim=0)), matrix, positive)


def recall(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    positive: int | None = None
) -> float:
    """Computes how often a true class is found.

    Args:
        predictions: Predicted class indices, shape (n,).
        targets: True class indices, shape (n,).
        positive: The class to score. By default, the macro average.

    Returns:
        The recall. A class that never occurs scores 0.

    Raises:
        ValueError: If the inputs are empty or differ in length.
    """
    matrix = _confusion(predictions, targets, positive)
    return _average(_safe_ratio(matrix.diagonal(), matrix.sum(dim=1)), matrix, positive)


def f1(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    positive: int | None = None
) -> float:
    """Computes the harmonic mean of precision and recall.

    Args:
        predictions: Predicted class indices, shape (n,).
        targets: True class indices, shape (n,).
        positive: The class to score. By default, the macro average.

    Returns:
        The F1 score. A class with no true positives scores 0.

    Raises:
        ValueError: If the inputs are empty or differ in length.
    """
    matrix = _confusion(predictions, targets, positive)
    denominator = matrix.sum(dim=0) + matrix.sum(dim=1)
    return _average(_safe_ratio(2 * matrix.diagonal(), denominator), matrix, positive)


def mcc(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Computes the Matthews correlation coefficient.

    Args:
        predictions: Predicted class indices, shape (n,).
        targets: True class indices, shape (n,).

    Returns:
        The coefficient between -1 and 1. It is 0 when either side holds only one
        class, since no correlation can be measured then.

    Raises:
        ValueError: If the inputs are empty or differ in length.
    """
    matrix = _confusion(predictions, targets)
    actual, predicted = matrix.sum(dim=1), matrix.sum(dim=0)
    total = matrix.sum()
    numerator = matrix.trace() * total - actual @ predicted
    denominator = ((total**2 - predicted @ predicted) * (total**2 - actual @ actual)).sqrt()
    return 0.0 if denominator == 0 else (numerator / denominator).item()


def auroc(probabilities: torch.Tensor, targets: torch.Tensor) -> float:
    """Computes the area under the ROC curve.

    Args:
        probabilities: The probability of class 1, shape (n,), or of each class,
            shape (n, k).
        targets: True class indices, shape (n,).

    Returns:
        The area. With more than two classes, the macro average of one-vs-rest.

    Raises:
        ValueError: If a class has no examples, since the area is undefined then,
            or the probabilities are out of range.
    """
    from torchmetrics.functional.classification import binary_auroc, multiclass_auroc

    probabilities, targets = _check_probabilities(probabilities, targets, every_class=True)
    if probabilities.ndim == 1:
        return binary_auroc(probabilities, targets, thresholds=None).item()
    return multiclass_auroc(
        probabilities,
        targets,
        num_classes=probabilities.shape[1],
        average="macro",
        thresholds=None
    ).item()


def auprc(probabilities: torch.Tensor, targets: torch.Tensor) -> float:
    """Computes the area under the precision-recall curve, as average precision.

    Args:
        probabilities: The probability of class 1, shape (n,), or of each class,
            shape (n, k).
        targets: True class indices, shape (n,).

    Returns:
        The area. With more than two classes, the macro average of one-vs-rest.

    Raises:
        ValueError: If a class has no examples, since the area is undefined then,
            or the probabilities are out of range.
    """
    from torchmetrics.functional.classification import binary_average_precision, multiclass_average_precision

    probabilities, targets = _check_probabilities(probabilities, targets, every_class=True)
    if probabilities.ndim == 1:
        return binary_average_precision(probabilities, targets, thresholds=None).item()
    return multiclass_average_precision(
        probabilities,
        targets,
        num_classes=probabilities.shape[1],
        average="macro",
        thresholds=None
    ).item()


def brier(probabilities: torch.Tensor, targets: torch.Tensor) -> float:
    """Computes the mean squared error of predicted probabilities.

    Args:
        probabilities: The probability of class 1, shape (n,), or of each class,
            shape (n, k).
        targets: True class indices, shape (n,).

    Returns:
        The mean summed squared error, halved for two classes so that it matches the
        usual binary definition, as in scikit-learn.

    Raises:
        ValueError: If the probabilities are out of range.
    """
    probabilities, targets = _check_probabilities(probabilities, targets)
    if probabilities.ndim == 1:
        return ((probabilities - targets) ** 2).mean().item()
    one_hot = torch.nn.functional.one_hot(targets, probabilities.shape[1])
    summed = ((probabilities - one_hot) ** 2).sum(dim=1).mean()
    return (summed / 2 if probabilities.shape[1] == 2 else summed).item()


def ece(probabilities: torch.Tensor, targets: torch.Tensor, *, n_bins: int = 15) -> float:
    """Computes the expected calibration error.

    Args:
        probabilities: The probability of class 1, shape (n,), or of each class,
            shape (n, k).
        targets: True class indices, shape (n,).
        n_bins: How many equal-width confidence bins to use.

    Returns:
        The mean gap between confidence and accuracy, weighted by bin size.

    Raises:
        ValueError: If the probabilities are out of range.
    """
    from torchmetrics.functional.classification import binary_calibration_error, multiclass_calibration_error

    probabilities, targets = _check_probabilities(probabilities, targets)
    if probabilities.ndim == 1:
        return binary_calibration_error(probabilities, targets, n_bins=n_bins, norm="l1").item()
    return multiclass_calibration_error(
        probabilities,
        targets,
        num_classes=probabilities.shape[1],
        n_bins=n_bins,
        norm="l1"
    ).item()


def _confusion(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    positive: int | None = None
) -> torch.Tensor:
    from torchmetrics.functional.classification import multiclass_confusion_matrix

    predictions, targets = _check_lengths(predictions, targets)
    largest = max(predictions.max().item(), targets.max().item(), 0 if positive is None else positive)
    # the matrix function needs two classes, and a class no row uses changes nothing
    matrix = multiclass_confusion_matrix(predictions, targets, num_classes=max(largest + 1, 2))
    return matrix.double()


def _check_lengths(predictions: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if len(predictions) == 0:
        raise ValueError("Can't score empty inputs.")
    if len(predictions) != len(targets):
        raise ValueError(f"Inputs must have the same length, got {len(predictions)} and {len(targets)}.")
    return predictions.detach().cpu(), targets.detach().cpu()


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return torch.where(denominator > 0, numerator / denominator.clamp_min(1), torch.zeros_like(numerator))


def _average(scores: torch.Tensor, matrix: torch.Tensor, positive: int | None) -> float:
    if positive is not None:
        return scores[positive].item()
    present = (matrix.sum(dim=0) + matrix.sum(dim=1)) > 0
    return scores[present].mean().item()


def _check_probabilities(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    every_class: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities, targets = _check_lengths(probabilities, targets)
    if not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("These metrics need probabilities between 0 and 1.")
    if every_class:
        n_classes = 2 if probabilities.ndim == 1 else probabilities.shape[1]
        missing = sorted(set(range(n_classes)) - set(targets.tolist()))
        if missing:
            raise ValueError(f"The area is undefined, because the targets have no examples of class {missing}.")
    return probabilities, targets
