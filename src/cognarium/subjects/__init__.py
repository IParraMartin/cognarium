"""Subjects: anything that can give probabilities over responses."""

from __future__ import annotations

from cognarium.subjects.base import Subject
from cognarium.subjects.bayesian import BayesianLearner
from cognarium.subjects.custom import LSTM, CustomSubject
from cognarium.subjects.exemplar import ExemplarLearner
from cognarium.subjects.hf import HFSubject

__all__ = [
    "BayesianLearner",
    "CustomSubject",
    "ExemplarLearner",
    "HFSubject",
    "LSTM",
    "Subject"
]
