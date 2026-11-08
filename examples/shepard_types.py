"""Runs category learning on the six Shepard, Hovland and Jenkins (1961) structures.

People learn type I fastest, then type II, then types III to V, and type VI slowest
(Nosofsky et al., 1994). This prints the probability of an error per block and type,
each type's mean, and the rank correlation between the types' means and the human
order, then saves the curves as CSV.

Example: python examples/shepard_types.py --model gpt2 --device auto
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

import cognarium as cg

TYPES = ("I", "II", "III", "IV", "V", "VI")
# the human order, with types III to V tied, as Nosofsky et al. (1994) found them
HUMAN_RANK = {
    "I": 1.0,
    "II": 2.0,
    "III": 4.0,
    "IV": 4.0,
    "V": 4.0,
    "VI": 6.0
}
# how many of the 70 balanced labellings have each structure
GROUP_SIZES = {
    "I": 6,
    "II": 6,
    "III": 24,
    "IV": 8,
    "V": 24,
    "VI": 2
}
# Boolean complexity, the length of each type's shortest formula (Feldman, 2000)
COMPLEXITY = {
    "I": 1,
    "II": 4,
    "III": 6,
    "IV": 6,
    "V": 6,
    "VI": 10
}


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Reads the command line.

    Args:
        argv: The arguments, or None to read them from the command line.

    Returns:
        The parsed settings.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="gpt2", help="A Hugging Face model, 'exemplar' or 'bayesian'.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--participants", type=int, default=10, help="Learners per type.")
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=1, help="Times each object appears in a block.")
    parser.add_argument("--feedback", default="trial", choices=["trial", "block"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/shepard_types"))
    return parser.parse_args(argv)


def subject_for(name: str, device: str, dtype: str) -> cg.subjects.Subject:
    """Builds the subject the command line names.

    Args:
        name: A Hugging Face model, `exemplar` or `bayesian`.
        device: Where a neural model runs.
        dtype: A neural model's precision.

    Returns:
        The subject. The Bayesian learner considers the 70 labellings with four
        objects per category, with a prior that falls by half for each unit of a
        structure's Boolean complexity, so it prefers simple rules.
    """
    if name == "exemplar":
        return cg.subjects.ExemplarLearner()
    if name == "bayesian":
        prior = cg.tasks.ShepardCategories().type_prior({
            kind: 0.5 ** COMPLEXITY[kind] * GROUP_SIZES[kind]
            for kind in TYPES
        })
        return cg.subjects.BayesianLearner(prior=prior, decision="average")
    return cg.load(name, device=device, dtype=dtype)


def rank_correlation(first: Sequence[float], second: Sequence[float]) -> float:
    """Gives Spearman's correlation, with tied values sharing their mean rank.

    Args:
        first: One set of values.
        second: The other, in the same order.

    Returns:
        The correlation, or NaN when either side is constant.
    """
    ranks = pd.DataFrame({"first": first, "second": second}).rank()
    if ranks["first"].nunique() < 2 or ranks["second"].nunique() < 2:
        return float("nan")
    return float(np.corrcoef(ranks["first"], ranks["second"])[0, 1])


def main(argv: Sequence[str] | None = None) -> pd.DataFrame:
    """Runs every type, then prints and saves the learning curves.

    Args:
        argv: The command line arguments, or None to read them from the command line.

    Returns:
        The mean probability of an error, one row per type and block.
    """
    args = parse_args(argv)
    subject = subject_for(args.model, args.device, args.dtype)
    curves = []
    for name in TYPES:
        paradigm = cg.paradigms.CategoryLearning(
            cg.tasks.ShepardCategories(type=name),
            blocks=args.blocks,
            repeats=args.repeats,
            feedback=args.feedback,
            n_participants=args.participants,
            seed=args.seed
        )
        results = cg.run(paradigm, subject, out=args.out / f"type_{name}", progress=False)
        curve = results.metrics("answer_probability")
        curves.append(pd.DataFrame({"type": name, "block": curve["block"], "error": 1.0 - curve["value"]}))
    curves = pd.concat(curves, ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    curves.to_csv(args.out / "errors.csv", index=False)
    print(curves.pivot(index="block", columns="type", values="error")[list(TYPES)].round(3).to_string())
    means = curves.groupby("type")["error"].mean()[list(TYPES)]
    print("\nmean error: " + ", ".join(f"{name} {value:.3f}" for name, value in means.items()))
    correlation = rank_correlation(means.tolist(), [HUMAN_RANK[name] for name in TYPES])
    print(f"rank correlation with the human order: {correlation:.2f}")
    return curves


if __name__ == "__main__":
    main()
