"""Runs artificial grammar learning with both tests, judgment and string probability.

The judgment test asks whether each new string is grammatical and separates the
effect of grammaticality from that of chunk strength, as in Knowlton and Squire
(1996). The probability test compares each grammatical string with a one-letter
ungrammatical change by the probability of each, with no question asked, and runs
only on language models. This prints each test's effects and the judgment test's
probability of answering grammatical by status and chunk strength.

Example: python examples/grammar_learning.py --model gpt2 --device auto
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

import cognarium as cg

COLUMNS = [
    "effect",
    "mean_a",
    "mean_b",
    "difference",
    "ci_low",
    "ci_high",
    "detected"
]


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
    parser.add_argument("--participants", type=int, default=20, help="Learners, each with their own strings.")
    parser.add_argument("--n-train", type=int, default=16, help="Strings each learner studies.")
    parser.add_argument("--n-test", type=int, default=8, help="Grammatical test strings, and as many changed ones.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/grammar_learning"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, pd.DataFrame]:
    """Runs each test the subject can take, then prints what it found.

    Args:
        argv: The command line arguments, or None to read them from the command line.

    Returns:
        Each test's score table, by test name.
    """
    args = parse_args(argv)
    if args.model == "exemplar":
        subject, tests = cg.subjects.ExemplarLearner(criterion=1.0), ["judgment"]
    elif args.model == "bayesian":
        subject, tests = cg.subjects.BayesianLearner(decision="average"), ["judgment"]
    else:
        subject = cg.load(args.model, device=args.device, dtype=args.dtype)
        tests = ["judgment", "probability"]
    scores = {}
    for test in tests:
        paradigm = cg.paradigms.ArtificialGrammar(
            n_train=args.n_train,
            n_test=args.n_test,
            test=test,
            n_participants=args.participants,
            seed=args.seed
        )
        results = cg.run(paradigm, subject, out=args.out / test, progress=False)
        scores[test] = results.score()
        print(f"{test} test")
        print(scores[test][COLUMNS].round(3).to_string(index=False))
        if test == "judgment":
            endorsed = results.metrics("option_probability", option="grammatical")
            table = endorsed.pivot(index="status", columns="chunk", values="value")
            print("\nprobability of answering grammatical")
            print(table.round(3).to_string())
        print()
    return scores


if __name__ == "__main__":
    main()
