"""Reproduces the iterated function learning of Kalish, Griffiths and Lewandowsky (2007).

Four conditions start from y = x, y = 101 - x, a U-shaped curve, or a random pairing of
x and y, with x and y from 1 to 100. In each of 8 chains, every generation sees 50 of
the previous generation's answers as examples and gives a y for all 100 values of x.
The paper's learners drifted to y = x whatever they started from. This prints the
median correlation with y = x per generation, as in the paper's Figure 3F, and saves
every answer and correlation as CSV.

Example: python examples/function_learning.py --model Qwen/Qwen3.5-4B-Base --device cuda --dtype bfloat16
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

import cognarium as cg

INSTRUCTIONS = (
    "Each feature is a number from 1 to 100, and a hidden rule gives it a label, also a "
    "number from 1 to 100. Learn the rule from the examples and give the label of the last feature."
)
VALUES = range(1, 101)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Reads the command line.

    Args:
        argv: The arguments, or None to read them from the command line.

    Returns:
        The parsed settings.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B-Base", help="A Hugging Face model, or 'exemplar'.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--batch-size", type=int, default=100, help="Options scored at once after a prompt.")
    parser.add_argument("--chains", type=int, default=8, help="Families per condition. The paper had 8.")
    parser.add_argument("--generations", type=int, default=9, help="Learners per family. The paper had 9.")
    parser.add_argument("--examples", type=int, default=50, help="Answers passed on to the next generation.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Divides log probabilities before sampling.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Samples only from the likeliest answers covering this mass.")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["positive", "negative", "nonlinear", "random"],
        choices=["positive", "negative", "nonlinear", "random"]
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/function_learning"))
    return parser.parse_args(argv)


def starts(seed: int) -> dict[str, list[int]]:
    """Gives the first generation's y for each x, in each of the paper's conditions.

    Args:
        seed: Seeds the random pairing.

    Returns:
        One list of 100 answers per condition, rounded to whole numbers as in the paper.
    """
    pairing = np.random.default_rng(seed).permutation(len(VALUES)) + 1
    return {
        "positive": list(VALUES),
        "negative": [101 - x for x in VALUES],
        "nonlinear": [round(50.5 + 49.5 * math.sin(math.pi / 2 + x / (5 * math.pi))) for x in VALUES],
        "random": pairing.tolist()
    }


def main(argv: Sequence[str] | None = None) -> pd.DataFrame:
    """Runs every condition, then prints and saves the correlations with y = x.

    Args:
        argv: The command line arguments, or None to read them from the command line.

    Returns:
        The median correlation with y = x, one row per condition and generation, where
        generation 0 is the starting function and generation n the nth learner.
    """
    args = parse_args(argv)
    if args.model == "exemplar":
        subject = cg.subjects.ExemplarLearner()
    else:
        subject = cg.load(args.model, device=args.device, dtype=args.dtype, batch_size=args.batch_size)
    answers, correlations = [], []
    for condition in args.conditions:
        start = starts(args.seed)[condition]
        task = cg.tasks.Mapping(
            list(VALUES),
            [str(y) for y in VALUES],
            start=[str(y) for y in start],
            name=f"function_{condition}"
        )
        paradigm = cg.paradigms.IteratedLearning(
            task,
            generations=args.generations,
            chains=args.chains,
            n_prev_examples=args.examples,
            instructions=None if args.model == "exemplar" else INSTRUCTIONS,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed
        )
        results = cg.run(paradigm, subject, out=args.out / condition)
        table = results.to_pandas()
        answers.append(pd.DataFrame({
            "condition": condition,
            "chain": table["chain"],
            "generation": table["generation"] + 1,
            "x": table["item"] + 1,
            "y": table["response"].astype(int)
        }))
        scored = results.metrics("correlation", reference=lambda x: x, by=["generation", "chain"])
        first = float(np.corrcoef(list(VALUES), start)[0, 1])
        correlations.append(pd.DataFrame({
            "condition": condition,
            "chain": list(range(args.chains)) + scored["chain"].tolist(),
            "generation": [0] * args.chains + (scored["generation"] + 1).tolist(),
            "correlation": [first] * args.chains + scored["value"].tolist()
        }))
    args.out.mkdir(parents=True, exist_ok=True)
    pd.concat(answers).to_csv(args.out / "answers.csv", index=False)
    by_chain = pd.concat(correlations)
    by_chain.to_csv(args.out / "correlations.csv", index=False)
    medians = by_chain.groupby(["condition", "generation"], sort=False)["correlation"].median().reset_index()
    table = medians.pivot(index="generation", columns="condition", values="correlation")
    print(table[args.conditions].round(2).to_string())
    return medians


if __name__ == "__main__":
    main()
