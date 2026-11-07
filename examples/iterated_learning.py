"""Runs iterated learning on a Hugging Face model, an LSTM, an exemplar learner and a Bayesian learner.

Writes each run under runs/ and prints per-generation metrics. Rerunning resumes finished work.

Example: python examples/iterated_learning.py --model gpt2 --device auto
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import cognarium as cg

METRICS = (
    "accuracy",
    "fidelity",
    "entropy",
    "surprisal",
    "chain_entropy",
    "kl"
)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Reads the command line.

    Args:
        argv: The arguments, or None to read them from the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="gpt2", help="A Hugging Face model name or local folder.")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps or cuda.")
    parser.add_argument("--chains", type=int, default=50)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to put the runs. Defaults to runs/."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, cg.Results]:
    """Runs the experiment on all four subject types.

    Args:
        argv: The command line arguments, or None to read them from the command line.

    Returns:
        The results of each run, under `hf`, `lstm`, `exemplar` and `bayesian`.
    """
    args = parse_args(argv)
    task = cg.tasks.ShepardCategories(type="VI")
    size = {"task": task, "generations": args.generations, "chains": args.chains}
    reading = cg.paradigms.IteratedLearning(
        **size,
        instructions="Sort each object into category A or B.",
        temperature=0.7,
        top_p=0.9
    )
    plain = cg.paradigms.IteratedLearning(**size)
    runs = {
        "hf": (reading, cg.load(args.model, device=args.device), ["logits"]),
        "lstm": (reading, cg.subjects.LSTM(device=args.device), []),
        "exemplar": (plain, cg.subjects.ExemplarLearner(device=args.device), []),
        "bayesian": (plain, cg.subjects.BayesianLearner(device=args.device), [])
    }
    results = {}
    for name, (paradigm, subject, record) in runs.items():
        out = None if args.out is None else args.out / name
        results[name] = cg.run(
            paradigm,
            subject,
            record=record,
            out=out
        )
        table = results[name].metrics(*METRICS)
        print(f"\n{name}\n{table.pivot(index='generation', columns='metric', values='value').to_string()}")
    return results


if __name__ == "__main__":
    main()
