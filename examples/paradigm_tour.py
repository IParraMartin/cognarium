"""Runs every built-in paradigm on one model and prints how many effects each run detected.

DRM uses the small lists written below, since its paradigm needs lists from you.
Rerunning resumes finished runs.

Example: python examples/paradigm_tour.py --model gpt2 --device auto --out runs/tour
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

import cognarium as cg

DRM_LISTS = {
    "cold": (("ice", "snow", "winter", "frost", "chill", "freeze"), "cold", "ladder"),
    "music": (("song", "melody", "piano", "rhythm", "band", "tune"), "music", "carpet"),
    "bread": (("loaf", "bakery", "toast", "dough", "slice", "crust"), "bread", "comet"),
    "ocean": (("wave", "tide", "salt", "shore", "deep", "sailor"), "ocean", "pencil")
}
CHAIN_METRICS = (
    "accuracy",
    "fidelity",
    "entropy",
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
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/tour"),
        help="Where to put the runs."
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=sorted(cg.paradigms.PARADIGMS),
        help="Run only this paradigm. Repeat it to run several."
    )
    return parser.parse_args(argv)


def drm_rows() -> list[dict[str, str]]:
    """Writes the tour's DRM lists as stimulus rows.

    Returns:
        For each list, its studied words, its lure and one unrelated word. The lists were
        written for this tour and are not the published DRM lists.
    """
    rows = []
    for name, (studied, lure, unrelated) in DRM_LISTS.items():
        rows.extend({"list": name, "word": word, "role": "studied"} for word in studied)
        rows.append({"list": name, "word": lure, "role": "lure"})
        rows.append({"list": name, "word": unrelated, "role": "unrelated"})
    return rows


def build(name: str) -> cg.Paradigm:
    """Makes a paradigm with its default settings.

    Args:
        name: The paradigm's name in `cg.paradigms.PARADIGMS`.

    Returns:
        The paradigm. The two learning paradigms that need a category structure get
        Shepard type I for category learning and type VI for iterated learning, and DRM
        gets the tour's lists.
    """
    paradigm = cg.paradigms.PARADIGMS[name]
    if name == "iterated_learning":
        return paradigm(task=cg.tasks.ShepardCategories(type="VI"))
    if name == "category_learning":
        return paradigm(task=cg.tasks.ShepardCategories(type="I"))
    if name == "drm":
        return paradigm(stimuli=drm_rows())
    return paradigm()


def summary(paradigm: cg.Paradigm, results: cg.Results) -> pd.DataFrame:
    """Gives the table the tour reads a run's findings from.

    Args:
        paradigm: The paradigm that ran.
        results: Its results.

    Returns:
        The score table, or the chain metrics for a paradigm that declares no effects.
    """
    return results.score() if paradigm.effects else results.metrics(*CHAIN_METRICS)


def main(argv: Sequence[str] | None = None) -> dict[str, cg.Results]:
    """Runs the chosen paradigms and prints what each run found.

    Args:
        argv: The command line, or None to read it.

    Returns:
        Each paradigm's results, by name.
    """
    args = parse_args(argv)
    names = args.only or sorted(cg.paradigms.PARADIGMS)
    subject = cg.load(args.model, device=args.device)
    found = {}
    for name in names:
        paradigm = build(name)
        results = cg.run(
            paradigm,
            subject,
            out=args.out / name,
            progress=False
        )
        table = summary(paradigm, results)
        if "detected" in table:
            print(f"{name}: {len(table)} effects, {int(table['detected'].fillna(False).astype(bool).sum())} detected")
        else:
            print(f"{name}: {len(table)} rows of chain metrics")
        found[name] = results
    return found


if __name__ == "__main__":
    main()
