"""Runs every built-in paradigm on one model and writes each one's score table as a Markdown page.

Each page reports what the run gave, whether or not it looks like the human effect, and
nothing is tuned or filtered. DRM uses the small lists written below, since its
paradigm needs lists from you. Rerunning resumes finished runs. The tour also writes
one trial and the scores of each run for the screen on the docs' home page.

Example: python examples/paradigm_tour.py --model gpt2 --device auto --out runs/tour --docs docs/results
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

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
# the columns where a paradigm keeps what it read out of a free response
READ_COLUMNS = ("parsed", "recalled", "produced")
SCREEN_START = "window.COGNARIUM_RUNS = "


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
        "--docs",
        type=Path,
        default=Path("docs/results"),
        help="Where to write the pages."
    )
    parser.add_argument(
        "--screen",
        type=Path,
        default=Path("docs/javascripts/paradigm-runs.js"),
        help="Where to write the data for the home page's screen."
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


def page(
    paradigm: cg.Paradigm,
    results: cg.Results,
    model: str,
    device: str
) -> str:
    """Writes the Markdown a paradigm's docs section includes under "A real run".

    Args:
        paradigm: The paradigm that ran.
        results: Its results.
        model: The model's name, as it was loaded.
        device: Where the model ran.

    Returns:
        The model, device and date, the settings, the first prompt, and the score table,
        or for iterated learning, which declares no effects, its metrics by generation.
    """
    first = paradigm.step(0, None).queries[0]
    settings = {
        key: value
        for key, value in paradigm.config().items()
        if key not in ("name", "trials") and value not in (None, [], {})
    }
    options = first.trial.options
    if options:
        offered = "with the options " + ", ".join(f"`{option}`" for option in options)
    else:
        offered = "answered by free generation"
    day = datetime.date.today().isoformat()
    lines = [
        f"Model `{model}` on `{device}`, run on {day} with cognarium {cg.__version__}.",
        "",
        f"Settings: `{json.dumps(settings, sort_keys=True, default=str)}`",
        "",
        f"Sample prompt, {offered}:",
        "",
        "```text",
        first.prompt(),
        "```",
        "",
        markdown(summary(paradigm, results)),
        ""
    ]
    return "\n".join(lines)


def summary(paradigm: cg.Paradigm, results: cg.Results) -> pd.DataFrame:
    """Gives the table a paradigm's page reports.

    Args:
        paradigm: The paradigm that ran.
        results: Its results.

    Returns:
        The score table, or the chain metrics for a paradigm that declares no effects.
    """
    return results.score() if paradigm.effects else results.metrics(*CHAIN_METRICS)


def screen_entry(
    name: str,
    paradigm: cg.Paradigm,
    results: cg.Results
) -> dict[str, Any]:
    """Describes a run's first trial and its scores, for the screen on the docs' home page.

    Args:
        name: The paradigm's name in `cg.paradigms.PARADIGMS`.
        paradigm: The paradigm that ran.
        results: Its results.

    Returns:
        The paradigm's name, its docs page and anchor, the first line of its docstring,
        the first prompt, its options with their probabilities and the answer, or the
        free response and what the paradigm read out of it, and the score rows. For
        iterated learning, which declares no effects, the accuracy and fidelity of each
        generation and the number of chains take the place of the scores.
    """
    first = results.to_pandas().iloc[0]
    entry: dict[str, Any] = {
        "name": name,
        "page": type(paradigm).__module__.rsplit(".", 1)[-1],
        "anchor": type(paradigm).__name__.lower(),
        "task": (type(paradigm).__doc__ or "").strip().splitlines()[0],
        "prompt": paradigm.step(0, None).queries[0].prompt(),
        "answer": None if first["answer"] is None else str(first["answer"])
    }
    if len(first["options"]):
        entry["options"] = [str(option) for option in first["options"]]
        entry["probabilities"] = [math.exp(value) for value in first["log_probs"]]
    else:
        entry["response"] = str(first["response"])
        column = next(column for column in READ_COLUMNS if column in first.index)
        entry["read"] = [str(item) for item in first[column]]
    if paradigm.effects:
        entry["effects"] = [
            {key: _json(value) for key, value in row.items()}
            for row in results.score().to_dict("records")
        ]
    else:
        metrics = results.metrics("accuracy", "fidelity")
        entry["series"] = {
            metric: metrics[metrics["metric"] == metric].sort_values("generation")["value"].tolist()
            for metric in ("accuracy", "fidelity")
        }
        entry["chains"] = paradigm.config()["chains"]
    return entry


def write_screen(path: Path, entries: dict[str, dict[str, Any]]) -> None:
    """Writes the screen's data, keeping the entries of paradigms that didn't run this time.

    Args:
        path: The script to write, which sets `window.COGNARIUM_RUNS`.
        entries: The new entries, by paradigm name.
    """
    kept = {}
    if path.exists():
        text = path.read_text(encoding="utf-8")
        kept = {entry["name"]: entry for entry in json.loads(text[text.index("["):text.rindex("]") + 1])}
    kept.update(entries)
    ordered = [kept[name] for name in cg.paradigms.PARADIGMS if name in kept]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "/* One trial and the scores of each paradigm's run, written by examples/paradigm_tour.py. */\n"
        f"{SCREEN_START}{json.dumps(ordered, indent=4, ensure_ascii=False)};\n",
        encoding="utf-8"
    )


def markdown(table: pd.DataFrame) -> str:
    """Writes a table as Markdown, with numbers to four significant digits.

    Args:
        table: The table.

    Returns:
        The Markdown table.
    """
    cells = [[_cell(value) for value in row] for row in table.itertuples(index=False)]
    header = "| " + " | ".join(str(column) for column in table.columns) + " |"
    rule = "|" + "|".join("---" for _ in table.columns) + "|"
    body = ["| " + " | ".join(row) + " |" for row in cells]
    return "\n".join([header, rule, *body])


def main(argv: Sequence[str] | None = None) -> dict[str, cg.Results]:
    """Runs the chosen paradigms and writes their pages.

    Args:
        argv: The command line, or None to read it.

    Returns:
        Each paradigm's results, by name.
    """
    args = parse_args(argv)
    names = args.only or sorted(cg.paradigms.PARADIGMS)
    subject = cg.load(args.model, device=args.device)
    device = str(getattr(subject, "device", args.device))
    args.docs.mkdir(parents=True, exist_ok=True)
    found = {}
    entries = {}
    for name in names:
        paradigm = build(name)
        results = cg.run(
            paradigm,
            subject,
            out=args.out / name,
            progress=False
        )
        (args.docs / f"{name}.md").write_text(page(
            paradigm,
            results,
            args.model,
            device
        ), encoding="utf-8")
        table = summary(paradigm, results)
        if "detected" in table:
            print(f"{name}: {len(table)} effects, {int(table['detected'].fillna(False).astype(bool).sum())} detected")
        else:
            print(f"{name}: {len(table)} rows of chain metrics")
        found[name] = results
        entries[name] = screen_entry(name, paradigm, results)
    write_screen(args.screen, entries)
    return found


def _json(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    return value.item() if hasattr(value, "item") else value


def _cell(value: Any) -> str:
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.4g}"
    return "" if value is None else str(value).replace("|", "\\|")


if __name__ == "__main__":
    main()
