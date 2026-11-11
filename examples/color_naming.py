"""Runs in-context iterated learning of colour names, as in Imel and Zaslavsky (2026).

Each chain passes a naming of the 330 World Color Survey chips down 13 generations.
The first naming pairs each chip with one of k made-up words at random. Every
generation then sees 6k chips named by the last one, shown as RGB values or, with
--show image, as coloured chips to a vision-language model, and names all 330
chips one at a time, still seeing its own 10 latest answers. The
paper's chains drifted towards efficient systems, near the information bottleneck
frontier of Zaslavsky et al. (2018). This prints and saves, per generation, each
system's complexity, accuracy, efficiency loss and alignment with the nearest
optimal system, with the number of words in use, where generation 0 is the start.

Example: python examples/color_naming.py --model Qwen/Qwen3.5-4B-Base --device cuda --dtype bfloat16 --words 3
Example: python examples/color_naming.py --model Qwen/Qwen3.5-4B-Base --show image --chip-pixels 1024 --words 3
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import torch

import cognarium as cg
from cognarium.metrics import naming

WORDS = (
    "Narp",
    "Tovo",
    "Feglu",
    "Mib",
    "Blim",
    "Zarn"
)
MEASURES = (
    "ib_complexity",
    "ib_accuracy",
    "efficiency_loss",
    "ib_alignment"
)
TEMPLATES = {
    "text": cg.PromptTemplate(stimulus="Color: {text} -> Label:", response=""),
    "image": cg.PromptTemplate(stimulus="Color: {image} -> Label:", response="")
}


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
    parser.add_argument("--words", type=int, nargs="+", default=[2, 3, 4, 5, 6], help="Vocabulary sizes, 2 to 6.")
    parser.add_argument("--chains", type=int, default=4, help="Chains per vocabulary size. The paper had 4.")
    parser.add_argument("--generations", type=int, default=13, help="Learners per chain. The paper had 13.")
    parser.add_argument("--memory", type=int, default=10, help="Own latest answers each learner still sees.")
    parser.add_argument("--show", choices=sorted(TEMPLATES), default="text", help="How each chip is shown.")
    parser.add_argument("--chip-size", type=int, default=32, help="The side of each chip image in pixels.")
    parser.add_argument(
        "--chip-pixels",
        type=int,
        default=None,
        help="The pixels a Qwen-style processor resizes each chip to. 1024 gives one token per chip."
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--sigma", type=float, default=8.0, help="Meaning width in CIELAB units.")
    parser.add_argument("--chips", type=Path, default=None, help="A local copy of the WCS chip table.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/color_naming"))
    args = parser.parse_args(argv)
    if not set(args.words) <= set(range(2, len(WORDS) + 1)):
        parser.error(f"--words takes sizes from 2 to {len(WORDS)}.")
    return args


def instructions_for(words: Sequence[str], show: str) -> str:
    """Writes the instructions, adapted from Imel and Zaslavsky (2026) for a base model.

    Args:
        words: The labels on offer.
        show: How each chip is shown, `"text"` or `"image"`.

    Returns:
        The text placed before each prompt.
    """
    color = "shown as a small colored square" if show == "image" else (
        "given by its red, green and blue values from 0 to 1"
    )
    return (
        f"Each line pairs a color, {color}, with a label from this list: {', '.join(words)}. Based on the "
        "preceding examples, give the label that best describes the last color, as exactly one word from the list."
    )


def start_measures(task: cg.tasks.Mapping, chains: int, seed: int) -> pd.DataFrame:
    """Measures each chain's random first naming, as generation 0.

    Args:
        task: The task the chains transmit.
        chains: How many chains there are.
        seed: The run's seed, which draws each chain's first naming.

    Returns:
        One row per chain and measure, with columns `generation`, `chain`,
        `metric` and `value`.
    """
    points, sigma, prior = task.meaning_space()
    meanings = naming.meanings(points, sigma)
    frontier = naming.ib_frontier(prior, meanings)
    rows = []
    for chain in range(chains):
        labels = task.start(chain, seed)
        encoder = torch.zeros(len(labels), len(task.options), dtype=torch.float64)
        encoder[range(len(labels)), [task.options.index(label) for label in labels]] = 1.0
        loss, beta = naming.efficiency_loss(encoder, prior, meanings, frontier)
        values = {
            "ib_complexity": naming.ib_complexity(encoder, prior),
            "ib_accuracy": naming.ib_accuracy(encoder, prior, meanings),
            "efficiency_loss": loss,
            "ib_alignment": 1.0 - naming.gnid(encoder, frontier.encoder_at(beta), prior),
            "terms": float(len(set(labels)))
        }
        rows += [{"generation": 0, "chain": chain, "metric": name, "value": value} for name, value in values.items()]
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> pd.DataFrame:
    """Runs a set of chains for each vocabulary size, then prints and saves the measures.

    Args:
        argv: The command line arguments, or None to read them from the command line.

    Returns:
        One row per vocabulary size, generation, chain and measure, with the
        number of words in use as the measure `terms`.
    """
    args = parse_args(argv)
    if args.model == "exemplar":
        subject = cg.subjects.ExemplarLearner()
    elif args.show == "image":
        pixels = args.chip_pixels
        subject = cg.load(
            args.model,
            device=args.device,
            dtype=args.dtype,
            multimodal=True,
            processor_kwargs=None if pixels is None else {
                "images_kwargs": {"size": {"shortest_edge": pixels, "longest_edge": pixels}}
            }
        )
    else:
        subject = cg.load(args.model, device=args.device, dtype=args.dtype)
    chips = cg.data.load("wcs_chips", accept_terms=True, path=args.chips).to_pandas()
    lab = chips[["L", "a", "b"]].to_numpy(dtype=float)
    rgb = cg.media.lab_to_srgb(lab).round(3)
    images = [cg.media.swatch(color, size=args.chip_size) for color in rgb] if args.show == "image" else None
    tables = []
    for size in args.words:
        words = WORDS[:size]
        task = cg.tasks.Mapping(
            rgb.tolist(),
            words,
            images=images,
            template=TEMPLATES[args.show],
            points=lab.tolist(),
            sigma=args.sigma,
            name=f"wcs_{size}_words"
        )
        paradigm = cg.paradigms.IteratedLearning(
            task,
            generations=args.generations,
            chains=args.chains,
            n_prev_examples=6 * size,
            memory=args.memory,
            instructions=None if args.model == "exemplar" else instructions_for(words, args.show),
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed
        )
        results = cg.run(paradigm, subject, out=args.out / f"words_{size}")
        measured = results.metrics(*MEASURES, by=["generation", "chain"])
        answers = results.to_pandas()
        terms = answers.groupby(["generation", "chain"])["response"].nunique().rename("value").reset_index()
        learned = pd.concat([measured, terms.assign(metric="terms")], ignore_index=True)
        learned["generation"] += 1
        table = pd.concat([start_measures(task, args.chains, args.seed), learned], ignore_index=True)
        tables.append(table[["generation", "chain", "metric", "value"]].assign(words=size))
    measures = pd.concat(tables, ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    measures.to_csv(args.out / "measures.csv", index=False)
    means = measures.groupby(["words", "generation", "metric"])["value"].mean().unstack("metric")
    print(means[[*MEASURES, "terms"]].round(3).to_string())
    return measures


if __name__ == "__main__":
    main()
