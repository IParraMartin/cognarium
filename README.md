# cognarium

Run cognitive science experiments on language models and other learners. A standard
experiment takes a few lines: cognarium builds the trials, runs them through the model,
scores the effect the paradigm is known for, and saves everything needed to reproduce
it. The same paradigms work on Hugging Face models, LSTMs, exemplar learners and
Bayesian learners, on CPU, MPS or CUDA.

## Status

This is version 0.1.0. It has three learning paradigms, each of which also runs on
exemplar and Bayesian learners as baselines. It has CSV and JSON conversion, a registry
of open datasets pinned by checksum, comparison with human data, per-trial logits, the
logit lens, hidden-state datasets, resumable runs, and information, classification,
signal detection, geometry, statistics and naming metrics. Quantization and multi-GPU
sweeps come next.

## Experiments

| Paradigm | What it does | Built on |
|---|---|---|
| `IteratedLearning` | Each generation learns from the previous generation's answers and passes its own on, for the Shepard structures or any category system or function | Kirby, Cornish and Smith (2008); Kalish, Griffiths and Lewandowsky (2007); Griffiths and Kalish (2007); Imel and Zaslavsky (2026) |
| `CategoryLearning` | Learns the Shepard structures, any category system or a function from corrective feedback, block by block | Shepard, Hovland and Jenkins (1961); Nosofsky et al. (1994) |
| `ArtificialGrammar` | Studies strings from a finite-state grammar, then judges new ones, with grammaticality and chunk strength crossed, or compares them by string probability | Reber (1967); Knowlton and Squire (1996); Hu and Levy (2023) |

The [paradigms page](docs/paradigms/learning.md) explains each one, how it maps onto the
human task, what it can't capture, and results on GPT-2 and Qwen3.5-4B-Base, with full
references.

## Install

From a clone of this repository:

```bash
uv pip install .
```

For development, and to build the docs:

```bash
uv sync --extra dev --extra docs
uv run mkdocs serve
```

## Quickstart

```python
import cognarium as cg

subject = cg.load("gpt2", device="auto")
results = cg.run(cg.paradigms.ArtificialGrammar(), subject)

results.score()
results.to_pandas()
results.metrics("accuracy", "entropy")
```

`score()` gives one row per effect the paradigm declares, with a bootstrap interval over
items and a permutation test. Where a measure stands in for a human one, such as
surprisal for reaction time, the table names it in `proxy_for`.

Each run writes to `runs/<paradigm>/<key>/`. Running the same call again continues after
the last saved step, and a folder that holds a different run is refused.

## Documentation

The [docs](docs/index.md) cover installing, a longer quickstart, the concepts, writing
your own paradigm, recording hidden states, metrics, data and reproducibility. Each
paradigm's section says how it maps onto the human task, the work it builds on, and
what it can't capture.

## Contents

- `src/cognarium/`: the library
  - `paradigms/`: the base `Paradigm` class and the three built-in experiments
  - `tasks/`: what a paradigm presents, such as the Shepard structures or any mapping
    from items to a fixed set of labels
  - `subjects/`: Hugging Face models, custom modules, exemplar and Bayesian learners
  - `models/`: the bundled LSTM and its tokenizer
  - `recording/`: layer maps, state capture, the state store and the logit lens
  - `metrics/`: plain metric functions, human comparison, and the measures behind
    `Results.metrics`
  - `data/`: CSV and JSON conversion, the dataset registry and its downloads
  - `runner.py`: `run`, which saves each step as it finishes
  - `results.py`: `Results`, which reads a run back
- `examples/`: runnable scripts: a tour that runs every built-in paradigm on one model,
  iterated learning on every kind of subject, the six Shepard types against the human
  order, grammar learning with both tests, the iterated function learning design of
  Kalish, Griffiths and Lewandowsky (2007), and in-context iterated learning of colour
  names on the World Color Survey chips (Imel and Zaslavsky, 2026)
- `docs/`: the documentation site
- `tests/`: the test suite, which also runs every code block in the docs

## Reproducing

```bash
uv run python examples/paradigm_tour.py --model gpt2 --device auto
uv run pytest
```

Every run's `manifest.json` holds its settings, seeds, library versions, model revision,
dataset checksums, and the device and batch size of each session. Exact numeric equality
across devices isn't guaranteed.
