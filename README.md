# cognarium

Run cognitive science experiments on language models and other learners. A standard
experiment takes a few lines: cognarium builds the trials, runs them through the model,
scores the effect the paradigm is known for, and saves everything needed to reproduce
it. The same paradigms work on Hugging Face models, LSTMs, exemplar learners and
Bayesian learners, on CPU, MPS or CUDA.

## Status

This is version 0.1.0. It has three learning paradigms: in-context iterated learning
and category learning on any category system or function, and artificial grammar
learning with judgment and string-probability tests. Each also runs on exemplar and
Bayesian learners as baselines. It has CSV and JSON conversion, a registry of open
datasets pinned by checksum, comparison with human data, per-trial logits, the logit
lens, hidden-state datasets, resumable runs, and information, classification, signal
detection, geometry, statistics and naming metrics. Batching, quantization and
multi-GPU sweeps come next, then speech, vision and perception paradigms.

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
paradigm's section says how it maps onto the human task and what it can't capture.

## Contents

- `src/cognarium/`: the library
  - `paradigms/`: the built-in experiments, grouped by what they measure
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
- `examples/`: runnable scripts: a tour that runs every paradigm on one model, the six
  Shepard types against the human order, grammar learning with both tests, and a
  reproduction of iterated function learning (Kalish, Griffiths and Lewandowsky, 2007)
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
