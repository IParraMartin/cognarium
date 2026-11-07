# cognarium

Run cognitive science experiments on language models and other learners. The same
paradigm and the same scoring work on a Hugging Face model, an LSTM, an exemplar
learner and a Bayesian learner, on CPU, MPS or CUDA.

This is an early version. It has iterated learning on Shepard category structures,
full-vocabulary logits, resumable runs and the metrics iterated learning needs. More
paradigms, the logit lens, hidden-state storage and dataset downloads come later.

## Contents

- `src/cognarium/`: the library
  - `paradigms/`: experiments, written as a function from past results to the next step
  - `tasks/`: what a paradigm presents, such as the Shepard structures
  - `subjects/`: Hugging Face models, custom modules, exemplar and Bayesian learners
  - `models/`: the bundled LSTM and its tokenizer
  - `metrics/`: information theory functions and the measures behind `Results.metrics`
  - `runner.py`: `run`, which saves each step as it finishes
  - `results.py`: `Results`, which reads a run back
- `examples/`: runnable scripts
- `tests/`: the test suite, including the exact stationary distribution checks

## Install

From a clone of this repository:

```bash
uv pip install .
```

For development:

```bash
uv sync --extra dev
```

## Quickstart

```python
import cognarium as cg

subject = cg.load("gpt2", device="auto")

results = cg.run(
    cg.paradigms.IteratedLearning(
        task=cg.tasks.ShepardCategories(type="VI"),
        generations=10,
        chains=50,
        instructions="Sort each object into category A or B.",
        examples=[("small red circle", "A")],
        temperature=0.7,
        top_p=0.9
    ),
    subject,
    learning="in_context",
    record=["logits"]
)

results.metrics("accuracy", "entropy", "surprisal")
results.logits(0)
results.to_pandas()

bayes = cg.subjects.BayesianLearner()
cg.run(
    cg.paradigms.IteratedLearning(
        task=cg.tasks.ShepardCategories(type="VI"),
        generations=10
    ),
    bayes,
    learning="posterior"
).metrics("entropy", "kl")
```

Each run writes to `runs/<paradigm>/<key>/`. Running the same call again continues after
the last saved step, and a finished run makes no further model calls. A folder that
holds a different run is refused, with a message naming what changed, and nothing is
deleted.

Temperature, top-k and top-p shape the response distribution the same way on every
subject. Raw scores are always recorded next to the shaped ones.

## Reproducing

```bash
uv run python examples/iterated_learning.py --model gpt2 --device auto
uv run pytest
```

Seeds derive per chain, generation and item, so sampling doesn't depend on batch size.
Exact numeric equality across devices isn't guaranteed. Each run's `manifest.json`
holds its settings, seeds, library versions, model revision, and the device and batch size of each session.
