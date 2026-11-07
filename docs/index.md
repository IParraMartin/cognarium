# cognarium

cognarium runs cognitive science experiments on language models and other learners. A
standard experiment takes a few lines: it builds the trials, runs them through the
model, scores the effect the paradigm is known for, and saves everything needed to
reproduce it.

```python
import cognarium as cg

subject = cg.load("gpt2", device="auto")
results = cg.run(cg.paradigms.Stroop(), subject)
results.score()
```

The same paradigms and scoring work across kinds of subjects: every paradigm runs on any
Hugging Face language model, and the learning paradigms also run on LSTMs, exemplar
learners and Bayesian learners.

## What it does

- **Paradigms.** 25 built-in paradigms across cognitive control, memory, learning,
  language and judgment, each documented with what it maps onto in the human task and
  what it can't capture. Writing your own takes about 20 lines.
- **Subjects.** Hugging Face models, LSTMs and custom PyTorch modules, exemplar (GCM)
  learners and Bayesian learners, on CPU, Apple MPS or CUDA.
- **Recording.** Per-trial logits, the logit lens, and hidden states stored as a
  shareable dataset.
- **Metrics.** Information theory, classification, signal detection, representational
  geometry, statistics with confidence intervals, and comparisons with human data.
- **Data.** CSV and JSON files become datasets, arrays or trials, and a registry fetches
  open cognitive science datasets from their authors, pinned by checksum.

## Proxies are labelled as proxies

A model has no reaction times. Where a paradigm's human effect is about speed, cognarium
measures something else, such as the surprisal of the correct answer, and every score
table names what that measure stands in for in its `proxy_for` column.

## Where to go next

- [Install](install.md) the library.
- Run a first experiment in the [quickstart](quickstart.md).
- Read the [concepts](concepts.md) behind trials, paradigms and subjects.
- Browse the paradigms, starting with [cognitive control](paradigms/control.md).
