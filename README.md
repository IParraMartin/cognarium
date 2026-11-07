# cognarium

Run cognitive science experiments on language models and other learners. The same
paradigm and the same scoring work on a Hugging Face model, an LSTM, an exemplar
learner and a Bayesian learner, on CPU, MPS or CUDA.

This is an early version. It has iterated learning on Shepard category structures,
full-vocabulary logits, the logit lens, hidden-state datasets, resumable runs, and
information, classification, signal detection, geometry and statistics metrics. More
paradigms and dataset downloads come later.

## Contents

- `src/cognarium/`: the library
  - `paradigms/`: experiments, written as a function from past results to the next step
  - `tasks/`: what a paradigm presents, such as the Shepard structures
  - `subjects/`: Hugging Face models, custom modules, exemplar and Bayesian learners
  - `models/`: the bundled LSTM and its tokenizer
  - `recording/`: layer maps, state capture, the state store and the logit lens
  - `metrics/`: plain metric functions and the measures behind `Results.metrics`
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
    record=["logits", "hidden_states"]
)

results.metrics("accuracy", "entropy", "surprisal")
results.logits(0)
results.to_pandas()
results.lens()
results.hidden.save("il_gpt2/")

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

## Hidden states

`record=["hidden_states"]` stores one row per query for each layer, as float16 by
default. `layers`, `pool` (`"last"`, `"mean"` or None for every token) and `state_dtype`
choose what is kept. Exemplar learners store their similarities and Bayesian learners
their posteriors in the same format, so any two can be compared.

```python
store = cg.HiddenStates.open("il_gpt2/")
states = {layer: store.to_tensor(layer) for layer in store.layers}
cg.metrics.layer_matrix(states, "linear_cka")
store.push_to_hub("you/il_gpt2")
```

`results.lens()` reads each layer through the model's own final norm and output head. A
model whose last block doesn't reproduce its own logits that way is refused, as when it
rescales them after the head.
Models outside the known architectures can be described with
`cg.load(model, layer_map=cg.LayerMap(...))`, and custom modules with
`cg.load(module, tokenizer=tok, layers={"name": "path"}, readout="path")`.

## Metrics

Every metric is a plain function in `cg.metrics`. Those that need `float64`, or that have
known MPS problems, always compute on CPU. Many can also be named in `results.metrics(...)`.

| Family | Functions | Takes |
|---|---|---|
| Information | `entropy`, `conditional_entropy`, `joint_entropy`, `mutual_information`, `ksg_mutual_information`, `kl_divergence`, `js_divergence`, `cross_entropy`, `surprisal`, `perplexity`, `information_bottleneck` | Log probabilities, or samples for KSG. Results are in bits by default |
| Classification | `accuracy`, `balanced_accuracy`, `precision`, `recall`, `f1`, `mcc`, `auroc`, `auprc`, `brier`, `ece` | Predicted labels or probabilities, and target labels |
| Signal detection | `d_prime`, `criterion`, `a_prime` | Hits, misses, false alarms and correct rejections |
| Geometry | `linear_cka`, `rbf_cka`, `rsa`, `procrustes`, `mse`, `svcca`, `pwcca`, `participation_ratio`, `intrinsic_dimension`, `layer_matrix` | State matrices with one row per sample. `layer_matrix` takes a dictionary of them |
| Statistics | `cohens_d`, `hedges_g`, `bootstrap_ci`, `permutation_test`, `fdr` | Values per group, or p values for `fdr` |

Undefined inputs raise an error instead of returning NaN or a silent 0, such as AUROC
with one class or CKA on a matrix with no variance.

## Reproducing

```bash
uv run python examples/iterated_learning.py --model gpt2 --device auto
uv run pytest
```

Seeds derive per chain, generation and item, so sampling doesn't depend on batch size.
Exact numeric equality across devices isn't guaranteed. Each run's `manifest.json`
holds its settings, seeds, library versions, model revision, and the device and batch size of each session.
