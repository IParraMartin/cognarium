# Metrics

Every metric is a plain function in `cg.metrics`, usable without running an experiment.
Information measures are in bits by default. Metrics that need `float64`, or that have
known problems on MPS, always compute on the CPU. An undefined input raises an error
instead of returning NaN or a silent 0, such as AUROC with one class, or a correlation
with a constant side.

```python
import cognarium as cg
import torch

log_probs = torch.log_softmax(torch.randn(8, 4, generator=torch.Generator().manual_seed(0)), dim=-1)
cg.metrics.entropy(log_probs)
```

| Family | Functions |
|---|---|
| Information | `entropy`, `conditional_entropy`, `joint_entropy`, `mutual_information`, `ksg_mutual_information`, `kl_divergence`, `js_divergence`, `cross_entropy`, `surprisal`, `perplexity`, `information_bottleneck` |
| Classification | `accuracy`, `balanced_accuracy`, `precision`, `recall`, `f1`, `mcc`, `auroc`, `auprc`, `brier`, `ece` |
| Signal detection | `d_prime`, `criterion`, `a_prime` |
| Geometry | `linear_cka`, `rbf_cka`, `rsa`, `procrustes`, `mse`, `svcca`, `pwcca`, `participation_ratio`, `intrinsic_dimension`, `layer_matrix` |
| Statistics | `cohens_d`, `hedges_g`, `bootstrap_ci`, `permutation_test`, `fdr` |
| Human comparison | `item_correlation`, `split_half_ceiling`, `effect_comparison` |

## On results

`results.metrics(...)` computes named metrics over a run, grouped by the paradigm's
conditions, and `results.contrast(...)` compares one measure between two levels with a
bootstrap interval and a permutation test.

```python
subject = cg.load("gpt2", device="auto")
results = cg.run(cg.paradigms.NBack(), subject)
results.metrics("accuracy", "entropy")
```

```python
results.metrics("d_prime", positive="yes")
```

## Geometry

The geometry metrics take state matrices with one row per sample. `layer_matrix`
compares every pair of layers.

```python
x = torch.randn(50, 16, generator=torch.Generator().manual_seed(1))
rotation, _ = torch.linalg.qr(torch.randn(16, 16, generator=torch.Generator().manual_seed(2)))
cg.metrics.linear_cka(x, x @ rotation)
```

## Comparing with people

`item_correlation` correlates a model's item values with people's, with a bootstrap
interval over items. `split_half_ceiling` estimates how well any model could do, from
the agreement between random halves of the participants, corrected by the
Spearman-Brown formula. `effect_comparison` standardises the same effect in the model and
in people, since their scales differ, and reports the difference with an interval.

```python
import numpy as np

rng = np.random.default_rng(0)
pattern = rng.normal(size=30)
ratings = pattern + rng.normal(size=(20, 30))
model = pattern + 0.5 * rng.normal(size=30)

ceiling = cg.metrics.split_half_ceiling(ratings)
cg.metrics.item_correlation(model, ratings.mean(axis=0))["correlation"], ceiling["ceiling"]
```

```python
cg.metrics.effect_comparison(
    rng.normal(1.0, 1.0, size=40),
    rng.normal(0.0, 1.0, size=40),
    rng.normal(0.3, 1.0, size=40),
    rng.normal(0.0, 1.0, size=40)
)
```

`results.compare_to_human(human, measure)` does the item correlation straight from a run,
as the [quickstart](quickstart.md) shows.
