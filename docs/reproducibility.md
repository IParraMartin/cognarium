# Reproducibility

## The manifest

Every run writes `manifest.json` beside its results, with the settings, seeds, library
versions, the model's revision, any dataset checksums, and the device and batch size of
each session.

```python
import cognarium as cg

subject = cg.load("gpt2", device="auto")
results = cg.run(cg.paradigms.Flanker(), subject, out="runs/flanker")
results.manifest["versions"]
```

## Resuming

A run saves each step as it finishes. Running the same call again continues after the
last saved step, and a finished run makes no more model calls.

```python
again = cg.run(cg.paradigms.Flanker(), subject, out="runs/flanker")
again.to_pandas().equals(results.to_pandas())
```

A folder that holds a different run is refused, with a message naming what changed, and
nothing is deleted.

```python
try:
    cg.run(cg.paradigms.Flanker(seed=1), subject, out="runs/flanker")
except cg.RunConflictError as error:
    print(error)
```

## Seeds

Seeds derive from the paradigm's `seed` per participant, chain, generation and item, so
the trials and sampled responses don't depend on the batch size or the device. Exact
numeric equality of model outputs across devices isn't guaranteed: the same model can
give slightly different log probabilities on CPU, MPS and CUDA.
