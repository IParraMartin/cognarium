# Quickstart

This page runs the Stroop task on GPT-2, reads the scores, and compares the model's item
means with a table of human data.

## Run a paradigm

```python
import cognarium as cg

subject = cg.load("gpt2", device="auto")
results = cg.run(cg.paradigms.Stroop(), subject)
```

The run writes everything to `runs/stroop/<key>/`, where the key is a digest of the
settings. Running the same call again reads the saved run and makes no model calls.

## Read the scores

```python
results.score()
```

Each row is an effect the paradigm declares. `difference` is `mean_a` minus `mean_b`,
with a bootstrap interval over items, a permutation test's `p_value`, and `detected`
when the interval excludes zero. `measure` names what was compared, and `proxy_for`
names the human behaviour a proxy measure stands in for, such as reaction time for
`answer_surprisal`.

Every row of the experiment is a pandas table:

```python
table = results.to_pandas()
table[["item", "congruency", "answer", "response"]].head()
```

## Ask your own questions

`contrast` compares any measure between two levels of any column, and `metrics`
summarises the run:

```python
results.contrast("answer_probability", by="congruency", levels=("congruent", "incongruent"))
results.metrics("accuracy", "entropy")
```

## Compare with people

`compare_to_human` averages a measure per item and correlates it with human item means.
The values below are invented, only to show the shape of the table, so replace them with
your own data.

```python
import pandas as pd

items = sorted(table["item"].unique())
human = pd.DataFrame({"item": items, "value": [600.0 + 7 * (position % 5) for position in range(len(items))]})
results.compare_to_human(human, "answer_surprisal")
```

The result gives the correlation with an interval over items, the number of shared items,
and, when you pass `ceiling=`, a noise ceiling from
`cg.metrics.split_half_ceiling(ratings)` beside it.

## Next

- Change settings such as `temperature`, `instructions` or `n_participants` on the
  paradigm: see [concepts](concepts.md).
- Swap in your own stimuli with `stimuli=`, or write a paradigm: see
  [custom paradigms](custom-paradigms.md).
- Record hidden states and read the logit lens: see [recording](recording.md).
