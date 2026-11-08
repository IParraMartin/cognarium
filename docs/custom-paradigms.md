# Custom paradigms

## Your own items

The learning paradigms take their items as a task. `cg.tasks.Mapping` holds any items,
described by numbers or by words, with a fixed set of labels, so iterated learning runs
on your own category system or function without a new class, as the
[learning paradigms](paradigms/learning.md) page shows.

A paradigm you write can read rows from a file instead. Set `item_columns` to the columns
it needs, and `stimuli=` then takes a path to a CSV, JSON, JSON Lines or Parquet file, a
pandas table, a `Dataset` or a list of mappings. `self.item_rows(rng)` gives the rows. A
missing column or an empty cell raises an error naming the row.

```python
import cognarium as cg
import pandas as pd

subject = cg.load("gpt2", device="auto")


class Acceptability(cg.Paradigm):
    """Choose the acceptable sentence of a pair that differs in one place."""

    item_columns = ("good", "bad")
    effects = (cg.Effect("preference", "answer_probability", against=0.5),)

    def trials(self, rng):
        for index, row in enumerate(self.item_rows(rng)):
            first, second = (row["good"], row["bad"]) if index % 2 == 0 else (row["bad"], row["good"])
            yield cg.Trial(
                cg.Stimulus(text=f"A: {first}\nB: {second}\nWhich sentence is acceptable?"),
                options=["A", "B"],
                answer="A" if first == row["good"] else "B",
                item=index
            )


pairs = pd.DataFrame({
    "good": ["The keys are on the table.", "She has eaten.", "Who did you see?"],
    "bad": ["The keys is on the table.", "She have eaten.", "Who did you see him?"]
})
pairs.to_csv("pairs.csv", index=False)
results = cg.run(Acceptability(stimuli="pairs.csv"), subject)
results.to_pandas()[["item", "answer", "response"]]
```

Results record where the stimuli came from.

## A new paradigm

A paradigm is a subclass of `cg.Paradigm` with a `trials` method. Keyword arguments
become attributes, so the paradigm below reads `self.pairs`. Effects are declared as
data: each names a measure, the column whose levels it compares, and the two levels.

```python
class Priming(cg.Paradigm):
    """Decide whether a target is a word, after a related or an unrelated prime."""

    context = cg.context.isolated
    effects = (
        cg.Effect("priming", "answer_surprisal", by="relatedness", levels=("unrelated", "related")),
    )

    def trials(self, rng):
        for prime, target, relatedness in self.pairs:
            yield cg.Trial(
                cg.Stimulus(text=f"Prime: {prime}. Target: {target}. Is the target a word?"),
                options=["word", "nonword"],
                answer="word",
                condition={"relatedness": relatedness},
                item=target
            )


pairs = [
    ("doctor", "nurse", "related"),
    ("table", "nurse", "unrelated"),
    ("bread", "butter", "related"),
    ("cloud", "butter", "unrelated"),
    ("cat", "dog", "related"),
    ("lamp", "dog", "unrelated")
]
results = cg.run(Priming(pairs=pairs), subject)
results.score()
```

`rng` is a NumPy generator seeded for each participant, for anything random in the trials.
For more control, set `template` to a `cg.PromptTemplate` to change how trials are
written, `group_by` to the columns metrics are reported by, or override
`score` to compute effects some other way.

## Measures for effects

| Measure | One value per row |
|---|---|
| `answer_probability` | The probability of the correct answer |
| `answer_surprisal` | The surprisal of the correct answer, in bits, a proxy for reaction time |
| `answer_margin` | How far, in nats, the correct answer's log probability is above the best other option's |
| `option_probability` | The probability of one option, named with `options={"option": ...}` |
| `expected_option` | The probability-weighted position among ordered options |
| `entropy` | The entropy of the response distribution |
| `accuracy`, `surprisal` | Whether the response was correct, and the surprisal of the response, in bits |

A numeric column the paradigm writes, such as a count it parses, works as a measure too.

Surprisal, entropy and the margin stand in for reaction time, and `score()` says so in
`proxy_for`. When a measure stands in for something else, name it on the effect, as in
`cg.Effect("errors", "answer_margin", ..., proxy_for="error rate")`.
