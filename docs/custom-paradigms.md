# Custom paradigms

## Your own stimuli

Every built-in paradigm takes `stimuli=` with the columns its page names. A path to a
CSV, JSON, JSON Lines or Parquet file works, as do a pandas table, a `Dataset` and a list
of mappings. A missing column or an empty cell raises an error naming the row.

```python
import cognarium as cg
import pandas as pd

subject = cg.load("gpt2", device="auto")

pairs = pd.DataFrame({
    "good": ["The keys are on the table.", "She has eaten.", "Who did you see?"],
    "bad": ["The keys is on the table.", "She have eaten.", "Who did you see him?"],
    "phenomenon": ["agreement", "agreement", "islands"]
})
pairs.to_csv("pairs.csv", index=False)
results = cg.run(cg.paradigms.MinimalPairs(stimuli="pairs.csv"), subject)
results.to_pandas()[["phenomenon", "answer", "response"]]
```

Results record where the stimuli came from, and say "built-in, not human-normed" when a
paradigm used its own.

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

A numeric column the paradigm writes, such as `n_recalled`, works as a measure too.

Surprisal, entropy and the margin stand in for reaction time, and `score()` says so in
`proxy_for`. When a measure stands in for something else, name it on the effect, as the
built-in `errors` effects do: `cg.Effect("errors", "answer_margin", ..., proxy_for="error rate")`.
