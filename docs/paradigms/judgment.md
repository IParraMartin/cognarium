# Judgment and reasoning

Paradigms about estimates, risky choice, probability and other minds.

Every section ends with a run on a real model, reported as it came out: a paradigm doesn't have to reproduce the human effect, and none was tuned until it did.

```python
import cognarium as cg

subject = cg.load("gpt2", device="auto")
```

## Anchoring

Compare a quantity with a number, then estimate it.

Each question is asked once after a low anchor and once after a high one, as in "Is the number of keys on a standard piano more or less than 220?", and the estimate is a choice among the same five ordered ranges either way.

### Maps onto

The anchoring effect (Tversky and Kahneman, 1974): estimates move toward a number that was considered first, even an arbitrary one. The measure is the expected position among the ranges, which should be higher after the high anchor.

### What it can't capture

Ranges replace free estimates, so a shift within a range is lost, and the anchor is in the same prompt as the estimate rather than a separate question answered first. The built-in questions were written for cognarium and are not human-normed.

### Run it

```python
paradigm = cg.paradigms.Anchoring()
results = cg.run(paradigm, subject)
results.score()
```

### A real run

--8<-- "results/anchoring.md"

## Framing

Choose between a sure outcome and a gamble, described as gains or as losses.

Each scenario is told twice with the same outcomes: once as what will be saved and once as what will be lost. In both, the sure option equals the gamble's expected outcome, and the text says to answer `sure` or `gamble`.

### Maps onto

Risky-choice framing (Tversky and Kahneman, 1981): people prefer the sure option when outcomes are described as gains and the gamble when they are described as losses.

### What it can't capture

Nothing is at stake, and both frames of a scenario are judged independently rather than by different people. The built-in scenarios were written for cognarium and are not human-normed.

### Run it

```python
paradigm = cg.paradigms.Framing()
results = cg.run(paradigm, subject)
results.score()
```

### A real run

--8<-- "results/framing.md"

## ConjunctionFallacy

Judge whether a single property or that property joined with another is likelier.

Each vignette is told with a description that fits the added property and with a neutral one. The two statements are numbered 1 and 2, the single property takes each position equally often, and the answer is always the single property, since a conjunction can't be more probable than one of its parts.

### Maps onto

The conjunction fallacy (Tversky and Kahneman, 1983): a description that fits one property makes people rate that property joined with another as more likely than the single property. The primary effect is the drop in choosing the single property after a fitting description, and `error_rate` compares that choice with chance.

### What it can't capture

A choice between two statements replaces ranking several, and a subject may read "more likely" as "more fitting". The built-in vignettes were written for cognarium and are not human-normed.

### Run it

```python
paradigm = cg.paradigms.ConjunctionFallacy()
results = cg.run(paradigm, subject)
results.score()
```

### A real run

--8<-- "results/conjunction_fallacy.md"

## FalseBelief

Say where someone will look for an object that was moved, seen or unseen.

Each story is told twice from one template: once where the person is away when the object is moved, so they believe it is still in its first place, and once where they watch it move. The options are the two places, and each place holds the answer equally often.

### Maps onto

The change-of-location false-belief task (Wimmer and Perner, 1983): someone who didn't see an object moved should look where they left it. The true-belief control shows a subject that always names the object's real place, or always its first place, for what it is.

### What it can't capture

Passing doesn't show theory of mind, because text statistics can produce the same answers, and the story is read rather than watched. The built-in stories were written for cognarium and are not human-normed.

### Run it

```python
paradigm = cg.paradigms.FalseBelief()
results = cg.run(paradigm, subject)
results.score()
```

### A real run

--8<-- "results/false_belief.md"
