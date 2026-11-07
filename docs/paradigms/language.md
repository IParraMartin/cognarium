# Language and semantics

Paradigms about words and sentences: priming, recognising words, parsing, grammaticality and producing words from a cue.

Every section ends with a run on a real model, reported as it came out: a paradigm doesn't have to reproduce the human effect, and none was tuned until it did.

```python
import cognarium as cg

subject = cg.load("gpt2", device="auto")
```

## SemanticPriming

Decide whether a target is a word, after a related or an unrelated prime.

### Maps onto

Semantic priming in lexical decision (Meyer and Schvaneveldt, 1971): a word is recognised faster after a related word, as `nurse` after `doctor`.

### What it can't capture

There is no timing, so the surprisal of the correct answer stands in for decision time and is labelled a proxy, and the prime is read in the same prompt rather than flashed before the target. Each target appears once with a related and once with an unrelated prime, so the comparison is within items, and nonword targets are fillers so that "word" isn't always right. The built-in pairs were written for cognarium and are not human-normed.

### Run it

```python
paradigm = cg.paradigms.SemanticPriming()
results = cg.run(paradigm, subject)
results.score()
```

### A real run

--8<-- "results/semantic_priming.md"

## StructuralPriming

Read a dative sentence, then choose between two forms of another.

### Maps onto

Structural priming (Bock, 1986): after a prepositional-object prime, as in "gave a book to the girl", people more often describe a new event with a prepositional object too, and after a double-object prime, as in "gave the girl a book", with a double object.

### What it can't capture

The subject chooses between two given sentences rather than producing one, and prime and target share their verb, which in people strengthens priming, so the effect isn't purely structural. The prepositional-object form is always the first option. The built-in sentences were written for cognarium and are not human-normed.

### Run it

```python
paradigm = cg.paradigms.StructuralPriming()
results = cg.run(paradigm, subject)
results.score()
```

### A real run

--8<-- "results/structural_priming.md"

## LexicalDecision

Decide whether a string of letters is a word.

### Maps onto

Visual lexical decision: people accept frequent words faster than rare ones, the word frequency effect, and reject pronounceable nonwords.

### What it can't capture

There is no timing, so the surprisal of the correct answer stands in for decision time and is labelled a proxy. Discrimination is available as `results.metrics("d_prime", positive="word")`. The built-in words and nonwords were written for cognarium, and their frequency labels are our own rough estimate rather than corpus counts, so they are not human-normed.

### Run it

```python
paradigm = cg.paradigms.LexicalDecision()
results = cg.run(paradigm, subject)
results.score()
```

### A real run

--8<-- "results/lexical_decision.md"

## GardenPath

Score the word that resolves a temporarily ambiguous sentence.

Each item is a sentence beginning in an ambiguous form, as in "The boat sailed down the river", and an unambiguous one, "The boat that was sailed down the river". The options are the disambiguating word, "sank", and a foil that continues the misleading reading. Both are scored as the next word, straight after the beginning.

### Maps onto

Garden-path effects in reading: the disambiguating word is read slowly after an ambiguous beginning, and less so after an unambiguous one. Its surprisal stands in for the slowdown, as in Futrell et al. (2019).

### What it can't capture

There is no reading time, and surprisal is renormalised over the disambiguator and one foil, so it is relative to that foil. The built-in sentences were written for cognarium and are not human-normed.

### Run it

```python
paradigm = cg.paradigms.GardenPath()
results = cg.run(paradigm, subject)
results.score()
```

### A real run

--8<-- "results/garden_path.md"

## MinimalPairs

Prefer the acceptable sentence of a pair that differs in one place.

### Maps onto

Acceptability judgments of minimal pairs, as used for language models in BLiMP (Warstadt et al., 2020): the acceptable sentence should be the more probable one.

### What it can't capture

This is a preference between two sentences, not a rating of one, and summed log probability favours shorter sentences, so the two sentences of a pair should be the same length. Each pair is shown in its own random order, and results are reported by phenomenon. The built-in pairs were written for cognarium, are the same length in words, and are not human-normed.

### Run it

```python
paradigm = cg.paradigms.MinimalPairs()
results = cg.run(paradigm, subject)
results.score()
```

### A real run

--8<-- "results/minimal_pairs.md"

## Fluency

Name as many words as you can from a category, or starting with a letter.

### Maps onto

Verbal fluency, where people usually produce more words for a semantic category than for a letter in the same time.

### What it can't capture

There is no time limit, so the token budget stands in for one, and words given for a category aren't checked for membership, since cognarium holds no category norms. A response is read as a list separated by commas, semicolons or new lines, or by spaces when it has none of those. The categories and letters are common choices, not human-normed.

### Run it

```python
paradigm = cg.paradigms.Fluency()
results = cg.run(paradigm, subject)
results.score()
```

### A real run

--8<-- "results/fluency.md"
