# Memory

Paradigms about holding and retrieving items: matching recent items, recalling sequences and lists, recognising probes, and false memories.

Every section ends with a run on a real model, reported as it came out: a paradigm doesn't have to reproduce the human effect, and none was tuned until it did.

```python
import cognarium as cg

subject = cg.load("gpt2", device="auto")
```

## NBack

Say whether each letter matches the one n letters back.

### Maps onto

The n-back task: holding more letters in mind should make matches harder to judge, so performance should fall as n grows.

### What it can't capture

Every earlier letter stays in the context, so the model never has to hold anything, and this measures picking out a position rather than a limited store. Surprisal stands in for response time. Each level of n is its own block in a random order, and the first n letters of a block can't match, so they are left out of the effect. Sequences are generated, not human-normed. d′ comes from `results.metrics("d_prime", positive="yes")`, by load.

### Run it

```python
paradigm = cg.paradigms.NBack()
results = cg.run(paradigm, subject)
results.score()
```

## Span

Repeat a list of digits in order.

### Maps onto

The digit span task: longer lists should be repeated correctly less often.

### What it can't capture

The list stays in the prompt while the model answers, so this measures copying a sequence under interference rather than memory decay. A response counts as correct only with every digit in order. Lists are generated, not human-normed.

### Run it

```python
paradigm = cg.paradigms.Span()
results = cg.run(paradigm, subject)
results.score()
```

## Sternberg

Say whether a probe letter was in a memory set.

### Maps onto

The Sternberg memory scanning task: larger sets should make the judgment harder, which in people shows as slower responses.

### What it can't capture

The set stays in the prompt, and there is no response time, so surprisal stands in for scanning time. Half the probes come from the set. Sets are generated, not human-normed. d′ comes from `results.metrics("d_prime", positive="yes")`, by set size.

### Run it

```python
paradigm = cg.paradigms.Sternberg()
results = cg.run(paradigm, subject)
results.score()
```

## FreeRecall

Study a list of words, then write every word you remember.

### Maps onto

Immediate free recall: words at the end of a list should be recalled more often than those in the middle, which is the recency effect, and words at the start often are too, which is primacy.

### What it can't capture

The list stays in the prompt, so recall here is retrieval from context rather than memory, and there's no delay or distractor task. Each participant's lists share no words. The word pool was written for cognarium and is not human-normed.

### Run it

```python
paradigm = cg.paradigms.FreeRecall()
results = cg.run(paradigm, subject)
results.score()
```

## DRM

Study lists of related words, then judge whether test words were on the list.

### Maps onto

The Deese-Roediger-McDermott false memory task: an unstudied word that every list word relates to, the lure, should be called old more often than an unrelated new word.

### What it can't capture

The list stays in the prompt, so a false alarm reflects association in the text rather than a memory that was never formed. No word lists ship with cognarium, because no open source of them has been verified. Pass your own through `stimuli` with the columns `list`, `word` and `role`, where role is `studied`, `lure` or `unrelated`. cognarium doesn't check them, so the stimuli are not human-normed unless your lists are. Every word is tested once, after its list.

### Run it

No lists ship with cognarium, so two are written here, and the effects need at least two. The tour's run below used four lists written the same way.

```python
written = {
    "cold": (("ice", "snow", "winter", "frost"), "cold", "ladder"),
    "music": (("song", "melody", "piano", "rhythm"), "music", "carpet")
}
lists = []
for name, (studied, lure, unrelated) in written.items():
    lists += [{"list": name, "word": word, "role": "studied"} for word in studied]
    lists += [{"list": name, "word": lure, "role": "lure"}, {"list": name, "word": unrelated, "role": "unrelated"}]
paradigm = cg.paradigms.DRM(stimuli=lists)
results = cg.run(paradigm, subject)
results.score()
```
