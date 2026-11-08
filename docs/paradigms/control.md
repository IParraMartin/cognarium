# Control

Paradigms about cognitive control: conflict between what a stimulus says and what the task asks, withholding a response, and switching between rules.

Every section ends with a run on a real model, reported as it came out: a paradigm doesn't have to reproduce the human effect, and none was tuned until it did.

```python
import cognarium as cg

subject = cg.load("gpt2", device="auto")
```

## Stroop

Name the ink colour of a colour word.

### Maps onto

The Stroop colour-naming task: the word and its ink either match or conflict, and conflict should make the ink harder to name.

### What it can't capture

The model reads a description of the ink rather than seeing a colour, so this measures conflict between two described attributes, and surprisal stands in for naming time. The built-in items are every pairing of the colours, not human-normed.

### Run it

```python
paradigm = cg.paradigms.Stroop()
results = cg.run(paradigm, subject)
results.score()
```

## Flanker

Name the middle letter of a row of letters.

### Maps onto

The Eriksen flanker task: the letters around the target either match it or call for another response.

### What it can't capture

The letters are tokens in a row rather than positions in space, so there is no spatial attention, and surprisal stands in for response time. The built-in items are every target and flanker pairing, not human-normed.

### Run it

```python
paradigm = cg.paradigms.Flanker()
results = cg.run(paradigm, subject)
results.score()
```

## Simon

Press the key a colour is mapped to, wherever the shape appears.

### Maps onto

The Simon task: the side a stimulus appears on is irrelevant, but responses should be harder when it differs from the side of the correct key.

### What it can't capture

The model reads the word "left" or "right" rather than seeing a location, so the conflict is between words, and surprisal stands in for response time. Each item states its own colour-to-key mapping, and half the items use each mapping. The built-in items are generated, not human-normed.

### Run it

```python
paradigm = cg.paradigms.Simon()
results = cg.run(paradigm, subject)
results.score()
```

## GoNoGo

Press for every letter except one.

### Maps onto

The go/no-go task: frequent go trials build a habit of pressing, which the rare no-go letter must override.

### What it can't capture

The habit can only come from the subject's own recent responses in its context, and there's no time pressure, so surprisal stands in for the difficulty of withholding. Sequences are generated, not human-normed. d′ comes from `results.metrics("d_prime", positive="press")`.

### Run it

```python
paradigm = cg.paradigms.GoNoGo()
results = cg.run(paradigm, subject)
results.score()
```

## StopSignal

Press for each letter unless a stop cue follows it.

### Maps onto

The stop-signal task: a cue that arrives later should be harder to obey, so the chance of pressing despite it should rise with its delay.

### What it can't capture

The delay is a count of tokens before the cue, with no time in it, and the model reads the whole trial at once, so no stop-signal reaction time is reported, only how pressing changes with delay. Go trials show the same filler with no cue. Sequences are generated, not human-normed.

### Run it

```python
paradigm = cg.paradigms.StopSignal()
results = cg.run(paradigm, subject)
results.score()
```

## TaskSwitching

Judge either the letter or the number of a pair, as each trial's cue says.

### Maps onto

Task switching with explicit cues: answering should be harder right after the task changes than when it repeats.

### What it can't capture

There's no time to prepare between cue and stimulus, and the previous tasks are only in context as text, so surprisal stands in for the switch cost. The first trial has no previous task and is labelled `start`, which leaves it out of the effect. Sequences are generated, not human-normed.

### Run it

```python
paradigm = cg.paradigms.TaskSwitching()
results = cg.run(paradigm, subject)
results.score()
```
