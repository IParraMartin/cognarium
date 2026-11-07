# Learning and transmission

Paradigms where the subject learns during the experiment. These run on neural models in context and, with the same trials and scoring, on exemplar and Bayesian learners.

Every section ends with a run on a real model, reported as it came out: a paradigm doesn't have to reproduce the human effect, and none was tuned until it did.

```python
import cognarium as cg

subject = cg.load("gpt2", device="auto")
```

The same paradigms run on the non-neural learners, which learn from the same trials:

```python
task = cg.tasks.ShepardCategories(type="VI")
for learner in (cg.subjects.ExemplarLearner(), cg.subjects.BayesianLearner()):
    print(cg.run(cg.paradigms.CategoryLearning(task=task), learner).score())
```

## IteratedLearning

Each generation learns from the last generation's responses.

Every chain starts from the task's seed structure. Each generation sees a random subset of the previous generation's labels, labels every item, and passes those labels on.

### Maps onto

Iterated learning experiments, where each learner is trained on what the previous learner produced and chains drift toward the structures learners find easiest (Kirby, Cornish and Smith, 2008; Griffiths, Christian and Kalish, 2008). For a Bayesian learner that samples from its posterior, the chains converge to its prior (Griffiths and Kalish, 2007), which the test suite checks exactly.

### What it can't capture

A language model reads the transmitted labels in its prompt rather than learning them, and every generation is the same model with a fresh context, so the chains show the model's in-context biases rather than a population of different people. The paradigm declares no single effect: its measures are per generation, such as `accuracy` against the seed structure, `fidelity` to the previous generation, `chain_entropy` across chains and `kl` from a reference distribution. The eight objects are abstract, not human-normed.

### Run it

```python
paradigm = cg.paradigms.IteratedLearning(task=cg.tasks.ShepardCategories(type="VI"))
results = cg.run(paradigm, subject)
results.metrics("accuracy", "fidelity", "chain_entropy", "kl")
```

### A real run

--8<-- "results/iterated_learning.md"

## CategoryLearning

Learn a category structure from corrective feedback, block by block.

Every block shows each object once, in a new order, and asks for its category. Before a block, the subject sees every earlier block's objects with their correct labels. Each participant is a separate learner with their own orders.

### Maps onto

Supervised category learning with the Shepard, Hovland and Jenkins (1961) structures, where people learn type I fastest and type VI slowest.

### What it can't capture

A person gets feedback after each response and forgets, while the model reads its whole history at once, so there is no forgetting and no effect of its own errors. The objects are the eight abstract ones of the task, not human-normed.

### Run it

```python
paradigm = cg.paradigms.CategoryLearning(task=cg.tasks.ShepardCategories(type="I"))
results = cg.run(paradigm, subject)
results.score()
```

### A real run

--8<-- "results/category_learning.md"

## ArtificialGrammar

Study strings from a finite-state grammar, then judge new ones.

The grammar is Reber's (1967), over the letters T, P, S, X and V, with strings of 3 to 8 letters. Each participant studies their own strings, each labelled grammatical, then judges new grammatical strings and as many ungrammatical ones, each made by changing one letter of a tested grammatical string so that it leaves the grammar. Letters are written apart, as in `Letters: T X S`, so a subject that reads words sees letters. A string's features count its letter pairs, including its first and last letters, and strings that would share features are never used.

### Maps onto

Artificial grammar learning, where people trained on grammatical strings later judge new ones above chance without being able to say the rules.

### What it can't capture

People aren't told the strings follow rules, while here every study string carries the same label, and the measure is a judgment probability rather than the endorsement rate. The strings come from the grammar, not human-normed.

### Run it

```python
paradigm = cg.paradigms.ArtificialGrammar()
results = cg.run(paradigm, subject)
results.score()
```

### A real run

--8<-- "results/artificial_grammar.md"

## StatisticalLearning

Find the words in a stream of syllables, then judge words and part-words.

The language has nonsense words of three consonant-vowel syllables, each syllable used in one word only, drawn from the seed. Each participant hears their own stream, in which every word occurs equally often and never twice in a row. The stream is shown as every run of three syllables in it, each labelled familiar. Tests are the words and as many part-words, which join the end of one word to the start of the word that followed it in the stream, so they occurred too, but less often. A learner who found the words calls words familiar and part-words not.

### Maps onto

Statistical word segmentation (Saffran, Aslin and Newport, 1996), where syllables inside a word predict each other better than syllables across a word boundary.

### What it can't capture

The stream is shown as overlapping windows of text rather than continuous speech, because every subject type must read the same trials, and each window is judged rather than two items compared. The syllables are generated, not human-normed.

### Run it

```python
paradigm = cg.paradigms.StatisticalLearning()
results = cg.run(paradigm, subject)
results.score()
```

### A real run

--8<-- "results/statistical_learning.md"
