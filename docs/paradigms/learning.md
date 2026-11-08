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

Every chain starts from the task's starting labels. Each generation sees `n_prev_examples` of the previous generation's labelled items as examples, labels every item, and passes those labels on. The task is either the Shepard structures or a `cg.tasks.Mapping`, which holds any category system or function on a fixed set of labels.

### Maps onto

Iterated learning experiments, where each learner is trained on what the previous learner produced and chains drift toward the structures learners find easiest (Kirby, Cornish and Smith, 2008; Griffiths, Christian and Kalish, 2008). For a Bayesian learner that samples from its posterior, the chains converge to its prior (Griffiths and Kalish, 2007), which the test suite checks exactly.

### What it can't capture

A language model reads the transmitted labels in its prompt rather than learning them, and every generation is the same model with a fresh context, so the chains show the model's in-context biases rather than a population of different people. A language model also reads features as written numbers or words, where people see colours, bars or animals. The paradigm declares no single effect: its measures are per generation, such as `accuracy` against the starting labels, `fidelity` to the previous generation, `chain_entropy` across chains and `kl` from a reference distribution. The eight Shepard objects are abstract, not human-normed.

### Run it

```python
paradigm = cg.paradigms.IteratedLearning(task=cg.tasks.ShepardCategories(type="VI"))
results = cg.run(paradigm, subject)
results.metrics("accuracy", "fidelity", "chain_entropy", "kl")
```

### Any category system

A `cg.tasks.Mapping` takes one feature vector per item and the fixed set of labels a subject picks from. Each item and its label are written `Features: [60, 40, 0] -> Label: wug`, and `instructions` come before the examples. With no `start`, each chain starts from its own random labelling. These are 12 hues around a circle, in CIELAB coordinates:

```python
import math

chips = [
    [60, round(40 * math.cos(math.radians(hue))), round(40 * math.sin(math.radians(hue)))]
    for hue in range(0, 360, 30)
]
colours = cg.tasks.Mapping(chips, ("wug", "dax", "fep", "kiki"))
paradigm = cg.paradigms.IteratedLearning(
    colours,
    generations=5,
    chains=4,
    n_prev_examples=6,
    instructions="Name each colour with one of the words wug, dax, fep or kiki."
)
results = cg.run(paradigm, subject)
```

Animals, kinship terms or spatial relations work the same way, with their features as numbers or as words. Items described by words can't use the bottleneck measures below, which need distances between items.

### Measuring the naming systems

A chain's naming system in a generation is its answer for every item. As in Imel and Zaslavsky (2026), `ib_complexity` and `ib_accuracy` place each system on the information bottleneck trade-off of Zaslavsky et al. (2018), `efficiency_loss` is how far it falls from the optimal trade-off, and `ib_alignment` is 1 minus the gNID between the system and the nearest optimal one. Each item's meaning is a Gaussian over the items around it, with a width set by `sigma` on the `Mapping`. The values are averages over chains, and `by` keeps one row per chain:

```python
results.metrics("ib_complexity", "ib_accuracy", "efficiency_loss", "ib_alignment")
```

To compare with a known system, pass it as `reference`, either one label per item or an items × labels table of probabilities. `alignment` is 1 minus the gNID to it, and `co_naming_mse` compares how likely each pair of items is to share a label, so the label names don't have to match:

```python
reference = ["red"] * 3 + ["yellow"] * 3 + ["green"] * 3 + ["blue"] * 3
results.metrics("alignment", "co_naming_mse", reference=reference, by=["generation", "chain"])
```

### Remembering its own answers

With `memory`, each generation names its items one at a time in a random order and also sees its own last `memory` answers, as in the iterated in-context learning of Imel and Zaslavsky (2026). Each answer is then a separate step:

```python
paradigm = cg.paradigms.IteratedLearning(colours, generations=3, chains=2, n_prev_examples=6, memory=4)
results = cg.run(paradigm, subject)
```

For an instruction-tuned model, `cg.load(name, chat=True)` sends each prompt as one user message through the model's chat template, and scores the options as the start of its reply.

### Function learning

`Mapping.from_function` builds the function-learning task of Kalish, Griffiths and Lewandowsky (2007): each x is an item, the points of a scale are the labels, and the first generation sees the scale point nearest to the function. In the paper, chains that started from a negative slope drifted to y = x, and they reported the median correlation with y = x across chains. A smaller version:

```python
negative = cg.tasks.Mapping.from_function(range(1, 21), lambda x: 21 - x, scale=range(1, 21))
paradigm = cg.paradigms.IteratedLearning(negative, generations=3, chains=2, n_prev_examples=10)
results = cg.run(paradigm, subject)
per_chain = results.metrics("correlation", reference=lambda x: x, by=["generation", "chain"])
per_chain.groupby("generation")["value"].median()
```

`mse` gives the mean squared distance from the reference instead. `examples/function_learning.py` runs the paper's design: four starting functions, eight chains, nine generations and 50 examples.

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
