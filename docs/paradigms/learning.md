# Learning and transmission

These are the paradigms whose task survives being written as text. The learner's
evidence is the labelled examples in its context, and the measure is the probability
it gives each answer, so nothing rests on reaction times or on a capacity limit a
model doesn't have. Each one runs on a language model in context and, with the same
trials and scoring, on two learners that serve as baselines: an exemplar learner,
which labels by similarity to the examples it has stored (the GCM of Nosofsky, 1986),
and a Bayesian learner, which infers a labelling from a space of hypotheses. A model's
results can be read against what learning by similarity and learning by rule each
predict.

| Paradigm | The learner | What people show | Built on |
|---|---|---|---|
| [IteratedLearning](#iteratedlearning) | Learns from the previous generation's answers and passes its own on | Chains drift toward what learners find easiest, which reveals their inductive biases | Kirby, Cornish and Smith (2008); Kalish, Griffiths and Lewandowsky (2007); Griffiths and Kalish (2007) |
| [CategoryLearning](#categorylearning) | Labels each item and then sees its correct label | Shepard type I is learned fastest, type II next, types III to V more slowly and type VI slowest | Shepard, Hovland and Jenkins (1961); Nosofsky et al. (1994) |
| [ArtificialGrammar](#artificialgrammar) | Studies strings from a grammar, then judges new ones | New strings are judged above chance, and familiar chunks are endorsed whatever their grammaticality | Reber (1967); Knowlton and Squire (1996) |

Full citations are under [References](#references).

```python
import cognarium as cg

subject = cg.load("gpt2", device="auto")
baselines = (cg.subjects.ExemplarLearner(), cg.subjects.BayesianLearner(decision="average"))
```

`paradigm.preview(query)` writes out any query as the model reads it, ending with the
options it scores.

## IteratedLearning

Each generation learns from the last generation's responses.

Every chain starts from the task's starting labels. Each generation sees `n_prev_examples` of the previous generation's labelled items as examples, labels every item, and passes those labels on. The task is either the Shepard structures or a `cg.tasks.Mapping`, which holds any category system or function on a fixed set of labels.

### How it runs on a language model

Every generation is the same model with a fresh prompt. The prompt holds
`n_prev_examples` of the previous generation's answers, one item and its label per
line, and then asks for one item. These are the first query of a chain in the
function-learning task below, where y starts at 101 - x:

<!-- preview: cg.paradigms.IteratedLearning(cg.tasks.Mapping.from_function(range(1, 101), lambda x: 101 - x, scale=range(1, 101)), n_prev_examples=4, instructions="Learn the rule from the examples and give the label of the last feature.").preview(0).split("\n[options")[0] -->
```text
Learn the rule from the examples and give the label of the last feature.
Features: 27 -> Label: 74
Features: 51 -> Label: 50
Features: 83 -> Label: 18
Features: 63 -> Label: 38
Features: 1 -> Label:
```

The options are every label, here the numbers 1 to 100. Each is scored by its log
probability as the prompt's next tokens, summed over its tokens, and the scores are
normalised over the labels. When one label's tokens start another's, as 1 starts 10,
every label is scored up to the line break that ends an answer, so a short label
doesn't win only by being short. The generation's answer is sampled from that
distribution, after `temperature`, `top_p` and `top_k`, and passed on. Every item is
asked with the same examples unless `memory` is set.

### Maps onto

Iterated learning experiments, where each learner is trained on what the previous learner produced. Chains of people drift toward what they find easiest to learn: languages become structured and easier to learn (Kirby, Cornish and Smith, 2008), functions become linear with a positive slope whatever the starting function (Kalish, Griffiths and Lewandowsky, 2007), category structures become simpler (Griffiths, Christian and Kalish, 2008), and colour terms come to resemble those of the world's languages (Xu, Dowman and Griffiths, 2013). For a Bayesian learner that samples from its posterior, the chains converge to its prior (Griffiths and Kalish, 2007), which the test suite checks exactly.

With language models, iterated in-context learning has been used to elicit a model's priors (Zhu and Griffiths, 2024) and to follow colour naming systems as they become efficient (Imel and Zaslavsky, 2026). Ren et al. (2024) apply the same framework to models that learn from other models' outputs.

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

Animals, kinship terms or spatial relations work the same way, with their features as numbers or as words. Items described by words can use the bottleneck measures below only when `points` gives each item coordinates, since the measures need distances between items.

### Measuring the naming systems

A chain's naming system in a generation is its answer for every item. As in Imel and Zaslavsky (2026), `ib_complexity` and `ib_accuracy` place each system on the information bottleneck trade-off of Zaslavsky et al. (2018), `efficiency_loss` is how far it falls from the optimal trade-off, and `ib_alignment` is 1 minus the gNID between the system and the nearest optimal one. Each item's meaning is a Gaussian over the items around it, in the coordinates the `Mapping` gets as `points`, which default to its features, with a width set by `sigma`. The values are averages over chains, and `by` keeps one row per chain:

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

### The World Color Survey

`examples/color_naming.py` runs the design of Imel and Zaslavsky (2026) on the 330 chips of the World Color Survey (Kay et al., 2009): vocabularies of 2 to 6 made-up words, a random first naming, 6 examples per word, `memory=10`, 13 generations, temperature 0.6 and top-p 0.9. The prompt shows each chip's sRGB values from 0 to 1, which the paper found worked better for text-only models than CIELAB. The measures use the chip's CIELAB coordinates instead, passed as `points`, with `sigma=8` as in Zaslavsky et al. (2018). The chips are in the registry as `wcs_chips`, which states no licence, so loading it needs `accept_terms=True`. With `lab_to_srgb` from the example:

<!-- no-run -->
```python
chips = cg.data.load("wcs_chips", accept_terms=True).to_pandas()
lab = chips[["L", "a", "b"]].to_numpy()
colours = cg.tasks.Mapping(lab_to_srgb(lab).round(3).tolist(), ("Narp", "Tovo", "Feglu"), points=lab.tolist(), sigma=8.0)
```

On Qwen3.5-4B-Base, in bfloat16 with four chains per vocabulary size, the naming systems grow more efficient at every size. Each cell is a mean over the four chains:

| Words | Efficiency loss, random start | Efficiency loss, generation 13 | Alignment, generation 13 | Words in use, generation 13 |
|---|---|---|---|---|
| 2 | 0.86 | 0.11 | 0.07 | 1.5 |
| 3 | 1.30 | 0.17 | 0.15 | 2.0 |
| 4 | 1.57 | 0.23 | 0.14 | 2.5 |
| 5 | 1.79 | 0.25 | 0.28 | 3.0 |
| 6 | 1.91 | 0.39 | 0.45 | 4.5 |

Part of the fall comes from simplification: words drop out of use, and a system with fewer words lies nearer the frontier's simple end. The gain isn't only that, though. The 6-word systems lose some complexity, from 2.57 to 2.02 bits, while their accuracy triples, from 0.51 to 1.54 bits, and their alignment with the nearest optimal system rises from 0.04 to 0.45. The categories that form are contiguous regions of the chip grid. This is one run, reported as it came out.

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

Learn a category structure, or a function, from corrective feedback.

Every block shows each item `repeats` times in a new order and asks for its label.
With `feedback="trial"`, the default, each trial's context holds every earlier trial
with its correct label, so the subject learns as it goes. With `feedback="block"`,
every trial of a block shares one context holding the earlier blocks only, which is
faster for learners that train their weights, but leaves the first block at chance.
Each participant is a separate learner with their own orders.

The task says what is learned: one of the six Shepard structures, or a
`cg.tasks.Mapping`, which holds any items and labels, with its `start` as the labelling
taught. With no `start`, the run's seed draws a random labelling.

### How it runs on a language model

The prompt holds every earlier trial with its correct label, then the current item.
This is the fourth trial of type II:

<!-- preview: cg.paradigms.CategoryLearning(cg.tasks.ShepardCategories(type="II")).preview(3) -->
```text
Object: large red circle
Category: B
Object: large red square
Category: B
Object: small blue circle
Category: B
Object: small red circle
Category:
[options: A | B]
```

The measure is the probability of the correct label, normalised over the labels,
before the model sees it. One minus its mean per block is the learning curve, the
probability of an error per block that Nosofsky et al. (1994) report for people.
`score()` compares the last block with the first.

### Maps onto

Supervised category learning. With the Shepard, Hovland and Jenkins (1961)
structures, people learn type I fastest, type II next, types III to V more slowly
and type VI slowest (Nosofsky et al., 1994). The order has been explained by
selective attention to the relevant features (Kruschke, 1992) and by the Boolean
complexity of each structure's simplest rule (Feldman, 2000), and it is a standard
test for models of categorization. With a function read on a scale, it is function
learning from feedback (DeLosh, Busemeyer and McDaniel, 1997).

### What it can't capture

A person's context holds their own responses, and they forget. The model reads every
earlier item with its correct label and nothing else, so there is no forgetting and no
effect of its own errors, much as in observational training, where people see each
item with its label. The eight Shepard objects are abstract, not human-normed.

### Run it

```python
paradigm = cg.paradigms.CategoryLearning(cg.tasks.ShepardCategories(type="II"), blocks=4)
results = cg.run(paradigm, subject)
results.metrics("answer_probability", "accuracy")
```

### Any category system or function

A `Mapping` with items described by words works the same way:

```python
animals = cg.tasks.Mapping(
    [["fur", "four legs"], ["feathers", "two legs"], ["fur", "two legs"], ["scales", "no legs"]],
    ("dax", "wug"),
    start=["dax", "wug", "dax", "wug"]
)
results = cg.run(cg.paradigms.CategoryLearning(animals, blocks=3), subject)
```

With `Mapping.from_function`, the subject learns a function from feedback, and `mse`
and `correlation` score each participant's answers in a block against it:

```python
line = cg.tasks.Mapping.from_function(range(1, 11), lambda x: x, scale=range(1, 11))
results = cg.run(cg.paradigms.CategoryLearning(line, blocks=3), subject)
results.metrics("mse", "correlation", reference=lambda x: x)
```

### The six Shepard types

`examples/shepard_types.py` runs every type and prints the probability of an error by
block and type, each type's mean, and the rank correlation between the means and the
human order, with types III to V tied. These are the mean probabilities of an error
over six blocks, with ten learners per type:

| Learner | I | II | III | IV | V | VI | Rank correlation |
|---|---|---|---|---|---|---|---|
| Exemplar (GCM) | 0.381 | 0.492 | 0.438 | 0.415 | 0.474 | 0.529 | 0.58 |
| Bayesian, simplicity prior | 0.033 | 0.055 | 0.067 | 0.064 | 0.064 | 0.083 | 0.94 |
| GPT-2 | 0.396 | 0.449 | 0.376 | 0.358 | 0.427 | 0.488 | 0.21 |
| Qwen3.5-4B-Base | 0.063 | 0.105 | 0.102 | 0.102 | 0.113 | 0.138 | 0.70 |

The exemplar learner weighs every feature equally, so it finds type II harder than
types III to V, the order an exemplar model gives without attention learning
(Kruschke, 1992).
The Bayesian learner considers the 70 labellings with four objects per category, with
a prior that halves for each unit of Boolean complexity (Feldman, 2000), so it
prefers simple rules. Every type is learned within a block, so only the first block
tells the types apart.

GPT-2 learns every type, its probability of an error falling from about 0.5 in the
first block to between 0.25 and 0.44 in the sixth, but its order matches people's
only weakly: type VI is its hardest, as for people, while types II and V are harder
for it than types III and IV, and type I isn't its easiest. This is one run, reported
as it came out.

Qwen3.5-4B-Base, in bfloat16, learns every type within about three blocks, and by the
sixth its probability of an error is at most 0.005. Its order is close to people's:
type I is its easiest and type VI its hardest, the gap clearest in the second block
(0.03 against 0.31), while type II falls among types III to V rather than ahead of
them. This is also one run, reported as it came out.

## ArtificialGrammar

Study strings from a finite-state grammar, then judge new ones.

The grammar is Reber's (1967), over the letters T, P, S, X and V, with strings of 3
to 8 letters. Each participant studies their own strings, then is tested on new
grammatical strings and as many ungrammatical ones. Each ungrammatical string changes
one letter of a tested grammatical string so that it leaves the grammar, and of the
changes possible, the one closest to the original in chunk strength is used. A
string's chunk strength is how often, on average, its letter pairs and triples
occurred in the participant's study strings, so grammaticality and familiarity come
apart as in Knowlton and Squire (1996). Letters are written apart, so a model reads
letters rather than made-up words.

### How it runs on a language model

There are two tests. With `test="judgment"`, the default, each study string is
labelled grammatical, and each test string is judged:

<!-- preview: cg.paradigms.ArtificialGrammar(n_train=3, n_test=2).preview(0) -->
```text
Letters: P T T V P S
Judgment: grammatical
Letters: P T V P S
Judgment: grammatical
Letters: T S S X X T V V
Judgment: grammatical
Letters: S V P X T T V V
Judgment:
[options: grammatical | ungrammatical]
```

The measure is the probability of answering grammatical. `score()` gives two effects:
`grammaticality`, between grammatical and ungrammatical strings, and `chunk_strength`,
between the strings in the upper and lower half of chunk strength within each status.
Since every study string carries the same label, a model may answer grammatical to
everything, and both effects are differences, so that bias cancels.

With `test="probability"`, the study strings are listed bare, and each trial asks for
nothing. Its options are a grammatical string and its ungrammatical change, each
scored as a whole string, up to the line break that ends it, so a string that stops
too early scores low:

<!-- preview: cg.paradigms.ArtificialGrammar(n_train=3, n_test=2, test="probability").preview(0) -->
```text
Letters: P T T V P S
Letters: P T V P S
Letters: T S S X X T V V
Letters:
[options: S V P X T T V V | P V P X T T V V]
```

The measure is the probability of the grammatical string over the pair, and `score()`
compares it with chance. This is the model's own familiarity with each string, which
a judgment can misreport (Hu and Levy, 2023). Only a model that reads text gives it,
so the exemplar and Bayesian learners take the judgment test only. Results hold each
test string's `chunk_strength`, and under the probability test the
`chunk_difference` between the two strings of a pair.

### Maps onto

Artificial grammar learning, where people who studied grammatical strings judge new
ones above chance without being able to say the rules (Reber, 1967), and also endorse
strings made of familiar chunks, whatever their grammaticality (Knowlton and Squire,
1996). Knowlton and Squire crossed grammaticality with chunk strength to separate
learning of the grammar from familiarity with its fragments, and the ungrammatical
strings here are matched on chunk strength for the same reason. For language models,
Hu and Levy (2023) found that a model's answer to a judgment prompt can misreport
what its string probabilities know, which is why there is a probability test.

### What it can't capture

People aren't told the strings follow rules, while under the judgment test every study
string carries the same label. The measures are probabilities, not endorsement rates.
The strings come from the grammar, not human-normed.

### Run it

```python
paradigm = cg.paradigms.ArtificialGrammar()
results = cg.run(paradigm, subject)
results.score()
results.metrics("option_probability", option="grammatical")
```

```python
pairs = cg.run(cg.paradigms.ArtificialGrammar(test="probability"), subject)
pairs.score()
```

### Rules or chunks

`examples/grammar_learning.py` runs both tests. With 20 learners, each studying 16
strings and tested on 8 and their changes, the two baselines come apart, and the
language models barely show either effect. Each cell is the difference in the probability of answering
grammatical, with its 95% interval over strings:

| Learner | Grammaticality | Chunk strength |
|---|---|---|
| Exemplar (GCM) | 0.003 [0.001, 0.004] | 0.003 [0.002, 0.005] |
| Bayesian | 1.000 [1.000, 1.000] | 0.003 [-0.132, 0.133] |
| GPT-2 | 4e-6 [1.5e-6, 5e-6] | 2e-6 [-2e-7, 3e-6] |
| Qwen3.5-4B-Base | 0.003 [0.002, 0.005] | 0.001 [0.000, 0.003] |

The Bayesian learner infers the grammar and ignores chunk strength. The exemplar
learner, which compares letter pairs, follows both a little, since an illegal change
usually makes a letter pair it never saw. Matching each change on chunk strength keeps
the two effects apart: the grammatical and ungrammatical strings' mean chunk strengths
differ by 0.015 (4.70 against 4.68), against 1.7 between the upper and lower halves.

GPT-2 answers grammatical to every string, with a probability above 0.99999, copying
the label of the study strings, so its judgments carry almost nothing and its d′ is 0.
The probability test finds what the judgments hide: the grammatical string of a pair
has a probability of 0.818 averaged over strings, against 0.5 for chance, with an
interval of [0.778, 0.859], and it is the likelier one in 82% of pairs. The preference hardly
follows the pair's difference in chunk strength (r = 0.08), so it isn't familiarity
with chunks alone. This is one run, reported as it came out.

Qwen3.5-4B-Base, in bfloat16, also answers grammatical to nearly every string, with a
mean probability of 0.996 for grammatical strings against 0.992 for ungrammatical
ones, so both effects are detected but tiny. Its probability test is stronger than
GPT-2's: the grammatical string has a probability of 0.863 [0.830, 0.894], it is the
likelier one in 94% of pairs, and the preference doesn't follow the chunk difference
(r = 0.02).

## References

DeLosh, E. L., Busemeyer, J. R., & McDaniel, M. A. (1997). Extrapolation: The sine qua
non for abstraction in function learning. *Journal of Experimental Psychology:
Learning, Memory, and Cognition*, 23(4), 968–986.

Feldman, J. (2000). Minimization of Boolean complexity in human concept learning.
*Nature*, 407, 630–633.

Griffiths, T. L., Christian, B. R., & Kalish, M. L. (2008). Using category structures
to test iterated learning as a method for identifying inductive biases. *Cognitive
Science*, 32(1), 68–107.

Griffiths, T. L., & Kalish, M. L. (2007). Language evolution by iterated learning with
Bayesian agents. *Cognitive Science*, 31(3), 441–480.

Hu, J., & Levy, R. (2023). Prompting is not a substitute for probability measurements
in large language models. In *Proceedings of the 2023 Conference on Empirical Methods
in Natural Language Processing*.

Imel, N., & Zaslavsky, N. (2026). Evolution and compression in LLMs: On the emergence
of human-aligned categorization. In *International Conference on Learning
Representations*. arXiv:2509.08093.

Kalish, M. L., Griffiths, T. L., & Lewandowsky, S. (2007). Iterated learning:
Intergenerational knowledge transmission reveals inductive biases. *Psychonomic
Bulletin & Review*, 14(2), 288–294.

Kay, P., Berlin, B., Maffi, L., Merrifield, W. R., & Cook, R. (2009). *The World Color
Survey*. CSLI Publications.

Kirby, S., Cornish, H., & Smith, K. (2008). Cumulative cultural evolution in the
laboratory: An experimental approach to the origins of structure in human language.
*Proceedings of the National Academy of Sciences*, 105(31), 10681–10686.

Knowlton, B. J., & Squire, L. R. (1996). Artificial grammar learning depends on
implicit acquisition of both abstract and exemplar-specific information. *Journal of
Experimental Psychology: Learning, Memory, and Cognition*, 22(1), 169–181.

Kruschke, J. K. (1992). ALCOVE: An exemplar-based connectionist model of category
learning. *Psychological Review*, 99(1), 22–44.

Nosofsky, R. M. (1986). Attention, similarity, and the identification–categorization
relationship. *Journal of Experimental Psychology: General*, 115(1), 39–57.

Nosofsky, R. M., Gluck, M. A., Palmeri, T. J., McKinley, S. C., & Glauthier, P. (1994).
Comparing models of rule-based classification learning: A replication and extension
of Shepard, Hovland, and Jenkins (1961). *Memory & Cognition*, 22(3), 352–369.

Reber, A. S. (1967). Implicit learning of artificial grammars. *Journal of Verbal
Learning and Verbal Behavior*, 6(6), 855–863.

Ren, Y., Guo, S., Qiu, L., Wang, B., & Sutherland, D. J. (2024). Bias amplification in
language model evolution: An iterated learning perspective. In *Advances in Neural
Information Processing Systems 37*.

Shepard, R. N., Hovland, C. I., & Jenkins, H. M. (1961). Learning and memorization of
classifications. *Psychological Monographs: General and Applied*, 75(13), 1–42.

Xu, J., Dowman, M., & Griffiths, T. L. (2013). Cultural transmission results in
convergence towards colour term universals. *Proceedings of the Royal Society B*,
280(1758), 20123073.

Zaslavsky, N., Kemp, C., Regier, T., & Tishby, N. (2018). Efficient compression in
color naming and its evolution. *Proceedings of the National Academy of Sciences*,
115(31), 7937–7942.

Zhu, J.-Q., & Griffiths, T. L. (2024). Eliciting the priors of large language models
using iterated in-context learning. arXiv:2406.01860.
