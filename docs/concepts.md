# Concepts

An experiment has four parts: stimuli, trials, a paradigm that orders them and says what
to score, and a subject that responds.

## Stimuli and trials

A `Stimulus` holds any mix of `text`, `image`, `audio` and `features`, a numeric vector.
`features` lets an exemplar or Bayesian learner read the same trial a language model
reads as text. A `Trial` adds the response options, the correct answer if there is one,
condition labels and an item ID.

```python
import cognarium as cg

trial = cg.Trial(
    cg.Stimulus(text="Letters: T X S"),
    options=("grammatical", "ungrammatical"),
    answer="grammatical",
    condition={"status": "grammatical"},
    item="TXS"
)
trial.options
```

## Paradigms

A paradigm makes the trials, decides what stays in the context between them, and
declares the effects it scores. Settings that shape a run live on the paradigm:

| Setting | Meaning |
|---|---|
| `stimuli` | Your own rows, as a file, a table, a `Dataset` or a list of mappings |
| `n_participants` | How many virtual participants to run, each with its own seed and trial order |
| `personas` | Text placed before each participant's prompts, one per participant |
| `instructions` | Text placed before every prompt |
| `examples` | Demonstrations shown before the trials |
| `temperature`, `top_p`, `top_k` | How the response distribution is shaped before sampling |
| `max_new_tokens` | The length limit for paradigms answered by free generation |
| `seed` | Seeds everything random in the experiment |

```python
paradigm = cg.paradigms.ArtificialGrammar(
    instructions="Judge whether each string follows the rules of the studied strings.",
    temperature=0.7,
    n_participants=2,
    seed=1
)
```

The context policy says what a subject sees of earlier trials: `cg.context.isolated`
shows nothing, `cg.context.cumulative` shows everything since the block began, and
`cg.context.window(k)` shows the last `k` trials. With `feedback=True`, earlier trials
show their correct answer instead of the subject's own.

## Subjects

A subject is anything that gives probabilities over a trial's options. Other abilities,
such as recording hidden states, are optional, and a run that needs one the subject lacks
fails before it starts, naming what's missing.

```python
gpt2 = cg.load("gpt2", device="auto")
exemplar = cg.subjects.ExemplarLearner(c=2.0)
bayes = cg.subjects.BayesianLearner(decision="sample")
```

| Subject | Learns by | Records |
|---|---|---|
| Hugging Face model | `in_context`: reading earlier trials in its prompt | Layer activations, logit lens |
| LSTM or custom module | `in_context`, or `weights`: training on the trials | Layer activations, and the logit lens when a readout is given |
| Exemplar learner (GCM) | `exemplar`: storing labelled examples | Similarity to stored examples |
| Bayesian learner | `posterior`: updating a posterior over hypotheses | The posterior |

`cg.run(..., learning="auto")` picks each subject's natural way of learning.

## Responses

Paradigms are forced choice by default: every option is scored and the scores are
normalised over the options, so the response is a distribution on every kind of subject.
A paradigm that sets `response_mode = "generate"` is answered by free generation
instead, on neural subjects only, and its output is parsed afterwards. Temperature, top-k and top-p are applied by the runner, so
they behave the same on every subject, and the raw scores are always kept beside the
shaped ones.

## Virtual participants

A model gives the same probabilities every time it sees the same prompt. Participants
differ through their seed, their trial order, and when given, their persona. With
isolated trials and no personas, every participant sees the same prompts, so cognarium
warns, and refuses participant-level intervals rather than reporting one of zero width.

```python
results = cg.run(cg.paradigms.ArtificialGrammar(), gpt2)
results.contrast(
    "option_probability",
    by="status",
    levels=("grammatical", "ungrammatical"),
    option="grammatical",
    ci_by="item"
)
```
