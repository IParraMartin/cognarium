---
hide:
  - navigation
  - toc
---

<div class="cg-home" markdown>

<section class="cg-hero" markdown>

<div class="cg-hero__text" markdown>

# cognarium

<p class="cg-lede">Cognitive science experiments on language models and other learners.</p>

- 25 paradigms, from Stroop to false belief, each scored for the effect it's known for
- The same experiment on Hugging Face models, LSTMs, exemplar and Bayesian learners
- Logits, the logit lens and hidden states, saved as datasets you can share
- Open source under the Apache 2.0 licence

<div class="cg-actions" markdown>
[Get started](quickstart.md){ .md-button .md-button--primary }
[Browse paradigms](paradigms/control.md){ .md-button }
</div>

</div>

<figure class="cg-trial">
<div>
<p class="cg-trial__label">A person sees</p>
<div class="cg-screen cg-screen--hero" role="img" aria-label="The word RED printed in yellow ink on a grey screen">
<span class="cg-ink-yellow">RED</span>
</div>
</div>
<div>
<p class="cg-trial__label">GPT-2 reads</p>
<p class="cg-trial__prompt">The word RED is printed in yellow ink. What color is the ink?</p>
<ol class="cg-bars" aria-label="GPT-2's probability for each answer">
<li style="--p: 64%; --c: var(--cg-red)"><span>red</span><span class="cg-bars__track"><span class="cg-bars__fill"></span></span><span class="cg-bars__value">0.64</span></li>
<li style="--p: 11.5%; --c: var(--cg-blue)"><span>blue</span><span class="cg-bars__track"><span class="cg-bars__fill"></span></span><span class="cg-bars__value">0.12</span></li>
<li style="--p: 12.1%; --c: var(--cg-green)"><span>green</span><span class="cg-bars__track"><span class="cg-bars__fill"></span></span><span class="cg-bars__value">0.12</span></li>
<li class="cg-bars__answer" style="--p: 12.5%; --c: var(--cg-yellow)"><span>yellow</span><span class="cg-bars__track"><span class="cg-bars__fill"></span></span><span class="cg-bars__value">0.12</span></li>
</ol>
</div>
<figcaption>GPT-2 put 64% on the word and 12% on the ink. Both numbers come from one trial of the <a href="paradigms/control/#a-real-run">real run</a>.</figcaption>
</figure>

</section>

<section class="cg-band" markdown>

## Paradigms, grouped by what they measure

<div class="cg-cards" markdown>

<article class="cg-card" markdown>
<div class="cg-screen" aria-hidden="true"><span class="cg-specimen">HHCHH</span></div>

### [Cognitive control](paradigms/control.md)

Keeping to a rule when the stimulus pulls toward another response.

<div class="cg-chips" markdown>
[Stroop](paradigms/control.md#stroop)
[Flanker](paradigms/control.md#flanker)
[Simon](paradigms/control.md#simon)
[Go/no-go](paradigms/control.md#gonogo)
[Stop-signal](paradigms/control.md#stopsignal)
[Task switching](paradigms/control.md#taskswitching)
</div>
</article>

<article class="cg-card" markdown>
<div class="cg-screen" aria-hidden="true"><span class="cg-specimen cg-specimen--spaced">M K M</span></div>

### [Memory](paradigms/memory.md)

What a model keeps from earlier in its context, and what it fills in.

<div class="cg-chips" markdown>
[N-back](paradigms/memory.md#nback)
[Span](paradigms/memory.md#span)
[Sternberg](paradigms/memory.md#sternberg)
[Free recall](paradigms/memory.md#freerecall)
[DRM](paradigms/memory.md#drm)
</div>
</article>

<article class="cg-card" markdown>
<div class="cg-screen" aria-hidden="true">
  <svg class="cg-shapes" viewBox="0 0 200 60" focusable="false">
    <circle class="cg-fill-red" cx="24" cy="30" r="19"/>
    <rect class="cg-fill-blue" x="66" y="22" width="16" height="16" rx="2"/>
    <rect class="cg-fill-blue" x="106" y="11" width="38" height="38" rx="3"/>
    <circle class="cg-fill-red" cx="178" cy="30" r="8"/>
  </svg>
</div>

### [Learning and transmission](paradigms/learning.md)

Learning categories and grammars from examples, and passing them down a chain of learners.

<div class="cg-chips" markdown>
[Iterated learning](paradigms/learning.md#iteratedlearning)
[Category learning](paradigms/learning.md#categorylearning)
[Artificial grammar](paradigms/learning.md#artificialgrammar)
[Statistical learning](paradigms/learning.md#statisticallearning)
</div>
</article>

<article class="cg-card" markdown>
<div class="cg-screen" aria-hidden="true"><span class="cg-specimen cg-specimen--sentence">The soldier marched across the yard fainted.</span></div>

### [Language and semantics](paradigms/language.md)

Priming, grammaticality, word recognition and sentences that lead the reader astray.

<div class="cg-chips" markdown>
[Semantic priming](paradigms/language.md#semanticpriming)
[Structural priming](paradigms/language.md#structuralpriming)
[Lexical decision](paradigms/language.md#lexicaldecision)
[Garden path](paradigms/language.md#gardenpath)
[Minimal pairs](paradigms/language.md#minimalpairs)
[Fluency](paradigms/language.md#fluency)
</div>
</article>

<article class="cg-card" markdown>
<div class="cg-screen" aria-hidden="true"><span class="cg-specimen cg-specimen--sentence">More or less than 2000?</span></div>

### [Judgment and reasoning](paradigms/judgment.md)

Anchors, frames, conjunctions and what other people believe.

<div class="cg-chips" markdown>
[Anchoring](paradigms/judgment.md#anchoring)
[Framing](paradigms/judgment.md#framing)
[Conjunction fallacy](paradigms/judgment.md#conjunctionfallacy)
[False belief](paradigms/judgment.md#falsebelief)
</div>
</article>

<article class="cg-card" markdown>
<div class="cg-screen" aria-hidden="true"><span class="cg-specimen cg-specimen--fixation">+</span></div>

### [Your own paradigm](custom-paradigms.md)

Write a paradigm in about 20 lines, or run a built-in one on your own stimuli.

<div class="cg-chips" markdown>
[Write a paradigm](custom-paradigms.md)
[Stimuli from CSV or JSON](data.md)
[Open datasets](data.md#the-registry)
</div>
</article>

</div>

</section>

<section class="cg-run" markdown>

<div markdown>

## Run one

A run builds the trials, queries the model, scores the effect and saves everything
needed to repeat it. Run the same call again and it reads the saved run instead.

[Read the quickstart](quickstart.md)

</div>

<div markdown>

```python
import cognarium as cg

subject = cg.load("gpt2", device="auto")
results = cg.run(cg.paradigms.Stroop(), subject)
results.score()
```

</div>

</section>

<section class="cg-more" markdown>

<div markdown>

### Subjects

Hugging Face language models, LSTMs and your own PyTorch modules, exemplar and Bayesian
learners, on CPU, Apple MPS or CUDA. [Concepts](concepts.md)

</div>

<div markdown>

### Recording

Per-trial logits, the logit lens through the model's own head, and hidden states written
to disk as they arrive. [Recording](recording.md)

</div>

<div markdown>

### Data

CSV and JSON files become trials, and a registry fetches open datasets from their
authors, checked against a SHA-256 checksum. [Data](data.md)

</div>

<div markdown>

### Proxies, labelled

A model has no reaction times. Where the human effect is about speed, every score names
what its measure stands in for. [Metrics](metrics.md)

</div>

</section>

</div>
