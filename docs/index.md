---
hide:
  - navigation
  - toc
---

<div class="cg-home" markdown>

<section class="cg-hero" markdown>

<div class="cg-hero__art" aria-hidden="true"><canvas class="cg-hero__structure"></canvas><canvas class="cg-hero__activity"></canvas></div>

# Cognitive science experiments on language models

<p class="cg-lede">Learning experiments from cognitive science, run in context on any Hugging Face language model, and with the same trials on LSTMs, exemplar and Bayesian learners.</p>

<div class="cg-actions" markdown>

```bash
uv add cognarium
```

[Read the quickstart](quickstart.md){ .md-button .md-button--primary }
[Browse the paradigms](#paradigms){ .md-button }

</div>

<div class="cg-hero__controls"><button class="cg-hero__toggle" type="button" aria-pressed="false" aria-label="Pause the animation" title="Pause the animation" hidden><svg class="cg-hero__pause" viewBox="0 0 16 16" aria-hidden="true"><rect x="4" y="3" width="2.6" height="10" rx="0.8"/><rect x="9.4" y="3" width="2.6" height="10" rx="0.8"/></svg><svg class="cg-hero__play" viewBox="0 0 16 16" aria-hidden="true"><path d="M5 3.2v9.6a.7.7 0 0 0 1.06.6l7.6-4.8a.7.7 0 0 0 0-1.2l-7.6-4.8A.7.7 0 0 0 5 3.2Z"/></svg></button></div>

</section>

<section class="cg-index" id="paradigms">
<div class="cg-index__head">
<h2>Paradigms</h2>
<p>Each section says what the paradigm does, how it maps onto the human task, the work it builds on, and what the model version can't capture. The paradigms run with the same trials and scoring on four kinds of learner.</p>
</div>
<div class="cg-index__cols">
<div class="cg-index__col">
<div class="cg-group">
<h3><a href="paradigms/learning/">Learning and transmission</a></h3>
<ul class="cg-rows">
<li><a href="paradigms/learning/#iteratedlearning">Iterated learning</a><span class="cg-task">Each generation learns from the last generation&#x27;s answers, for any category system or function. After Kirby, Cornish and Smith (2008) and Griffiths and Kalish (2007).</span></li>
<li><a href="paradigms/learning/#artificialgrammar">Artificial grammar</a><span class="cg-task">Study strings from a finite-state grammar, then judge new ones, or compare them by probability. After Reber (1967) and Knowlton and Squire (1996).</span></li>
</ul>
</div>
</div>
<div class="cg-index__col">
<div class="cg-group">
<h3><a href="concepts/#subjects">Learners</a></h3>
<ul class="cg-rows">
<li><a href="concepts/#subjects">Hugging Face models</a><span class="cg-task">Any causal language model, learning in context from its prompt, with logits, hidden states and the logit lens recorded.</span></li>
<li><a href="multimodal/">Multimodal models</a><span class="cg-task">Vision-language models that see images and video in their prompt, alone or mixed.</span></li>
<li><a href="concepts/#subjects">PyTorch modules</a><span class="cg-task">The bundled LSTM or your own PyTorch module, learning in context or by training its weights.</span></li>
<li><a href="concepts/#subjects">Exemplar learner</a><span class="cg-task">Labels by similarity to the examples it has stored, as in the Generalized Context Model (Nosofsky, 1986).</span></li>
<li><a href="concepts/#subjects">Bayesian learner</a><span class="cg-task">Infers a labelling from a space of hypotheses and samples, averages or takes the most probable (Griffiths and Kalish, 2007).</span></li>
</ul>
</div>
</div>
</div>
</section>

<section class="cg-run" markdown>

<div markdown>

## Run one

These four lines run artificial grammar learning on GPT-2. The run builds the trials,
queries the model, scores the effects and saves everything to
`runs/artificial_grammar/`, so running it again reads the saved run instead.

[Read the quickstart](quickstart.md) for what the score table holds.

</div>

<div markdown>

```python
import cognarium as cg

subject = cg.load("gpt2", device="auto")
results = cg.run(cg.paradigms.ArtificialGrammar(), subject)
results.score()
```

</div>

</section>

</div>

