---
hide:
  - navigation
  - toc
---

<div class="cg-home" markdown>

<section class="cg-hero" markdown>

# Cognitive science experiments on language models

<p class="cg-lede">Learning experiments from cognitive science, run in context on any Hugging Face language model, and with the same trials on LSTMs, exemplar and Bayesian learners.</p>

<div class="cg-actions" markdown>

```bash
uv add cognarium
```

[Read the quickstart](quickstart.md){ .md-button .md-button--primary }
[Browse the paradigms](#paradigms){ .md-button }

</div>

</section>

<section class="cg-index" id="paradigms">
<div class="cg-index__head">
<h2>Paradigms</h2>
<p>Each section says what the paradigm does, how it maps onto the human task, the work it builds on, and what the model version can't capture.</p>
</div>
<div class="cg-index__cols">
<div class="cg-index__col">
<div class="cg-group">
<h3><a href="paradigms/learning/">Learning and transmission</a></h3>
<ul class="cg-rows">
<li><a href="paradigms/learning/#iteratedlearning">Iterated learning</a><span class="cg-task">Each generation learns from the last generation&#x27;s answers, for any category system or function. After Kirby, Cornish and Smith (2008) and Griffiths and Kalish (2007).</span></li>
<li><a href="paradigms/learning/#categorylearning">Category learning</a><span class="cg-task">Learn the Shepard structures, any category system or a function from corrective feedback. After Shepard, Hovland and Jenkins (1961) and Nosofsky et al. (1994).</span></li>
<li><a href="paradigms/learning/#artificialgrammar">Artificial grammar</a><span class="cg-task">Study strings from a finite-state grammar, then judge new ones, or compare them by probability. After Reber (1967) and Knowlton and Squire (1996).</span></li>
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

