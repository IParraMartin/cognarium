---
hide:
  - navigation
  - toc
---

<div class="cg-home" markdown>

<section class="cg-hero" markdown>

# Cognitive science experiments on language models

<p class="cg-lede">25 classic paradigms, from Stroop to false belief, on any Hugging Face language model. The learning paradigms also run on LSTMs, exemplar and Bayesian learners.</p>

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
<p>Each page says how the paradigm maps onto the human task and what the model version can't capture.</p>
</div>
<div class="cg-index__cols">
<div class="cg-index__col">
<div class="cg-group">
<h3><a href="paradigms/control/">Cognitive control</a></h3>
<ul class="cg-rows">
<li><a href="paradigms/control/#stroop">Stroop</a><span class="cg-task">Name the ink colour of a colour word.</span></li>
<li><a href="paradigms/control/#flanker">Flanker</a><span class="cg-task">Name the middle letter of a row of letters.</span></li>
<li><a href="paradigms/control/#simon">Simon</a><span class="cg-task">Press the key a colour is mapped to, wherever the shape appears.</span></li>
<li><a href="paradigms/control/#gonogo">Go/no-go</a><span class="cg-task">Press for every letter except one.</span></li>
<li><a href="paradigms/control/#stopsignal">Stop-signal</a><span class="cg-task">Press for each letter unless a stop cue follows it.</span></li>
<li><a href="paradigms/control/#taskswitching">Task switching</a><span class="cg-task">Judge either the letter or the number of a pair, as each trial&#x27;s cue says.</span></li>
</ul>
</div>
<div class="cg-group">
<h3><a href="paradigms/memory/">Memory</a></h3>
<ul class="cg-rows">
<li><a href="paradigms/memory/#nback">N-back</a><span class="cg-task">Say whether each letter matches the one n letters back.</span></li>
<li><a href="paradigms/memory/#span">Span</a><span class="cg-task">Repeat a list of digits in order.</span></li>
<li><a href="paradigms/memory/#sternberg">Sternberg</a><span class="cg-task">Say whether a probe letter was in a memory set.</span></li>
<li><a href="paradigms/memory/#freerecall">Free recall</a><span class="cg-task">Study a list of words, then write every word you remember.</span></li>
<li><a href="paradigms/memory/#drm">DRM</a><span class="cg-task">Study lists of related words, then judge whether test words were on the list.</span></li>
</ul>
</div>
</div>
<div class="cg-index__col">
<div class="cg-group">
<h3><a href="paradigms/learning/">Learning and transmission</a></h3>
<ul class="cg-rows">
<li><a href="paradigms/learning/#iteratedlearning">Iterated learning</a><span class="cg-task">Each generation learns from the last generation&#x27;s responses.</span></li>
<li><a href="paradigms/learning/#categorylearning">Category learning</a><span class="cg-task">Learn a category structure from corrective feedback, block by block.</span></li>
<li><a href="paradigms/learning/#artificialgrammar">Artificial grammar</a><span class="cg-task">Study strings from a finite-state grammar, then judge new ones.</span></li>
<li><a href="paradigms/learning/#statisticallearning">Statistical learning</a><span class="cg-task">Find the words in a stream of syllables, then judge words and part-words.</span></li>
</ul>
</div>
<div class="cg-group">
<h3><a href="paradigms/language/">Language and semantics</a></h3>
<ul class="cg-rows">
<li><a href="paradigms/language/#semanticpriming">Semantic priming</a><span class="cg-task">Decide whether a target is a word, after a related or an unrelated prime.</span></li>
<li><a href="paradigms/language/#structuralpriming">Structural priming</a><span class="cg-task">Read a dative sentence, then choose between two forms of another.</span></li>
<li><a href="paradigms/language/#lexicaldecision">Lexical decision</a><span class="cg-task">Decide whether a string of letters is a word.</span></li>
<li><a href="paradigms/language/#gardenpath">Garden path</a><span class="cg-task">Score the word that resolves a temporarily ambiguous sentence.</span></li>
<li><a href="paradigms/language/#minimalpairs">Minimal pairs</a><span class="cg-task">Prefer the acceptable sentence of a pair that differs in one place.</span></li>
<li><a href="paradigms/language/#fluency">Fluency</a><span class="cg-task">Name as many words as you can from a category, or starting with a letter.</span></li>
</ul>
</div>
<div class="cg-group">
<h3><a href="paradigms/judgment/">Judgment and reasoning</a></h3>
<ul class="cg-rows">
<li><a href="paradigms/judgment/#anchoring">Anchoring</a><span class="cg-task">Compare a quantity with a number, then estimate it.</span></li>
<li><a href="paradigms/judgment/#framing">Framing</a><span class="cg-task">Choose between a sure outcome and a gamble, described as gains or as losses.</span></li>
<li><a href="paradigms/judgment/#conjunctionfallacy">Conjunction fallacy</a><span class="cg-task">Judge whether a single property or that property joined with another is likelier.</span></li>
<li><a href="paradigms/judgment/#falsebelief">False belief</a><span class="cg-task">Say where someone will look for an object that was moved, seen or unseen.</span></li>
</ul>
</div>
</div>
</div>
</section>

<section class="cg-run" markdown>

<div markdown>

## Run one

These four lines run Stroop on GPT-2. The run builds the trials, queries the model,
scores the effects and saves everything to `runs/stroop/`, so running it again reads
the saved run instead.

A model has no reaction times, so each score names the human measure it stands in for.
[Read the quickstart](quickstart.md) for what the score table holds.

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

</div>

