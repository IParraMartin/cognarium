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

<figure class="cg-sequence" aria-label="One Stroop trial: what a person sees, what GPT-2 reads, and how GPT-2 answers">
<div class="cg-frame">
<div class="cg-monitor"><span class="cg-word">RED</span></div>
<p class="cg-frame__caption">A person sees the word RED in blue ink.</p>
</div>
<div class="cg-frame">
<div class="cg-monitor cg-monitor--text"><p>The word RED is printed in blue ink. What color is the ink?<br>Response:<span class="cg-caret"></span></p></div>
<p class="cg-frame__caption">GPT-2 reads a description of it.</p>
</div>
<div class="cg-frame">
<div class="cg-monitor cg-monitor--bars">
<ol aria-label="GPT-2's probability for each ink">
<li><span>red</span><span class="cg-track"><span class="cg-fill" style="--p: 63.45%"></span></span><span class="cg-value">0.63</span></li>
<li class="cg-answer"><span>blue</span><span class="cg-track"><span class="cg-fill" style="--p: 17.26%"></span></span><span class="cg-value">0.17</span></li>
<li><span>green</span><span class="cg-track"><span class="cg-fill" style="--p: 11.31%"></span></span><span class="cg-value">0.11</span></li>
<li><span>yellow</span><span class="cg-track"><span class="cg-fill" style="--p: 7.98%"></span></span><span class="cg-value">0.08</span></li>
</ol>
</div>
<p class="cg-frame__caption">It gives each ink a probability, and most of it goes to the word.</p>
</div>
<figcaption>One trial from <a href="paradigms/control/#a-real-run">the real run of Stroop on GPT-2</a>, unedited.</figcaption>
</figure>

</section>

<section class="cg-index" id="paradigms">
<div class="cg-index__head">
<h2>Paradigms</h2>
<p>Each page says how the paradigm maps onto the human task, what the model version can't capture, and shows one run on GPT-2, whatever it found. The dots count the effects that run found: <span class="cg-dot cg-dot--on"></span> found, <span class="cg-dot"></span> not found.</p>
</div>
<div class="cg-index__cols">
<div class="cg-index__col">
<div class="cg-group">
<h3><a href="paradigms/control/">Cognitive control</a></h3>
<ul class="cg-rows">
<li><a href="paradigms/control/#stroop">Stroop</a><span class="cg-task">Name the ink colour of a colour word.</span><span class="cg-result" role="img" aria-label="2 of 2 effects found in GPT-2" title="2 of 2 effects found in GPT-2"><span class="cg-dot cg-dot--on"></span><span class="cg-dot cg-dot--on"></span></span></li>
<li><a href="paradigms/control/#flanker">Flanker</a><span class="cg-task">Name the middle letter of a row of letters.</span><span class="cg-result" role="img" aria-label="1 of 2 effects found in GPT-2" title="1 of 2 effects found in GPT-2"><span class="cg-dot"></span><span class="cg-dot cg-dot--on"></span></span></li>
<li><a href="paradigms/control/#simon">Simon</a><span class="cg-task">Press the key a colour is mapped to, wherever the shape appears.</span><span class="cg-result" role="img" aria-label="0 of 2 effects found in GPT-2" title="0 of 2 effects found in GPT-2"><span class="cg-dot"></span><span class="cg-dot"></span></span></li>
<li><a href="paradigms/control/#gonogo">Go/no-go</a><span class="cg-task">Press for every letter except one.</span><span class="cg-result" role="img" aria-label="2 of 2 effects found in GPT-2" title="2 of 2 effects found in GPT-2"><span class="cg-dot cg-dot--on"></span><span class="cg-dot cg-dot--on"></span></span></li>
<li><a href="paradigms/control/#stopsignal">Stop-signal</a><span class="cg-task">Press for each letter unless a stop cue follows it.</span><span class="cg-result" role="img" aria-label="1 of 2 effects found in GPT-2" title="1 of 2 effects found in GPT-2"><span class="cg-dot"></span><span class="cg-dot cg-dot--on"></span></span></li>
<li><a href="paradigms/control/#taskswitching">Task switching</a><span class="cg-task">Judge either the letter or the number of a pair, as each trial&#x27;s cue says.</span><span class="cg-result" role="img" aria-label="0 of 2 effects found in GPT-2" title="0 of 2 effects found in GPT-2"><span class="cg-dot"></span><span class="cg-dot"></span></span></li>
</ul>
</div>
<div class="cg-group">
<h3><a href="paradigms/memory/">Memory</a></h3>
<ul class="cg-rows">
<li><a href="paradigms/memory/#nback">N-back</a><span class="cg-task">Say whether each letter matches the one n letters back.</span><span class="cg-result" role="img" aria-label="0 of 2 effects found in GPT-2" title="0 of 2 effects found in GPT-2"><span class="cg-dot"></span><span class="cg-dot"></span></span></li>
<li><a href="paradigms/memory/#span">Span</a><span class="cg-task">Repeat a list of digits in order.</span><span class="cg-result" role="img" aria-label="0 of 1 effects found in GPT-2" title="0 of 1 effects found in GPT-2"><span class="cg-dot"></span></span></li>
<li><a href="paradigms/memory/#sternberg">Sternberg</a><span class="cg-task">Say whether a probe letter was in a memory set.</span><span class="cg-result" role="img" aria-label="0 of 2 effects found in GPT-2" title="0 of 2 effects found in GPT-2"><span class="cg-dot"></span><span class="cg-dot"></span></span></li>
<li><a href="paradigms/memory/#freerecall">Free recall</a><span class="cg-task">Study a list of words, then write every word you remember.</span><span class="cg-result" role="img" aria-label="0 of 2 effects found in GPT-2" title="0 of 2 effects found in GPT-2"><span class="cg-dot"></span><span class="cg-dot"></span></span></li>
<li><a href="paradigms/memory/#drm">DRM</a><span class="cg-task">Study lists of related words, then judge whether test words were on the list.</span><span class="cg-result" role="img" aria-label="0 of 2 effects found in GPT-2" title="0 of 2 effects found in GPT-2"><span class="cg-dot"></span><span class="cg-dot"></span></span></li>
</ul>
</div>
</div>
<div class="cg-index__col">
<div class="cg-group">
<h3><a href="paradigms/learning/">Learning and transmission</a></h3>
<ul class="cg-rows">
<li><a href="paradigms/learning/#iteratedlearning">Iterated learning</a><span class="cg-task">Each generation learns from the last generation&#x27;s responses.</span><span class="cg-result cg-result--none">chain measures</span></li>
<li><a href="paradigms/learning/#categorylearning">Category learning</a><span class="cg-task">Learn a category structure from corrective feedback, block by block.</span><span class="cg-result" role="img" aria-label="0 of 1 effects found in GPT-2" title="0 of 1 effects found in GPT-2"><span class="cg-dot"></span></span></li>
<li><a href="paradigms/learning/#artificialgrammar">Artificial grammar</a><span class="cg-task">Study strings from a finite-state grammar, then judge new ones.</span><span class="cg-result" role="img" aria-label="1 of 1 effects found in GPT-2" title="1 of 1 effects found in GPT-2"><span class="cg-dot cg-dot--on"></span></span></li>
<li><a href="paradigms/learning/#statisticallearning">Statistical learning</a><span class="cg-task">Find the words in a stream of syllables, then judge words and part-words.</span><span class="cg-result" role="img" aria-label="0 of 1 effects found in GPT-2" title="0 of 1 effects found in GPT-2"><span class="cg-dot"></span></span></li>
</ul>
</div>
<div class="cg-group">
<h3><a href="paradigms/language/">Language and semantics</a></h3>
<ul class="cg-rows">
<li><a href="paradigms/language/#semanticpriming">Semantic priming</a><span class="cg-task">Decide whether a target is a word, after a related or an unrelated prime.</span><span class="cg-result" role="img" aria-label="1 of 1 effects found in GPT-2" title="1 of 1 effects found in GPT-2"><span class="cg-dot cg-dot--on"></span></span></li>
<li><a href="paradigms/language/#structuralpriming">Structural priming</a><span class="cg-task">Read a dative sentence, then choose between two forms of another.</span><span class="cg-result" role="img" aria-label="1 of 1 effects found in GPT-2" title="1 of 1 effects found in GPT-2"><span class="cg-dot cg-dot--on"></span></span></li>
<li><a href="paradigms/language/#lexicaldecision">Lexical decision</a><span class="cg-task">Decide whether a string of letters is a word.</span><span class="cg-result" role="img" aria-label="0 of 1 effects found in GPT-2" title="0 of 1 effects found in GPT-2"><span class="cg-dot"></span></span></li>
<li><a href="paradigms/language/#gardenpath">Garden path</a><span class="cg-task">Score the word that resolves a temporarily ambiguous sentence.</span><span class="cg-result" role="img" aria-label="1 of 1 effects found in GPT-2" title="1 of 1 effects found in GPT-2"><span class="cg-dot cg-dot--on"></span></span></li>
<li><a href="paradigms/language/#minimalpairs">Minimal pairs</a><span class="cg-task">Prefer the acceptable sentence of a pair that differs in one place.</span><span class="cg-result" role="img" aria-label="1 of 1 effects found in GPT-2" title="1 of 1 effects found in GPT-2"><span class="cg-dot cg-dot--on"></span></span></li>
<li><a href="paradigms/language/#fluency">Fluency</a><span class="cg-task">Name as many words as you can from a category, or starting with a letter.</span><span class="cg-result" role="img" aria-label="0 of 1 effects found in GPT-2" title="0 of 1 effects found in GPT-2"><span class="cg-dot"></span></span></li>
</ul>
</div>
<div class="cg-group">
<h3><a href="paradigms/judgment/">Judgment and reasoning</a></h3>
<ul class="cg-rows">
<li><a href="paradigms/judgment/#anchoring">Anchoring</a><span class="cg-task">Compare a quantity with a number, then estimate it.</span><span class="cg-result" role="img" aria-label="1 of 1 effects found in GPT-2" title="1 of 1 effects found in GPT-2"><span class="cg-dot cg-dot--on"></span></span></li>
<li><a href="paradigms/judgment/#framing">Framing</a><span class="cg-task">Choose between a sure outcome and a gamble, described as gains or as losses.</span><span class="cg-result" role="img" aria-label="1 of 1 effects found in GPT-2" title="1 of 1 effects found in GPT-2"><span class="cg-dot cg-dot--on"></span></span></li>
<li><a href="paradigms/judgment/#conjunctionfallacy">Conjunction fallacy</a><span class="cg-task">Judge whether a single property or that property joined with another is likelier.</span><span class="cg-result" role="img" aria-label="0 of 2 effects found in GPT-2" title="0 of 2 effects found in GPT-2"><span class="cg-dot"></span><span class="cg-dot"></span></span></li>
<li><a href="paradigms/judgment/#falsebelief">False belief</a><span class="cg-task">Say where someone will look for an object that was moved, seen or unseen.</span><span class="cg-result" role="img" aria-label="1 of 2 effects found in GPT-2" title="1 of 2 effects found in GPT-2"><span class="cg-dot cg-dot--on"></span><span class="cg-dot"></span></span></li>
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

```python
import cognarium as cg

subject = cg.load("gpt2", device="auto")
results = cg.run(cg.paradigms.Stroop(), subject)
results.score()
```

</div>

<div markdown>

<p class="cg-run__label"><code>results.score()</code> on GPT-2, from the real run</p>

| effect | measure | difference | 95% interval | detected |
|---|---|---|---|---|
| interference | answer surprisal | 1.54 | 0.98 to 2.02 | yes |
| errors | answer margin | 2.12 | 1.23 to 2.94 | yes |

A model has no reaction times, so each score names the human measure it stands in for.
[Read the quickstart](quickstart.md) for the rest of the table.

</div>

</section>

</div>
