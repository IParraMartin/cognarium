# Logits, the logit lens and hidden states

## Logits

`record=["logits"]` keeps the model's full next-token logits for every query, beside the
scores over the options that every run keeps.

```python
import cognarium as cg

subject = cg.load("gpt2", device="auto")
results = cg.run(cg.paradigms.ArtificialGrammar(), subject, record=["logits", "hidden_states"])
results.logits(0).shape
```

## Hidden states

`record=["hidden_states"]` stores one row per query for each layer, as `float16` by
default. `layers` picks which layers, `pool` keeps the `"last"` token, the `"mean"`
over tokens, or every token with None, and `state_dtype` sets the precision. A
background thread writes them to disk as they arrive, so memory stays flat.

```python
hidden = results.hidden
hidden.layers[:3]
```

Exemplar learners store their similarity to each stored example, and Bayesian learners
their posterior, in the same format, so any two subjects can be compared.

## The logit lens

`results.lens()` reads each layer's state through the model's own final norm and output
head. For every query and layer, it reports the top token, the entropy over the whole
vocabulary, and the log probabilities of the options, normalised over them. The last
layer must reproduce the model's own logits, so a model that rescales them after its
head is refused rather than misread.

```python
lens = results.lens()
lens.head()
```

## Saving and sharing

A store saves to a folder of safetensors shards with a Parquet index and a manifest, and
opens lazily.

```python
saved = hidden.save("grammar_gpt2")
store = cg.HiddenStates.open("grammar_gpt2")
states = {layer: store.to_tensor(layer) for layer in store.layers}
cg.metrics.layer_matrix(states, "linear_cka")
```

<!-- no-run -->
```python
store.push_to_hub("your-name/grammar_gpt2")
```

## Other models

Models outside the known architectures can be described with
`cg.load(model, layer_map=cg.LayerMap(...))`, and custom PyTorch modules with
`cg.load(module, tokenizer=tokenizer, layers={"name": "path"}, readout="path")`.
