Model `gpt2` on `mps`, run on 2026-11-07 with cognarium 0.0.1.

Settings: `{"context": {"feedback": false, "keep": 0}, "max_new_tokens": 32, "n_participants": 1, "ordered": false, "participants_identical": false, "response_mode": "choice", "seed": 0, "stimuli": "built-in, not human-normed", "temperature": 1.0, "top_p": 1.0}`

Sample prompt, with the options `The porter brought a suitcase to the traveler.`, `The porter brought the traveler a suitcase.`:

```text
Sentence: The waiter brought the couple a menu.
Sentence:
```

| effect | by | level_a | level_b | mean_a | mean_b | difference | ci_low | ci_high | p_value | detected | measure | proxy_for |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| priming | prime_structure | po | do | 0.8789 | 0.09922 | 0.7797 | 0.693 | 0.8536 | 0.0004998 | True | option_probability |  |
