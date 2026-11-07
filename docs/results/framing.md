Model `gpt2` on `mps`, run on 2026-11-07 with cognarium 0.0.1.

Settings: `{"context": {"feedback": false, "keep": 0}, "max_new_tokens": 32, "n_participants": 1, "ordered": false, "participants_identical": false, "response_mode": "choice", "seed": 0, "stimuli": "built-in, not human-normed", "temperature": 1.0, "top_p": 1.0}`

Sample prompt, with the options `sure`, `gamble`:

```text
An oil spill is drifting toward a bay, putting 200 seabirds at risk. If plan A is adopted, 100 seabirds will be lost. If plan B is adopted, there is a one in two chance that none of the seabirds will be lost and a one in two chance that all 200 will be lost. Answer sure to choose plan A, or gamble to choose plan B.
Answer:
```

| effect | by | level_a | level_b | mean_a | mean_b | difference | ci_low | ci_high | p_value | detected | measure | proxy_for |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| framing | frame | gain | loss | 0.8562 | 0.8512 | 0.004988 | 0.001354 | 0.008987 | 0.01349 | True | option_probability |  |
