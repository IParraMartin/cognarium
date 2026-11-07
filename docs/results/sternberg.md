Model `gpt2` on `mps`, run on 2026-11-07 with cognarium 0.0.1.

Settings: `{"context": {"feedback": false, "keep": 0}, "max_new_tokens": 32, "n_participants": 1, "ordered": false, "parameters": {"n_per_size": 16, "set_sizes": [2, 6]}, "participants_identical": false, "response_mode": "choice", "seed": 0, "stimuli": "built-in, not human-normed", "temperature": 1.0, "top_p": 1.0}`

Sample prompt, with the options `yes`, `no`:

```text
Memory set: D P. Probe: P. Was the probe in the set?
Response:
```

| effect | by | level_a | level_b | mean_a | mean_b | difference | ci_low | ci_high | p_value | detected | measure | proxy_for |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| set_size_cost | set_size | 6 | 2 | 1.184 | 1.181 | 0.003658 | -0.4604 | 0.5248 | 0.9695 | False | answer_surprisal | reaction time |
| errors | set_size | 2 | 6 | 0.004645 | -0.009172 | 0.01382 | -0.7149 | 0.6791 | 0.9015 | False | answer_margin | reaction time |
