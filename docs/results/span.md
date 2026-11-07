Model `gpt2` on `mps`, run on 2026-11-07 with cognarium 0.1.0.

Settings: `{"context": {"feedback": false, "keep": 0}, "max_new_tokens": 18, "n_participants": 1, "ordered": false, "parameters": {"lengths": [3, 5, 7], "n_per_length": 8}, "participants_identical": false, "response_mode": "generate", "seed": 0, "stimuli": "built-in, not human-normed", "temperature": 1.0, "top_p": 1.0}`

Sample prompt, answered by free generation:

```text
Remember these digits: 6 3 4 1 7. Repeat them in order.
Response:
```

| effect | by | level_a | level_b | mean_a | mean_b | difference | ci_low | ci_high | p_value | detected | measure | proxy_for |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| length_cost | length | 3 | 7 | 0 | 0 | 0 | 0 | 0 | 1 | False | correct |  |
