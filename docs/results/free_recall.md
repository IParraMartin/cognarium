Model `gpt2` on `mps`, run on 2026-11-07 with cognarium 0.1.0.

Settings: `{"context": {"feedback": false, "keep": 0}, "max_new_tokens": 64, "n_participants": 1, "ordered": false, "parameters": {"list_length": 12, "n_lists": 8}, "participants_identical": false, "response_mode": "generate", "seed": 0, "stimuli": "built-in, not human-normed", "temperature": 1.0, "top_p": 1.0}`

Sample prompt, answered by free generation:

```text
Study these words: piano, rabbit, elbow, mug, crown, window, guitar, button, ticket, lock, zipper, cushion. Now write every word you remember, separated by commas.
Response:
```

| effect | by | level_a | level_b | mean_a | mean_b | difference | ci_low | ci_high | p_value | detected | measure | proxy_for |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| recency | third | late | middle | 0 | 0 | 0 | 0 | 0 | 1 | False | recall |  |
| primacy | third | early | middle | 0.03125 | 0 | 0.03125 | 0 | 0.09375 | 1 | False | recall |  |
