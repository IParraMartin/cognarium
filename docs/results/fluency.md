Model `gpt2` on `mps`, run on 2026-11-07 with cognarium 0.1.0.

Settings: `{"context": {"feedback": false, "keep": 0}, "max_new_tokens": 64, "n_participants": 1, "ordered": false, "parameters": {"letters": ["f", "a", "s"], "semantic": ["animals", "fruits", "tools"]}, "participants_identical": false, "response_mode": "generate", "seed": 0, "stimuli": "built-in, not human-normed", "temperature": 1.0, "top_p": 1.0}`

Sample prompt, answered by free generation:

```text
List as many words starting with a as you can, separated by commas:
Response:
```

| effect | by | level_a | level_b | mean_a | mean_b | difference | ci_low | ci_high | p_value | detected | measure | proxy_for |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fluency_gap | kind | semantic | letter | 1 | 0.3333 | 0.6667 | -0.6667 | 2.667 | 1 | False | n_produced |  |
