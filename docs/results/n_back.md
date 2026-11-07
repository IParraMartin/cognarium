Model `gpt2` on `mps`, run on 2026-11-07 with cognarium 0.0.1.

Settings: `{"context": {"feedback": false, "keep": null}, "max_new_tokens": 32, "n_participants": 1, "ordered": true, "parameters": {"alphabet": ["B", "F", "H", "K", "L", "M", "Q", "R"], "levels": [1, 2], "sequence_length": 20, "target_rate": 0.33}, "participants_identical": false, "response_mode": "choice", "seed": 0, "stimuli": "built-in, not human-normed", "temperature": 1.0, "top_p": 1.0}`

Sample prompt, with the options `yes`, `no`:

```text
1-back. Letter: M Does it match the letter 1 back?
Response:
```

| effect | by | level_a | level_b | mean_a | mean_b | difference | ci_low | ci_high | p_value | detected | measure | proxy_for |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| load_cost | load | 2 | 1 | 2.951 | 2.549 | 0.4026 | -1.135 | 1.886 | 0.5987 | False | answer_surprisal | reaction time |
| errors | load | 1 | 2 | -0.4366 | -0.8381 | 0.4015 | -1.591 | 2.28 | 0.6997 | False | answer_margin | reaction time |
