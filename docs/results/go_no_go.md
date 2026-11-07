Model `gpt2` on `mps`, run on 2026-11-07 with cognarium 0.0.1.

Settings: `{"context": {"feedback": false, "keep": 8}, "max_new_tokens": 32, "n_participants": 1, "ordered": true, "parameters": {"go_ratio": 0.75, "letters": ["K", "M", "P", "R", "T", "L", "N", "B"], "n_trials": 40, "no_go_letter": "X"}, "participants_identical": false, "response_mode": "choice", "seed": 0, "stimuli": "built-in, not human-normed", "temperature": 1.0, "top_p": 1.0}`

Sample prompt, with the options `press`, `withhold`:

```text
Press unless the letter is X. Letter: B
Response:
```

| effect | by | level_a | level_b | mean_a | mean_b | difference | ci_low | ci_high | p_value | detected | measure | proxy_for |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| inhibition_cost | trial_type | no_go | go | 23.04 | 0.0004531 | 23.04 | 20.99 | 25.03 | 0.0004998 | True | answer_surprisal | reaction time |
| errors | trial_type | go | no_go | 16.69 | -15.97 | 32.66 | 30.87 | 34.33 | 0.0004998 | True | answer_margin | reaction time |
