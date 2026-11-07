Model `gpt2` on `mps`, run on 2026-11-07 with cognarium 0.0.1.

Settings: `{"context": {"feedback": false, "keep": 8}, "max_new_tokens": 32, "n_participants": 1, "ordered": true, "parameters": {"delays": [1, 5], "letters": ["K", "M", "P", "R", "T", "L", "N", "B"], "n_trials": 40, "stop_ratio": 0.25}, "participants_identical": false, "response_mode": "choice", "seed": 0, "stimuli": "built-in, not human-normed", "temperature": 1.0, "top_p": 1.0}`

Sample prompt, with the options `press`, `withhold`:

```text
Press unless you see STOP. Letter: B . . . . . STOP
Response:
```

| effect | by | level_a | level_b | mean_a | mean_b | difference | ci_low | ci_high | p_value | detected | measure | proxy_for |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| late_signal | delay | 5 | 1 | 0.999 | 1 | -0.001025 | -0.003083 | 6.592e-06 | 0.7031 | False | option_probability |  |
| stopping | trial_type | go | stop | 16.35 | -12.31 | 28.67 | 26.73 | 30.45 | 0.0004998 | True | answer_margin | reaction time |
