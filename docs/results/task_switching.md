Model `gpt2` on `mps`, run on 2026-11-07 with cognarium 0.1.0.

Settings: `{"context": {"feedback": false, "keep": 4}, "max_new_tokens": 32, "n_participants": 1, "ordered": true, "parameters": {"n_trials": 40}, "participants_identical": false, "response_mode": "choice", "seed": 0, "stimuli": "built-in, not human-normed", "temperature": 1.0, "top_p": 1.0}`

Sample prompt, with the options `yes`, `no`:

```text
Task: number. K5 Is the number even?
Response:
```

| effect | by | level_a | level_b | mean_a | mean_b | difference | ci_low | ci_high | p_value | detected | measure | proxy_for |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| switch_cost | transition | switch | repeat | 2.31 | 1.82 | 0.4898 | -1.084 | 2.057 | 0.5507 | False | answer_surprisal | reaction time |
| errors | transition | repeat | switch | 0.865 | 0.008899 | 0.8561 | -1.141 | 2.899 | 0.4188 | False | answer_margin | error rate |
