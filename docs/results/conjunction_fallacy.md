Model `gpt2` on `mps`, run on 2026-11-07 with cognarium 0.0.1.

Settings: `{"context": {"feedback": false, "keep": 0}, "max_new_tokens": 32, "n_participants": 1, "ordered": false, "participants_identical": false, "response_mode": "choice", "seed": 0, "stimuli": "built-in, not human-normed", "temperature": 1.0, "top_p": 1.0}`

Sample prompt, with the options `1`, `2`:

```text
Lucia is 36 and lives in a coastal city.
Which is more likely?
1: Lucia is a pharmacist and fosters rescue animals.
2: Lucia is a pharmacist.
Answer:
```

| effect | by | level_a | level_b | mean_a | mean_b | difference | ci_low | ci_high | p_value | detected | measure | proxy_for |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fallacy | description | neutral | representative | 0.512 | 0.4618 | 0.05012 | -0.02989 | 0.1258 | 0.2404 | False | answer_probability |  |
| error_rate |  | all | chance | 0.4618 | 0.5 | -0.03816 | -0.07967 | 0.004651 | 0.1039 | False | answer_probability |  |
