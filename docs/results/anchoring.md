Model `gpt2` on `mps`, run on 2026-11-07 with cognarium 0.1.0.

Settings: `{"context": {"feedback": false, "keep": 0}, "max_new_tokens": 32, "n_participants": 1, "ordered": false, "participants_identical": false, "response_mode": "choice", "seed": 0, "stimuli": "built-in, not human-normed", "temperature": 1.0, "top_p": 1.0}`

Sample prompt, with the options `under 5000`, `5000 to under 8000`, `8000 to under 11000`, `11000 to under 14000`, `14000 or more`:

```text
Is the depth of the deepest point in the ocean, in metres more or less than 2000? Your estimate of the depth of the deepest point in the ocean, in metres:
```

| effect | by | level_a | level_b | mean_a | mean_b | difference | ci_low | ci_high | p_value | detected | measure | proxy_for |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| anchoring | anchor | high | low | 3.263 | 2.572 | 0.6911 | 0.4153 | 1.062 | 0.0004998 | True | expected_option |  |
