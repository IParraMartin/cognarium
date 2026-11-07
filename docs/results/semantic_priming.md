Model `gpt2` on `mps`, run on 2026-11-07 with cognarium 0.0.1.

Settings: `{"context": {"feedback": false, "keep": 0}, "max_new_tokens": 32, "n_participants": 1, "ordered": false, "participants_identical": false, "response_mode": "choice", "seed": 0, "stimuli": "built-in, not human-normed", "temperature": 1.0, "top_p": 1.0}`

Sample prompt, with the options `word`, `nonword`:

```text
Prime: rabbit. Target: vesk. Is the target a word?
Response:
```

| effect | by | level_a | level_b | mean_a | mean_b | difference | ci_low | ci_high | p_value | detected | measure | proxy_for |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| priming | relatedness | unrelated | related | 0.0008933 | 0.001128 | -0.0002342 | -0.00041 | -4.602e-05 | 0.02499 | True | answer_surprisal | reaction time |
