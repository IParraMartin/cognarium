Model `gpt2` on `mps`, run on 2026-11-07 with cognarium 0.0.1.

Settings: `{"context": {"feedback": false, "keep": 0}, "max_new_tokens": 32, "n_participants": 1, "ordered": false, "parameters": {"flankers": 2, "letters": ["H", "S", "K", "C"], "n_repeats": 1}, "participants_identical": false, "response_mode": "choice", "seed": 0, "stimuli": "built-in, not human-normed", "temperature": 1.0, "top_p": 1.0}`

Sample prompt, with the options `H`, `S`, `K`, `C`:

```text
HHCHH What is the middle letter?
Response:
```

| effect | by | level_a | level_b | mean_a | mean_b | difference | ci_low | ci_high | p_value | detected | measure | proxy_for |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| interference | congruency | incongruent | congruent | 1.786 | 1.382 | 0.4046 | -0.04725 | 0.8151 | 0.1924 | False | answer_surprisal | reaction time |
| errors | congruency | congruent | incongruent | 0.3924 | -0.1412 | 0.5337 | 0.07132 | 0.988 | 0.1384 | True | answer_margin | reaction time |
