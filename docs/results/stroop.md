Model `gpt2` on `mps`, run on 2026-11-07 with cognarium 0.0.1.

Settings: `{"context": {"feedback": false, "keep": 0}, "max_new_tokens": 32, "n_participants": 1, "ordered": false, "parameters": {"colors": ["red", "blue", "green", "yellow"], "n_repeats": 1}, "participants_identical": false, "response_mode": "choice", "seed": 0, "stimuli": "built-in, not human-normed", "temperature": 1.0, "top_p": 1.0}`

Sample prompt, with the options `red`, `blue`, `green`, `yellow`:

```text
The word RED is printed in yellow ink. What color is the ink?
Response:
```

| effect | by | level_a | level_b | mean_a | mean_b | difference | ci_low | ci_high | p_value | detected | measure | proxy_for |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| interference | congruency | incongruent | congruent | 2.308 | 0.7692 | 1.539 | 0.9792 | 2.015 | 0.001499 | True | answer_surprisal | reaction time |
| errors | congruency | congruent | incongruent | 1.252 | -0.8679 | 2.12 | 1.227 | 2.944 | 0.002499 | True | answer_margin | reaction time |
