Model `gpt2` on `mps`, run on 2026-11-07 with cognarium 0.0.1.

Settings: `{"context": {"feedback": false, "keep": 0}, "max_new_tokens": 32, "n_participants": 1, "ordered": false, "parameters": {"colors": ["red", "blue"], "n_repeats": 1, "shapes": ["square", "circle", "triangle", "star"]}, "participants_identical": false, "response_mode": "choice", "seed": 0, "stimuli": "built-in, not human-normed", "temperature": 1.0, "top_p": 1.0}`

Sample prompt, with the options `left`, `right`:

```text
Press left for blue and right for red. A red star appears on the right. Which key?
Response:
```

| effect | by | level_a | level_b | mean_a | mean_b | difference | ci_low | ci_high | p_value | detected | measure | proxy_for |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| simon_effect | correspondence | non_corresponding | corresponding | 1.056 | 1.02 | 0.03611 | -0.169 | 0.2497 | 0.7281 | False | answer_surprisal | reaction time |
| errors | correspondence | corresponding | non_corresponding | 0.01428 | -0.0357 | 0.04998 | -0.2316 | 0.3438 | 0.7271 | False | answer_margin | reaction time |
