Model `gpt2` on `mps`, run on 2026-11-07 with cognarium 0.0.1.

Settings: `{"context": {"feedback": false, "keep": 0}, "max_new_tokens": 32, "n_participants": 1, "ordered": false, "participants_identical": false, "response_mode": "choice", "seed": 0, "stimuli": "built-in, not human-normed", "temperature": 1.0, "top_p": 1.0}`

Sample prompt, with the options `The baby kicked the blanket.`, `The baby slept the blanket.`:

```text
Sentence:
```

| effect | by | level_a | level_b | mean_a | mean_b | difference | ci_low | ci_high | p_value | detected | measure | proxy_for |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| preference |  | all | chance | 0.9721 | 0.5 | 0.4721 | 0.4529 | 0.4883 | 0.0004998 | True | answer_probability |  |
