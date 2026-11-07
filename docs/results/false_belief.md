Model `gpt2` on `mps`, run on 2026-11-07 with cognarium 0.1.0.

Settings: `{"context": {"feedback": false, "keep": 0}, "max_new_tokens": 32, "n_participants": 1, "ordered": false, "participants_identical": false, "response_mode": "choice", "seed": 0, "stimuli": "built-in, not human-normed", "temperature": 1.0, "top_p": 1.0}`

Sample prompt, with the options `coat pocket`, `backpack`:

```text
Malik puts the phone in the coat pocket. While Malik watches, someone moves the phone to the backpack. Malik leaves the room and comes back. Where will Malik look for the phone first?
Answer:
```

| effect | by | level_a | level_b | mean_a | mean_b | difference | ci_low | ci_high | p_value | detected | measure | proxy_for |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| belief_cost | belief | false | true | 2.129 | 1.066 | 1.062 | 0.002489 | 2.204 | 0.07796 | True | answer_surprisal | reaction time |
| above_chance |  | all | chance | 0.397 | 0.5 | -0.103 | -0.224 | 0.0111 | 0.1074 | False | answer_probability |  |
