Model `gpt2` on `mps`, run on 2026-11-07 with cognarium 0.1.0.

Settings: `{"context": {"feedback": true, "keep": null}, "max_new_tokens": 32, "n_participants": 1, "ordered": false, "parameters": {"n_test": 8, "n_train": 16, "noise": 0.2}, "participants_identical": false, "response_mode": "choice", "seed": 0, "stimuli": "built-in, not human-normed", "temperature": 1.0, "top_p": 1.0}`

Sample prompt, with the options `grammatical`, `ungrammatical`:

```text
Letters: P T T V P S
Judgment: grammatical
Letters: P T V P S
Judgment: grammatical
Letters: T S S X X T V V
Judgment: grammatical
Letters: T S X X V V
Judgment: grammatical
Letters: P V P X T T V V
Judgment: grammatical
Letters: P T V P X V V
Judgment: grammatical
Letters: P T T T T V V
Judgment: grammatical
Letters: P V P X T V P S
Judgment: grammatical
Letters: T S S S S X S
Judgment: grammatical
Letters: T X X V P X V V
Judgment: grammatical
Letters: P T T T T T V V
Judgment: grammatical
Letters: T S S X S
Judgment: grammatical
Letters: P V P X V P S
Judgment: grammatical
Letters: P T T V P X V V
Judgment: grammatical
Letters: P V V
Judgment: grammatical
Letters: T S S S X S
Judgment: grammatical
Letters: T X X V S
Judgment:
```

| effect | by | level_a | level_b | mean_a | mean_b | difference | ci_low | ci_high | p_value | detected | measure | proxy_for |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| grammaticality | status | grammatical | ungrammatical | 1 | 1 | 2.981e-06 | 1.473e-06 | 4.48e-06 | 0.004498 | True | option_probability |  |
