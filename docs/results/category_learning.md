Model `gpt2` on `mps`, run on 2026-11-07 with cognarium 0.1.0.

Settings: `{"context": {"feedback": true, "keep": null}, "max_new_tokens": 32, "n_participants": 1, "ordered": false, "parameters": {"blocks": 6}, "participants_identical": false, "response_mode": "choice", "seed": 0, "stimuli": "built-in, not human-normed", "task": {"name": "shepard", "type": "I"}, "temperature": 1.0, "top_p": 1.0}`

Sample prompt, with the options `A`, `B`:

```text
Object: large red circle
Category:
```

| effect | by | level_a | level_b | mean_a | mean_b | difference | ci_low | ci_high | p_value | detected | measure | proxy_for |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| learning | block | 5 | 0 | 0.6194 | 0.5116 | 0.1078 | -0.07273 | 0.2835 | 0.2859 | False | answer_probability |  |
