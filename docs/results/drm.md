Model `gpt2` on `mps`, run on 2026-11-07 with cognarium 0.0.1.

Settings: `{"context": {"feedback": false, "keep": 0}, "max_new_tokens": 32, "n_participants": 1, "ordered": false, "participants_identical": false, "response_mode": "choice", "seed": 0, "stimuli": "your own rows", "temperature": 1.0, "top_p": 1.0}`

Sample prompt, with the options `old`, `new`:

```text
Study these words: song, melody, piano, rhythm, band, tune. Was this word on the list? Word: carpet
Response:
```

| effect | by | level_a | level_b | mean_a | mean_b | difference | ci_low | ci_high | p_value | detected | measure | proxy_for |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| false_memory | role | lure | unrelated | 0.2758 | 0.2712 | 0.004576 | -0.04754 | 0.05671 | 0.8881 | False | option_probability |  |
| true_memory | role | studied | unrelated | 0.2856 | 0.2712 | 0.01437 | -0.01634 | 0.04729 | 0.6222 | False | option_probability |  |
