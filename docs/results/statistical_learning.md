Model `gpt2` on `mps`, run on 2026-11-07 with cognarium 0.0.1.

Settings: `{"context": {"feedback": true, "keep": null}, "max_new_tokens": 32, "n_participants": 1, "ordered": false, "parameters": {"n_words": 4, "noise": 0.3, "stream_words": 24}, "participants_identical": false, "response_mode": "choice", "seed": 0, "stimuli": "built-in, not human-normed", "temperature": 1.0, "top_p": 1.0}`

Sample prompt, with the options `familiar`, `unfamiliar`:

```text
Sounds: nu pe li
Judgment: familiar
Sounds: pe li la
Judgment: familiar
Sounds: li la va
Judgment: familiar
Sounds: la va ve
Judgment: familiar
Sounds: va ve ki
Judgment: familiar
Sounds: ve ki do
Judgment: familiar
Sounds: ki do ru
Judgment: familiar
Sounds: do ru mo
Judgment: familiar
Sounds: ru mo ku
Judgment: familiar
Sounds: mo ku ri
Judgment: familiar
Sounds: ku ri la
Judgment: familiar
Sounds: ri la va
Judgment: familiar
Sounds: la va ve
Judgment: familiar
Sounds: va ve ki
Judgment: familiar
Sounds: ve ki do
Judgment: familiar
Sounds: ki do ru
Judgment: familiar
Sounds: do ru nu
Judgment: familiar
Sounds: ru nu pe
Judgment: familiar
Sounds: nu pe li
Judgment: familiar
Sounds: pe li mo
Judgment: familiar
Sounds: li mo ku
Judgment: familiar
Sounds: mo ku ri
Judgment: familiar
Sounds: ku ri nu
Judgment: familiar
Sounds: ri nu pe
Judgment: familiar
Sounds: nu pe li
Judgment: familiar
Sounds: pe li la
Judgment: familiar
Sounds: li la va
Judgment: familiar
Sounds: la va ve
Judgment: familiar
Sounds: va ve ki
Judgment: familiar
Sounds: ve ki do
Judgment: familiar
Sounds: ki do ru
Judgment: familiar
Sounds: do ru mo
Judgment: familiar
Sounds: ru mo ku
Judgment: familiar
Sounds: mo ku ri
Judgment: familiar
Sounds: ku ri nu
Judgment: familiar
Sounds: ri nu pe
Judgment: familiar
Sounds: nu pe li
Judgment: familiar
Sounds: pe li la
Judgment: familiar
Sounds: li la va
Judgment: familiar
Sounds: la va ve
Judgment: familiar
Sounds: va ve mo
Judgment: familiar
Sounds: ve mo ku
Judgment: familiar
Sounds: mo ku ri
Judgment: familiar
Sounds: ku ri ki
Judgment: familiar
Sounds: ri ki do
Judgment: familiar
Sounds: ki do ru
Judgment: familiar
Sounds: do ru mo
Judgment: familiar
Sounds: ru mo ku
Judgment: familiar
Sounds: mo ku ri
Judgment: familiar
Sounds: ku ri ki
Judgment: familiar
Sounds: ri ki do
Judgment: familiar
Sounds: ki do ru
Judgment: familiar
Sounds: do ru nu
Judgment: familiar
Sounds: ru nu pe
Judgment: familiar
Sounds: nu pe li
Judgment: familiar
Sounds: pe li la
Judgment: familiar
Sounds: li la va
Judgment: familiar
Sounds: la va ve
Judgment: familiar
Sounds: va ve nu
Judgment: familiar
Sounds: ve nu pe
Judgment: familiar
Sounds: nu pe li
Judgment: familiar
Sounds: pe li ki
Judgment: familiar
Sounds: li ki do
Judgment: familiar
Sounds: ki do ru
Judgment: familiar
Sounds: do ru la
Judgment: familiar
Sounds: ru la va
Judgment: familiar
Sounds: la va ve
Judgment: familiar
Sounds: va ve mo
Judgment: familiar
Sounds: ve mo ku
Judgment: familiar
Sounds: mo ku ri
Judgment: familiar
Sounds: mo ku ri
Judgment:
```

| effect | by | level_a | level_b | mean_a | mean_b | difference | ci_low | ci_high | p_value | detected | measure | proxy_for |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| segmentation | test_type | word | part_word | 0.9984 | 0.9984 | 2.293e-05 | -0.0005211 | 0.0006138 | 0.97 | False | option_probability |  |
