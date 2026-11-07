Model `gpt2` on `mps`, run on 2026-11-07 with cognarium 0.0.1.

Settings: `{"bottleneck": 4, "chains": 50, "generations": 10, "groups": ["I", "II", "III", "IV", "V", "VI", "unbalanced"], "noise": 0.0, "seed": 0, "task": {"name": "shepard", "type": "VI"}, "temperature": 1.0, "top_p": 1.0}`

Sample prompt, with the options `A`, `B`:

```text
Object: small blue circle
Category: A
Object: large blue circle
Category: B
Object: large red circle
Category: A
Object: small blue square
Category: B
Object: small red circle
Category:
```

| generation | metric | value | reference |
|---|---|---|---|
| 0 | accuracy | 0.51 |  |
| 1 | accuracy | 0.545 |  |
| 2 | accuracy | 0.5025 |  |
| 3 | accuracy | 0.4925 |  |
| 4 | accuracy | 0.5125 |  |
| 5 | accuracy | 0.4875 |  |
| 6 | accuracy | 0.4825 |  |
| 7 | accuracy | 0.4875 |  |
| 8 | accuracy | 0.4975 |  |
| 9 | accuracy | 0.505 |  |
| 0 | fidelity | 0.51 |  |
| 1 | fidelity | 0.61 |  |
| 2 | fidelity | 0.7025 |  |
| 3 | fidelity | 0.835 |  |
| 4 | fidelity | 0.855 |  |
| 5 | fidelity | 0.89 |  |
| 6 | fidelity | 0.895 |  |
| 7 | fidelity | 0.905 |  |
| 8 | fidelity | 0.925 |  |
| 9 | fidelity | 0.9475 |  |
| 0 | entropy | 0.8229 |  |
| 1 | entropy | 0.6487 |  |
| 2 | entropy | 0.4284 |  |
| 3 | entropy | 0.3474 |  |
| 4 | entropy | 0.2585 |  |
| 5 | entropy | 0.2144 |  |
| 6 | entropy | 0.2064 |  |
| 7 | entropy | 0.1952 |  |
| 8 | entropy | 0.169 |  |
| 9 | entropy | 0.116 |  |
| 0 | chain_entropy | 1.036 |  |
| 1 | chain_entropy | 0.3825 |  |
| 2 | chain_entropy | 0.1414 |  |
| 3 | chain_entropy | 0.1414 |  |
| 4 | chain_entropy | 0.3274 |  |
| 5 | chain_entropy | 0.1414 |  |
| 6 | chain_entropy | 0.2823 |  |
| 7 | chain_entropy | 0.1414 |  |
| 8 | chain_entropy | 0.1414 |  |
| 9 | chain_entropy | 0 |  |
| 0 | kl | 0.6955 | final_third |
| 1 | kl | 0.1234 | final_third |
| 2 | kl | 0.08021 | final_third |
| 3 | kl | 0.05059 | final_third |
| 4 | kl | 0.1521 | final_third |
| 5 | kl | 0.05059 | final_third |
| 6 | kl | 0.06211 | final_third |
| 7 | kl | 0.05059 | final_third |
| 8 | kl | 0.03681 | final_third |
| 9 | kl | 0.02583 | final_third |
