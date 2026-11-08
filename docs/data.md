# Data

## From files

`from_csv` and `from_json` read a table and give each column its role: `text`, `label`,
`features`, `condition` and `item`. Only empty cells count as missing, so words like
"null" and "NA" stay text. JSON can be an array of rows, an object of columns, or one row
per line, and dotted paths reach into nested rows.

```python
import cognarium as cg
from pathlib import Path

Path("trials.csv").write_text(
    "prompt,answer,kind\nIs a robin a bird?,yes,typical\nIs a penguin a bird?,yes,atypical\nIs a bat a bird?,no,atypical\n",
    encoding="utf-8"
)
dataset = cg.data.from_csv("trials.csv", text="prompt", label="answer", condition=["kind"])
dataset.to_numpy()["label"]
```

A dataset converts to NumPy arrays, tensors, Parquet, a pandas table, or trials, and keeps
its vocabularies, so the numbering of labels doesn't depend on row order.

```python
trials = dataset.to_trials()
trials[0].options, trials[0].answer
```

## The registry

`cg.data.available()` lists the datasets cognarium can fetch. Every file comes from its
authors' own URL, pinned by a SHA-256 checksum and size, and is checked against that
checksum each time it loads. A download that doesn't match is removed. cognarium never
rehosts a file.

```python
cg.data.available()
```

Entries come in four tiers.

| Tier | What `load` does | Examples |
|---|---|---|
| A, openly licensed | Downloads, then prints the citation | `wordsim353`, `blimp`, `syntaxgym`, `provo`, `things_triplets`, `thingsplus`, `bigtom`, `many_labs_2` |
| B, restrictive or no stated licence | Shows the terms, and loads only with `accept_terms=True`. A dataset with no stated licence says that no rights are granted | `tomi`, `cifar10h`, `natural_stories`, `fantom`, `simlex999`, `wcs_chips`, `semantic_priming_project`, `elp_trials`, `brysbaert_concreteness`, `warriner` |
| C, gated | Says how to get a copy, then reads yours from `path=` | `psych101_test`, `elp_items` |
| D, not offered | Raises an error until a source and licence are confirmed | `men`, `usf_norms`, `kuperman_aoa`, `drm_lists`, `cogbench` |

Some entries are disabled, each with a note saying why, such as a download that needs a
login. `cg.data.available(include_disabled=True)` lists them too.

With the `Acceptability` paradigm from [custom paradigms](custom-paradigms.md):

<!-- no-run -->
```python
pairs = cg.data.load("blimp", phenomena=["causative"])
subject = cg.load("gpt2", device="auto")
results = cg.run(Acceptability(stimuli=pairs), subject)
```

<!-- no-run -->
```python
similarity = cg.data.load("simlex999", accept_terms=True)
```

Downloads are kept in your user cache folder, or in `cache_dir=`, and a valid cached
file makes no network request. Each dataset's `source` names the entry and its
checksum, so a run's manifest records exactly which file its stimuli came from.
`python -m cognarium.data.check` asks every verified link whether it still answers.
