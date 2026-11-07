"""Readers that turn each registry file, as its authors publish it, into a `Dataset`.

Each loader takes the path to the downloaded file and keeps the authors' column names,
except where a paradigm reads the columns under other names. Word columns keep words
such as "null" and "nan" as text, and only cells the authors left empty or marked as
missing become missing values.
"""

from __future__ import annotations

import io
import json
import re
import tarfile
import zipfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from cognarium.data.convert import Dataset, from_csv, from_json

_TEXT_ONLY = {
    "keep_default_na": False,
    "na_values": [""]
}
_BIGTOM_COLUMNS = (
    "story",
    "aware_of_event",
    "not_aware_of_event",
    "action_given_new_state",
    "action_given_initial_state",
    "belief_question",
    "desire_question",
    "action_question",
    "belief_aware",
    "desire_aware",
    "action_aware",
    "belief_not_aware",
    "desire_not_aware",
    "action_not_aware",
    "random_event",
    "aware_of_random_event",
    "not_aware_of_random_event",
    "source",
    "flag"
)
_CIFAR_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
)
_TOMI_SPLITS = ("train", "val", "test")
_ELP_HEADER = "Univ,Time,Date,Subject,DOB,Education"
_ELP_SKIPPED = frozenset({
    "9999.LDT",
    "793DATA.LDT",
    "Data999.LDT",
    "Data1000.LDT",
    "Data1010.LDT",
    "Data1016.LDT"
})
_ELP_TRIAL = re.compile(r"^\d+,\d+,\d+,\d+,\d+,")


def wordsim353(path: Path) -> Dataset:
    """Reads WordSim-353, from the combined set of its 353 word pairs.

    Args:
        path: The downloaded `wordsim353.zip`.

    Returns:
        One row per pair, with `Word 1`, `Word 2` and `Human (mean)`, the mean rating of
        how related the words are, from 0 for unrelated to 10 for identical.
    """
    with zipfile.ZipFile(path) as archive:
        table = pd.read_csv(io.BytesIO(archive.read("combined.csv")), **_TEXT_ONLY)
    return Dataset(table)


def blimp(path: Path, *, phenomena: Sequence[str] | None = None) -> Dataset:
    """Reads BLiMP's 67 sets of 1,000 minimal pairs, in the columns `MinimalPairs` reads.

    Args:
        path: The downloaded `BLiMP.zip`.
        phenomena: The sets to keep, named by their `UID`, such as `causative`. Keeps
            every set by default.

    Returns:
        One row per pair, with `good` and `bad`, the acceptable and unacceptable
        sentence, `phenomenon`, the set's `UID`, `field`, `linguistics_term`, `pair_id`,
        and the authors' flags for which scoring methods suit the pair.

    Raises:
        ValueError: If a phenomenon isn't one of the sets.
    """
    rows = []
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if re.fullmatch(r"data/[^/]+\.jsonl", name):
                for line in archive.read(name).decode("utf-8").splitlines():
                    if line.strip():
                        rows.append(json.loads(line))
    table = pd.DataFrame(rows).rename(
        columns={
            "sentence_good": "good",
            "sentence_bad": "bad",
            "UID": "phenomenon",
            "pairID": "pair_id"
        }
    )
    table["pair_id"] = table["pair_id"].astype(int)
    if phenomena is not None:
        unknown = sorted(set(phenomena) - set(table["phenomenon"]))
        if unknown:
            raise ValueError(f"BLiMP has no sets {unknown}. Its sets are {sorted(set(table['phenomenon']))}.")
        table = table[table["phenomenon"].isin(list(phenomena))]
    return Dataset(table, condition=["phenomenon"])


def syntaxgym(path: Path) -> Dataset:
    """Reads the 34 SyntaxGym test suites of Hu et al. (2020).

    Args:
        path: The downloaded archive of the authors' repository.

    Returns:
        One row per suite, item and condition, with `suite`, `item`, `condition`,
        `sentence`, the regions joined by spaces, `regions`, the region texts in order,
        `region_names`, and `predictions`, the suite's success criteria as the authors'
        formulas.
    """
    rows = []
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if re.search(r"/test_suites/json/[^/]+\.json$", name):
                suite = json.loads(archive.read(name))
                names = [suite["region_meta"][key] for key in sorted(suite["region_meta"], key=int)]
                predictions = [prediction["formula"] for prediction in suite["predictions"]]
                for item in suite["items"]:
                    for condition in item["conditions"]:
                        regions = [region["content"] for region in sorted(condition["regions"], key=lambda r: r["region_number"])]
                        rows.append({
                            "suite": suite["meta"]["name"],
                            "item": item["item_number"],
                            "condition": condition["condition_name"],
                            "sentence": " ".join(region for region in regions if region.strip()),
                            "regions": regions,
                            "region_names": names,
                            "predictions": predictions
                        })
    return Dataset(pd.DataFrame(rows), text="sentence", condition=["suite", "condition"])


def provo(path: Path) -> Dataset:
    """Reads the Provo Corpus eye-tracking data, merged by its authors with the cloze norms.

    Args:
        path: The downloaded `Provo_Corpus-Eyetracking_Data.csv`.

    Returns:
        One row per participant and word, with the authors' 63 columns: `Participant_ID`,
        `Text_ID`, `Word_Number` and `Word` identify the reading, the `IA_` columns are
        the eye tracker's interest-area measures, such as `IA_FIRST_FIXATION_DURATION`,
        and `OrthographicMatch` and its neighbours are the word's cloze predictability.
    """
    return from_csv(path, na_values=["", "NA"], low_memory=False)


def things_triplets(path: Path) -> Dataset:
    """Reads the 4.7 million THINGS odd-one-out judgments.

    Args:
        path: The downloaded `triplets_large_final_correctednc_correctedorder.csv`, which
            is tab-separated.

    Returns:
        One row per judgment, with `image1`, `image2` and `image3`, the THINGS concepts
        shown, numbered from 1 to 1854, `choice`, which of the three was the odd one out,
        `RT` in milliseconds, `noise_ceiling`, which marks repeated triplets, and the
        anonymised `subject_id`.
    """
    return from_csv(path, sep="\t", na_values=["", "NaN"])


def thingsplus(path: Path) -> Dataset:
    """Reads the THINGSplus property and size norms for the 1,854 THINGS concepts.

    Args:
        path: The downloaded `_property-ratings.tsv`.

    Returns:
        One row per concept, with `Word`, `uniqueID`, and the mean and standard deviation
        of each rated property on a 1 to 7 scale, such as `property_manmade_mean`, and of
        real-world size, as `size_mean`.
    """
    table = pd.read_csv(path, sep="\t", **_TEXT_ONLY)
    table = table.drop(columns=[column for column in table.columns if str(column).startswith("Unnamed")])
    return Dataset(table, item="uniqueID")


def bigtom(path: Path) -> Dataset:
    """Reads the 200 BigToM stories, each with its true- and false-belief variants.

    Args:
        path: The downloaded `bigtom.csv`, which is separated by semicolons and has no
            header.

    Returns:
        One row per story, in the order of the authors' generator: `story`, the events
        the agent is and isn't aware of, the questions, and the answers for each
        condition, such as `belief_aware` and `belief_not_aware`. The last two columns,
        `source` and `flag`, are written by the generator and not documented.
    """
    table = pd.read_csv(
        path,
        sep=";",
        header=None,
        names=list(_BIGTOM_COLUMNS),
        **_TEXT_ONLY
    )
    return Dataset(table, text="story")


def navarro2018(path: Path) -> Dataset:
    """Reads the prior elicitation of Navarro et al. (2018), from the MTurk sample.

    Args:
        path: The downloaded `data_MTurk_direct.csv`.

    Returns:
        One row per participant, with `ppt`, `Gender`, where 1 is male and 2 female,
        `Age`, `Country`, and for each candidate named in the task the probability in
        percent that the participant gave them.
    """
    return from_csv(path, item="ppt")


def many_labs_2(path: Path) -> Dataset:
    """Reads one slate of the Many Labs 2 cleaned data.

    Args:
        path: The downloaded CSV of slate 1 or slate 2.

    Returns:
        One row per participant, with the site in `.id`, the session's metadata, and the
        columns of each study in the slate, prefixed by the study's code as the authors'
        codebook lists them.
    """
    table = pd.read_csv(
        path,
        keep_default_na=False,
        na_values=["", "NA"],
        low_memory=False
    )
    return Dataset(table.drop(columns=[table.columns[0]]))


def tomi(path: Path, *, split: str = "test") -> Dataset:
    """Reads one split of ToMi, with each question's story and the types the authors give.

    Args:
        path: The downloaded `tomi_balanced_story_types.zip`.
        split: `train`, `val` or `test`.

    Returns:
        One row per question, with `story`, its sentences up to the question, `question`,
        `answer`, `question_type`, such as `first_order_0_tom` or `reality`, and
        `story_type`, such as `false_belief`.

    Raises:
        ValueError: If the split is unknown, or a split's questions and types don't pair
            up.
    """
    if split not in _TOMI_SPLITS:
        raise ValueError(f"ToMi's splits are {_TOMI_SPLITS}, got '{split}'.")
    with zipfile.ZipFile(path) as archive:
        lines = archive.read(f"tomi_balanced_story_types/fb_all_{split}.txt").decode("utf-8").splitlines()
        traces = archive.read(f"tomi_balanced_story_types/fb_all_{split}.trace").decode("utf-8").splitlines()
    rows, story = [], []
    for line in lines:
        number, _, text = line.partition(" ")
        if number == "1":
            story = []
        if "\t" in text:
            question, answer, _ = text.split("\t")
            rows.append({"story": " ".join(story), "question": question, "answer": answer})
        else:
            story.append(text)
    if len(rows) != len(traces):
        raise ValueError(f"ToMi's {split} split has {len(rows)} questions but {len(traces)} types.")
    for row, trace in zip(rows, traces, strict=True):
        *_, row["question_type"], row["story_type"] = trace.split(",")
    return Dataset(pd.DataFrame(rows), condition=["question_type", "story_type"])


def cifar10h(path: Path) -> Dataset:
    """Reads the CIFAR-10H human label distributions for the CIFAR-10 test images.

    Args:
        path: The downloaded `cifar10h-probs.npy`.

    Returns:
        One row per test image, in the original CIFAR-10 order, with `image`, its
        position, and one column per class, from `airplane` to `truck`, holding the share
        of about 50 people who chose it. The class columns are the dataset's features.
    """
    probabilities = np.load(path, allow_pickle=False)
    table = pd.DataFrame(probabilities, columns=list(_CIFAR_CLASSES))
    table.insert(0, "image", np.arange(len(table)))
    return Dataset(table, features=list(_CIFAR_CLASSES), item="image")


def natural_stories(path: Path) -> Dataset:
    """Reads the Natural Stories self-paced reading times, as the authors processed them.

    Args:
        path: The downloaded `processed_RTs.tsv`.

    Returns:
        One row per reader and word, with `WorkerId`, `item`, the story from 1 to 10,
        `zone`, the word's position in it, `word`, `RT` in milliseconds, `correct`, the
        reader's comprehension score out of 6, and the word's mean and spread across
        readers.
    """
    return from_csv(
        path,
        sep="\t",
        text="word",
        low_memory=False
    )


def fantom(path: Path) -> Dataset:
    """Reads the FANToM conversations with their questions.

    Args:
        path: The downloaded `fantom.tar.gz`.

    Returns:
        One row per conversation, with `set_id`, `full_context` and `short_context`, the
        conversation, `joining_speaker`, `missed_info`, what that speaker missed, and the
        question sets, such as `beliefQAs` and `factQA`, as the authors' records.

    Raises:
        ValueError: If the archive doesn't hold `fantom_v1.json`.
    """
    with tarfile.open(path, "r:gz") as archive:
        member = next((entry for entry in archive.getmembers() if entry.name.endswith("fantom_v1.json")), None)
        if member is None or not member.isfile():
            raise ValueError(f"{Path(path).name} doesn't hold fantom_v1.json.")
        records = json.load(archive.extractfile(member))
    return Dataset(pd.DataFrame(records), item="set_id")


def simlex999(path: Path) -> Dataset:
    """Reads SimLex-999's ratings of how similar 999 word pairs are.

    Args:
        path: The downloaded `SimLex-999.zip`.

    Returns:
        One row per pair, with `word1`, `word2`, `POS`, `SimLex999`, the mean similarity
        on a 0 to 10 scale, `SD(SimLex)`, the words' concreteness, and their association
        strength in `Assoc(USF)`.
    """
    with zipfile.ZipFile(path) as archive:
        table = pd.read_csv(io.BytesIO(archive.read("SimLex-999/SimLex-999.txt")), sep="\t", **_TEXT_ONLY)
    return Dataset(table)


def semantic_priming_project(path: Path) -> Dataset:
    """Reads every lexical decision trial of the Semantic Priming Project.

    Reading the workbook's 847,723 rows takes about 20 seconds.

    Args:
        path: The downloaded `all ldt subs_all trials3.xlsx`.

    Returns:
        One row per trial, with `Subject`, `Session`, `isi`, the gap in milliseconds
        between prime and target, which is 50 or 1050, `prime`, `target`, `lexicality`,
        where 1 is a word, `type`, the prime's kind such as `first` associate, `rel`,
        `rel` for related, `un` for unrelated and `nw` for a nonword target,
        `target.ACC`, `target.RT` in milliseconds, and the authors' trimmed and z-scored
        times. The authors' `#NULL!` cells are missing values.
    """
    table = pd.read_excel(
        path,
        engine="calamine",
        keep_default_na=False,
        na_values=["", "#NULL!"]
    )
    return Dataset(table)


def elp_trials(path: Path) -> Dataset:
    """Reads every lexical decision trial of the English Lexicon Project.

    Each participant's file holds a header, then a block of trials for each session,
    then questionnaires, which are left out. The files the project's notes say to skip
    are skipped.

    Args:
        path: The downloaded `ldt_raw.zip`.

    Returns:
        One row per trial, with `file`, the participant's file, `university`, coded from
        1 to 6, `subject`, `session`, counted from 1 within each file, `trial`, the
        position in the session, `item_serial`, `lexicality`, where 1 is a word and 0 a
        nonword, `accuracy`, 1 for correct and 0 for wrong, with the few other codes
        kept as the files hold them, `rt` in milliseconds, and `item`.
    """
    rows = []
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if Path(name).name in _ELP_SKIPPED or not name.upper().endswith(".LDT"):
                continue
            rows.extend(_elp_file(name, archive.read(name).decode("latin-1").splitlines()))
    columns = [
        "file",
        "university",
        "subject",
        "session",
        "trial",
        "item_serial",
        "lexicality",
        "accuracy",
        "rt",
        "item"
    ]
    return Dataset(pd.DataFrame(rows, columns=columns), text="item")


def brysbaert_concreteness(path: Path) -> Dataset:
    """Reads the concreteness ratings of Brysbaert et al. (2014).

    Args:
        path: The downloaded supplementary workbook.

    Returns:
        One row per word or two-word expression, with `Word`, `Bigram`, which is 1 for
        the two-word ones, `Conc.M`, the mean concreteness from 1 for abstract to 5 for
        concrete, `Conc.SD`, `Unknown`, `Total`, `Percent_known` and `SUBTLEX`, the
        word's frequency count.
    """
    table = pd.read_excel(
        path,
        sheet_name=0,
        engine="calamine",
        **_TEXT_ONLY
    )
    table["Word"] = table["Word"].astype(str)
    return Dataset(table, text="Word")


def warriner(path: Path) -> Dataset:
    """Reads the valence, arousal and dominance norms of Warriner et al. (2013).

    Args:
        path: The downloaded supplementary archive.

    Returns:
        One row per word, with `Word`, and for valence `V`, arousal `A` and dominance
        `D` on a 1 to 9 scale, the mean, standard deviation and number of ratings, such
        as `V.Mean.Sum`, `V.SD.Sum` and `V.Rat.Sum`. Suffixes other than `.Sum` split
        the raters by gender (`.M`, `.F`), age (`.Y`, `.O`) and education (`.L`, `.H`).
    """
    with zipfile.ZipFile(path) as archive:
        table = pd.read_csv(io.BytesIO(archive.read("BRM-emot-submit.csv")), low_memory=False, **_TEXT_ONLY)
    table = table.drop(columns=[table.columns[0]])
    # one rating cell holds "NA", which is only a missing value outside the word column
    ratings = [column for column in table.columns if column != "Word"]
    table[ratings] = table[ratings].replace("NA", np.nan).apply(pd.to_numeric)
    return Dataset(table, text="Word")


def psych101(path: Path) -> Dataset:
    """Reads a Psych-101 file, such as the gated test split, from your own copy.

    Args:
        path: A JSON Lines file with one participant's transcript per line.

    Returns:
        One row per participant and experiment, with `text`, the experiment told in
        words with the participant's choices between `<<` and `>>`, `experiment` and
        `participant`.
    """
    return from_json(path, text="text")


def elp_items(path: Path) -> Dataset:
    """Reads an item-level export of the English Lexicon Project.

    Args:
        path: The CSV file the project's website exports, which includes `Word`.

    Returns:
        One row per item, with the columns chosen when exporting. The project's `#`
        cells are missing values.
    """
    return from_csv(path, text="Word", na_values=["", "#"])


def _elp_file(name: str, lines: list[str]) -> list[tuple]:
    rows, session, reading, header = [], 0, False, None
    for position, line in enumerate(lines):
        if line.strip() == _ELP_HEADER:
            header = lines[position + 1].split(",") if position + 1 < len(lines) else None
            session += 1
            reading = False
            continue
        if header is not None and line.split(",") == header:
            reading = True
            continue
        if not line.strip():
            reading = False
            continue
        if reading and _ELP_TRIAL.match(line):
            trial, serial, lexicality, accuracy, rt, item = line.split(",", 5)
            rows.append((
                name,
                int(header[0]),
                int(header[3]),
                session,
                int(trial),
                int(serial),
                int(lexicality),
                int(accuracy),
                int(rt),
                item
            ))
    return rows
