"""The datasets cognarium can fetch, what each one's terms allow, and loading them.

Entries come in four tiers. Tier A datasets are openly licensed: they download and
print their citation. Tier B datasets have restrictive terms or none stated: they show
those terms and load only with `accept_terms=True`. Tier C datasets are gated behind a
login or a licence agreement: their entry says how to get a copy, and loads it from
`path=`. Tier D datasets aren't offered until their source and licence are confirmed.

Every file is fetched from its authors' own URL, which is pinned with a checksum and
size, and cognarium never rehosts one. An entry stays disabled until its URL and
checksum have been verified.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from cognarium.data.convert import Dataset
from cognarium.data.download import fetch

NO_LICENCE = "No licence stated"
_UNCONFIRMED = "Unconfirmed"
_TIERS = ("A", "B", "C", "D")
_MEDIA = "Media datasets arrive with the speech and vision subjects, so this one is disabled for now."
_NOT_OFFERED = "It isn't offered until a source and licence for it are confirmed."


@dataclass(frozen=True)
class Entry:
    """One dataset in the registry.

    Attributes:
        name: The name `load` takes.
        tier: `A` for openly licensed, `B` for restrictive or no stated terms, `C` for
            gated behind a login or agreement, and `D` for not offered.
        title: The dataset's usual name.
        licence: The licence, as its authors state it, or `NO_LICENCE`.
        citation: The paper to cite when using it.
        url: The authors' URL for the file, or `hf://datasets/<owner>/<repo>/<file>@<commit>`
            for a file on the Hugging Face Hub at a pinned commit.
        sha256: The file's SHA-256 checksum, in hex.
        size: The file's size in bytes.
        loader: Turns the file into a `Dataset`. It takes the file's path and the options
            `load` passes on.
        verified: The date the URL and checksum were checked. An entry without one stays
            disabled.
        instructions: How to get a copy of a tier C dataset.
        note: Why the entry is disabled, or what its terms say beyond the licence's name.
        file_name: The name the download is saved under, for a URL that doesn't end in
            one.
    """

    name: str
    tier: str
    title: str
    licence: str
    citation: str
    url: str | None = None
    sha256: str | None = None
    size: int | None = None
    loader: Callable[..., Dataset] | None = None
    verified: str | None = None
    instructions: str | None = None
    note: str | None = None
    file_name: str | None = None

    @property
    def status(self) -> str:
        """`ready`, `local copy` for a usable tier C entry, `disabled` or `not offered`."""
        if self.tier == "D":
            return "not offered"
        if self.loader is None or (self.tier != "C" and self.verified is None):
            return "disabled"
        return "local copy" if self.tier == "C" else "ready"


def available(*, include_disabled: bool = False) -> pd.DataFrame:
    """Lists the datasets `load` can give.

    Args:
        include_disabled: Whether to list entries that are disabled, such as ones whose
            download hasn't been verified. Tier D entries are never listed.

    Returns:
        One row per dataset, with `name`, `tier`, `title`, `licence`, `size` in bytes and
        `status`, which is `ready`, `local copy` or `disabled`.
    """
    rows = [
        {
            "name": entry.name,
            "tier": entry.tier,
            "title": entry.title,
            "licence": entry.licence,
            "size": entry.size,
            "status": entry.status
        }
        for entry in REGISTRY.values()
        if entry.status != "not offered" and (include_disabled or entry.status != "disabled")
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "name",
            "tier",
            "title",
            "licence",
            "size",
            "status"
        ]
    )


def load(
    name: str,
    *,
    accept_terms: bool = False,
    path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    **options: Any
) -> Dataset:
    """Loads a dataset from the registry, downloading it the first time.

    The download is checked against the entry's checksum, kept in the cache, and checked
    again on later calls, which make no network request. Each call prints the citation.
    The dataset's `source` names the entry and its checksum, so a run's manifest records
    exactly which file its stimuli came from.

    Args:
        name: The dataset, as `available()` lists it.
        accept_terms: Confirms that you have read a tier B dataset's terms and will keep
            to them. A tier B dataset doesn't load without it.
        path: A copy of the file you already have, which is read instead of downloading.
            A tier C dataset needs it.
        cache_dir: Where downloads are kept. See `fetch`.
        **options: Passed to the entry's loader.

    Returns:
        The dataset.

    Raises:
        ValueError: If the name is unknown, the dataset isn't offered or is disabled,
            its terms weren't accepted, a tier C dataset has no `path`, or the download
            doesn't match its checksum. The message says why, and what to do.
        FileNotFoundError: If `path` doesn't exist.
        ConnectionError: If the download fails and there's no copy in the cache.
    """
    if name not in REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. cognarium.data.available() lists the ones on offer.")
    entry = REGISTRY[name]
    if entry.status == "not offered":
        raise ValueError(f"{entry.title} isn't offered. {entry.note}")
    if entry.status == "disabled":
        raise ValueError(f"{entry.title} is disabled. {entry.note}")
    if entry.tier == "B" and not accept_terms:
        raise ValueError(_terms(entry))
    if path is not None:
        file = Path(path)
        if not file.exists():
            raise FileNotFoundError(f"There's no file at {file}.")
        source = f"{entry.name} from {file.name}"
    elif entry.tier == "C":
        raise ValueError(f"{entry.title} can't be downloaded for you. {entry.instructions}")
    else:
        file = fetch(entry, cache_dir=cache_dir)
        source = f"{entry.name} (sha256 {entry.sha256})"
    print(f"{entry.title}, {entry.licence}. Please cite: {entry.citation}")
    dataset = entry.loader(file, **options)
    dataset.source = source
    return dataset


def _terms(entry: Entry) -> str:
    if entry.licence == NO_LICENCE:
        stated = (
            f"{entry.title} states no licence, so no rights are granted: its authors share it, but haven't "
            "said what you may do with it."
        )
    else:
        stated = f"{entry.title} is shared under {entry.licence}."
    extra = f" {entry.note}" if entry.note else ""
    return f"{stated}{extra} Read its terms, then pass accept_terms=True to load it."


_ENTRIES = (
    Entry(
        name="wordsim353",
        tier="A",
        title="WordSim-353",
        licence=_UNCONFIRMED,
        citation=(
            "Finkelstein, L., Gabrilovich, E., Matias, Y., Rivlin, E., Solan, Z., Wolfman, G., & Ruppin, E. "
            "(2002). Placing search in context: The concept revisited. ACM Transactions on Information "
            "Systems, 20(1), 116-131."
        ),
        note="Its download hasn't been verified yet."
    ),
    Entry(
        name="blimp",
        tier="A",
        title="BLiMP",
        licence=_UNCONFIRMED,
        citation=(
            "Warstadt, A., Parrish, A., Liu, H., Mohananey, A., Peng, W., Wang, S.-F., & Bowman, S. R. (2020). "
            "BLiMP: The Benchmark of Linguistic Minimal Pairs for English. Transactions of the Association "
            "for Computational Linguistics, 8, 377-392."
        ),
        note="Its download hasn't been verified yet."
    ),
    Entry(
        name="syntaxgym",
        tier="A",
        title="SyntaxGym test suites",
        licence=_UNCONFIRMED,
        citation=(
            "Gauthier, J., Hu, J., Wilcox, E., Qian, P., & Levy, R. (2020). SyntaxGym: An online platform for "
            "targeted evaluation of language models. Proceedings of ACL: System Demonstrations, 70-76."
        ),
        note="Its download hasn't been verified yet."
    ),
    Entry(
        name="provo",
        tier="A",
        title="Provo Corpus",
        licence=_UNCONFIRMED,
        citation=(
            "Luke, S. G., & Christianson, K. (2018). The Provo Corpus: A large eye-tracking corpus with "
            "predictability norms. Behavior Research Methods, 50(2), 826-833."
        ),
        note="Its download hasn't been verified yet."
    ),
    Entry(
        name="zuco",
        tier="A",
        title="ZuCo",
        licence=_UNCONFIRMED,
        citation=(
            "Hollenstein, N., Rotsztejn, J., Troendle, M., Pedroni, A., Zhang, C., & Langer, N. (2018). ZuCo, "
            "a simultaneous EEG and eye-tracking resource for natural sentence reading. Scientific Data, 5, "
            "180291."
        ),
        note="Its EEG and eye-tracking recordings arrive with the media datasets, so it is disabled for now."
    ),
    Entry(
        name="things_triplets",
        tier="A",
        title="THINGS odd-one-out triplets",
        licence=_UNCONFIRMED,
        citation=(
            "Hebart, M. N., Contier, O., Teichmann, L., Rockter, A. H., Zheng, C. Y., Kidder, A., Corriveau, "
            "A., Vaziri-Pashkam, M., & Baker, C. I. (2023). THINGS-data, a multimodal collection of "
            "large-scale datasets for investigating object representations in human brain and behavior. "
            "eLife, 12, e82580."
        ),
        note="Its download hasn't been verified yet."
    ),
    Entry(
        name="thingsplus",
        tier="A",
        title="THINGSplus norms",
        licence=_UNCONFIRMED,
        citation=(
            "Stoinski, L. M., Perkuhn, J., & Hebart, M. N. (2024). THINGSplus: New norms and metadata for the "
            "THINGS database of 1854 object concepts and 26,107 natural object images. Behavior Research "
            "Methods, 56, 1583-1603."
        ),
        note="Its download hasn't been verified yet."
    ),
    Entry(
        name="geirhos_cue_conflict",
        tier="A",
        title="Geirhos cue-conflict images",
        licence=_UNCONFIRMED,
        citation=(
            "Geirhos, R., Rubisch, P., Michaelis, C., Bethge, M., Wichmann, F. A., & Brendel, W. (2019). "
            "ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and "
            "robustness. International Conference on Learning Representations."
        ),
        note=_MEDIA
    ),
    Entry(
        name="speech_commands",
        tier="A",
        title="Speech Commands",
        licence=_UNCONFIRMED,
        citation=(
            "Warden, P. (2018). Speech Commands: A dataset for limited-vocabulary speech recognition. "
            "arXiv:1804.03209."
        ),
        note=_MEDIA
    ),
    Entry(
        name="librispeech",
        tier="A",
        title="LibriSpeech",
        licence=_UNCONFIRMED,
        citation=(
            "Panayotov, V., Chen, G., Povey, D., & Khudanpur, S. (2015). Librispeech: An ASR corpus based on "
            "public domain audio books. IEEE International Conference on Acoustics, Speech and Signal "
            "Processing, 5206-5210."
        ),
        note=_MEDIA
    ),
    Entry(
        name="psych101",
        tier="A",
        title="Psych-101",
        licence=_UNCONFIRMED,
        citation=(
            "Binz, M., Akata, E., Bethge, M., Brändle, F., Callaway, F., Coda-Forno, J., ... Schulz, E. "
            "(2025). A foundation model to predict and capture human cognition. Nature, 644, 1002-1009."
        ),
        note="Its download hasn't been verified yet."
    ),
    Entry(
        name="bigtom",
        tier="A",
        title="BigToM",
        licence=_UNCONFIRMED,
        citation=(
            "Gandhi, K., Fränken, J.-P., Gerstenberg, T., & Goodman, N. D. (2023). Understanding social "
            "reasoning in language models with language models. Advances in Neural Information Processing "
            "Systems, Datasets and Benchmarks Track."
        ),
        note="Its download hasn't been verified yet."
    ),
    Entry(
        name="nosofsky1994",
        tier="A",
        title="Nosofsky et al. (1994) category learning",
        licence=_UNCONFIRMED,
        citation=(
            "Nosofsky, R. M., Gluck, M. A., Palmeri, T. J., McKinley, S. C., & Glauthier, P. (1994). Comparing "
            "models of rule-based classification learning: A replication and extension of Shepard, Hovland, "
            "and Jenkins (1961). Memory & Cognition, 22(3), 352-369."
        ),
        note="Its download hasn't been verified yet."
    ),
    Entry(
        name="navarro2018",
        tier="A",
        title="Navarro et al. (2018) iterated learning",
        licence=_UNCONFIRMED,
        citation=(
            "Navarro, D. J., Perfors, A., Kary, A., Brown, S. D., & Donkin, C. (2018). When extremists win: "
            "Cultural transmission via iterated learning when populations are heterogeneous. Cognitive "
            "Science, 42(7), 2108-2149."
        ),
        note="Its download hasn't been verified yet."
    ),
    Entry(
        name="omniglot",
        tier="A",
        title="Omniglot",
        licence=_UNCONFIRMED,
        citation=(
            "Lake, B. M., Salakhutdinov, R., & Tenenbaum, J. B. (2015). Human-level concept learning through "
            "probabilistic program induction. Science, 350(6266), 1332-1338."
        ),
        note=_MEDIA
    ),
    Entry(
        name="narratives",
        tier="A",
        title="Narratives stimuli",
        licence=_UNCONFIRMED,
        citation=(
            "Nastase, S. A., Liu, Y.-F., Hillman, H., Zadbood, A., Hasenfratz, L., Keshavarzian, N., ... "
            "Hasson, U. (2021). The \"Narratives\" fMRI dataset for evaluating models of naturalistic language "
            "comprehension. Scientific Data, 8, 250."
        ),
        note="Its download hasn't been verified yet."
    ),
    Entry(
        name="many_labs_2",
        tier="A",
        title="Many Labs 2",
        licence=_UNCONFIRMED,
        citation=(
            "Klein, R. A., Vianello, M., Hasselman, F., Adams, B. G., Adams, R. B., Jr., Alper, S., ... Nosek, "
            "B. A. (2018). Many Labs 2: Investigating variation in replicability across samples and settings. "
            "Advances in Methods and Practices in Psychological Science, 1(4), 443-490."
        ),
        note="Its download hasn't been verified yet."
    ),
    Entry(
        name="meco",
        tier="A",
        title="MECO",
        licence=_UNCONFIRMED,
        citation=(
            "Siegelman, N., Schroeder, S., Acartürk, C., Ahn, H.-D., Alexeeva, S., Amenta, S., ... Kuperman, V. "
            "(2022). Expanding horizons of cross-linguistic research on reading: The Multilingual Eye-movement "
            "Corpus (MECO). Behavior Research Methods, 54, 2843-2863."
        ),
        note="Its download hasn't been verified yet."
    ),
    Entry(
        name="swow",
        tier="B",
        title="Small World of Words (English)",
        licence=_UNCONFIRMED,
        citation=(
            "De Deyne, S., Navarro, D. J., Perfors, A., Brysbaert, M., & Storms, G. (2019). The \"Small World "
            "of Words\" English word association norms for over 12,000 cue words. Behavior Research Methods, "
            "51, 987-1006."
        ),
        note="Its download hasn't been verified yet."
    ),
    Entry(
        name="tomi",
        tier="B",
        title="ToMi",
        licence=_UNCONFIRMED,
        citation=(
            "Le, M., Boureau, Y.-L., & Nickel, M. (2019). Revisiting the evaluation of theory of mind through "
            "question answering. Proceedings of EMNLP-IJCNLP, 5872-5877."
        ),
        note="Its download hasn't been verified yet."
    ),
    Entry(
        name="cifar10h",
        tier="B",
        title="CIFAR-10H",
        licence=_UNCONFIRMED,
        citation=(
            "Peterson, J. C., Battleday, R. M., Griffiths, T. L., & Russakovsky, O. (2019). Human uncertainty "
            "makes classification more robust. Proceedings of the IEEE/CVF International Conference on "
            "Computer Vision, 9617-9626."
        ),
        note="Its download hasn't been verified yet."
    ),
    Entry(
        name="natural_stories",
        tier="B",
        title="Natural Stories",
        licence=_UNCONFIRMED,
        citation=(
            "Futrell, R., Gibson, E., Tily, H. J., Blank, I., Vishnevetsky, A., Piantadosi, S. T., & Fedorenko, "
            "E. (2021). The Natural Stories corpus: A reading-time corpus of English texts containing rare "
            "syntactic constructions. Language Resources and Evaluation, 55, 63-77."
        ),
        note="Its download hasn't been verified yet."
    ),
    Entry(
        name="fantom",
        tier="B",
        title="FANToM",
        licence=_UNCONFIRMED,
        citation=(
            "Kim, H., Sclar, M., Zhou, X., Le Bras, R., Kim, G., Choi, Y., & Sap, M. (2023). FANToM: A "
            "benchmark for stress-testing machine theory of mind in interactions. Proceedings of EMNLP, "
            "14397-14413."
        ),
        note="Its download hasn't been verified yet."
    ),
    Entry(
        name="simlex999",
        tier="B",
        title="SimLex-999",
        licence=_UNCONFIRMED,
        citation=(
            "Hill, F., Reichart, R., & Korhonen, A. (2015). SimLex-999: Evaluating semantic models with "
            "(genuine) similarity estimation. Computational Linguistics, 41(4), 665-695."
        ),
        note="Its download hasn't been verified yet."
    ),
    Entry(
        name="semantic_priming_project",
        tier="B",
        title="Semantic Priming Project",
        licence=_UNCONFIRMED,
        citation=(
            "Hutchison, K. A., Balota, D. A., Neely, J. H., Cortese, M. J., Cohen-Shikora, E. R., Tse, C.-S., "
            "Yap, M. J., Bengson, J. J., Niemeyer, D., & Buchanan, E. (2013). The semantic priming project. "
            "Behavior Research Methods, 45, 1099-1114."
        ),
        note="Its download hasn't been verified yet."
    ),
    Entry(
        name="elp_trials",
        tier="B",
        title="English Lexicon Project, trial-level",
        licence=_UNCONFIRMED,
        citation=(
            "Balota, D. A., Yap, M. J., Cortese, M. J., Hutchison, K. A., Kessler, B., Loftis, B., Neely, J. "
            "H., Nelson, D. L., Simpson, G. B., & Treiman, R. (2007). The English Lexicon Project. Behavior "
            "Research Methods, 39, 445-459."
        ),
        note="Its download hasn't been verified yet."
    ),
    Entry(
        name="brysbaert_concreteness",
        tier="B",
        title="Brysbaert et al. (2014) concreteness ratings",
        licence=_UNCONFIRMED,
        citation=(
            "Brysbaert, M., Warriner, A. B., & Kuperman, V. (2014). Concreteness ratings for 40 thousand "
            "generally known English word lemmas. Behavior Research Methods, 46, 904-911."
        ),
        note="Its download hasn't been verified yet."
    ),
    Entry(
        name="warriner",
        tier="B",
        title="Warriner et al. (2013) valence, arousal and dominance norms",
        licence=_UNCONFIRMED,
        citation=(
            "Warriner, A. B., Kuperman, V., & Brysbaert, M. (2013). Norms of valence, arousal, and dominance "
            "for 13,915 English lemmas. Behavior Research Methods, 45, 1191-1207."
        ),
        note="Its download hasn't been verified yet."
    ),
    Entry(
        name="timit",
        tier="C",
        title="TIMIT",
        licence="LDC User Agreement for Non-Members",
        citation=(
            "Garofolo, J. S., Lamel, L. F., Fisher, W. M., Fiscus, J. G., Pallett, D. S., Dahlgren, N. L., & "
            "Zue, V. (1993). TIMIT Acoustic-Phonetic Continuous Speech Corpus (LDC93S1). Linguistic Data "
            "Consortium."
        ),
        instructions="It is licensed through the Linguistic Data Consortium, as catalogue entry LDC93S1.",
        note=_MEDIA
    ),
    Entry(
        name="common_voice",
        tier="C",
        title="Common Voice",
        licence="CC0-1.0",
        citation=(
            "Ardila, R., Branson, M., Davis, K., Henretty, M., Kohler, M., Meyer, J., Morais, R., Saunders, "
            "L., Tyers, F. M., & Weber, G. (2020). Common Voice: A massively-multilingual speech corpus. "
            "Proceedings of LREC, 4218-4222."
        ),
        instructions="Download a release from https://commonvoice.mozilla.org/datasets, which asks for an email.",
        note=_MEDIA
    ),
    Entry(
        name="things_images",
        tier="C",
        title="THINGS images",
        licence="THINGS image terms, for academic research only",
        citation=(
            "Hebart, M. N., Dickter, A. H., Kidder, A., Kwok, W. Y., Corriveau, A., Van Wicklin, C., & Baker, "
            "C. I. (2019). THINGS: A database of 1,854 object concepts and more than 26,000 naturalistic "
            "object images. PLOS ONE, 14(10), e0223792."
        ),
        instructions="Download the images from https://osf.io/jum2f, which gives the archive password.",
        note=_MEDIA
    ),
    Entry(
        name="psych101_test",
        tier="C",
        title="Psych-101 test split",
        licence=_UNCONFIRMED,
        citation=(
            "Binz, M., Akata, E., Bethge, M., Brändle, F., Callaway, F., Coda-Forno, J., ... Schulz, E. "
            "(2025). A foundation model to predict and capture human cognition. Nature, 644, 1002-1009."
        ),
        instructions=(
            "It is gated on the Hugging Face Hub: request access at "
            "https://huggingface.co/datasets/marcelbinz/Psych-101-test, download it, and pass path= to the file."
        ),
        note="Its loader hasn't been written yet."
    ),
    Entry(
        name="elp_items",
        tier="C",
        title="English Lexicon Project, item-level",
        licence="ELP terms of use",
        citation=(
            "Balota, D. A., Yap, M. J., Cortese, M. J., Hutchison, K. A., Kessler, B., Loftis, B., Neely, J. "
            "H., Nelson, D. L., Simpson, G. B., & Treiman, R. (2007). The English Lexicon Project. Behavior "
            "Research Methods, 39, 445-459."
        ),
        instructions=(
            "Register at https://elexicon.wustl.edu, export the item-level data as a CSV file that includes "
            "the Word column, and pass path= to it."
        ),
        note="Its loader hasn't been written yet."
    ),
    Entry(
        name="men",
        tier="D",
        title="MEN",
        licence=_UNCONFIRMED,
        citation=(
            "Bruni, E., Tran, N.-K., & Baroni, M. (2014). Multimodal distributional semantics. Journal of "
            "Artificial Intelligence Research, 49, 1-47."
        ),
        note=_NOT_OFFERED
    ),
    Entry(
        name="usf_norms",
        tier="D",
        title="USF free association norms",
        licence=_UNCONFIRMED,
        citation=(
            "Nelson, D. L., McEvoy, C. L., & Schreiber, T. A. (2004). The University of South Florida free "
            "association, rhyme, and word fragment norms. Behavior Research Methods, Instruments, & Computers, "
            "36, 402-407."
        ),
        note=_NOT_OFFERED
    ),
    Entry(
        name="kuperman_aoa",
        tier="D",
        title="Kuperman et al. (2012) age-of-acquisition ratings",
        licence=_UNCONFIRMED,
        citation=(
            "Kuperman, V., Stadthagen-Gonzalez, H., & Brysbaert, M. (2012). Age-of-acquisition ratings for "
            "30,000 English words. Behavior Research Methods, 44, 978-990."
        ),
        note=_NOT_OFFERED
    ),
    Entry(
        name="drm_lists",
        tier="D",
        title="DRM word lists",
        licence=_UNCONFIRMED,
        citation=(
            "Roediger, H. L., & McDermott, K. B. (1995). Creating false memories: Remembering words not "
            "presented in lists. Journal of Experimental Psychology: Learning, Memory, and Cognition, 21(4), "
            "803-814."
        ),
        note=f"{_NOT_OFFERED} The DRM paradigm takes your own lists through stimuli=."
    ),
    Entry(
        name="cogbench",
        tier="D",
        title="CogBench",
        licence=_UNCONFIRMED,
        citation=(
            "Coda-Forno, J., Binz, M., Wang, J. X., & Schulz, E. (2024). CogBench: A large language model "
            "walks into a psychology lab. Proceedings of the 41st International Conference on Machine "
            "Learning."
        ),
        note=_NOT_OFFERED
    )
)

REGISTRY: dict[str, Entry] = {entry.name: entry for entry in _ENTRIES}
