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

from cognarium.data import loaders
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
        licence="CC BY 4.0",
        citation=(
            "Finkelstein, L., Gabrilovich, E., Matias, Y., Rivlin, E., Solan, Z., Wolfman, G., & Ruppin, E. "
            "(2002). Placing search in context: The concept revisited. ACM Transactions on Information "
            "Systems, 20(1), 116-131."
        ),
        url="https://gabrilovich.com/resources/data/wordsim353/wordsim353.zip",
        sha256="4f12a5fe9650184662306f8ae72c7e6bb44c2627e0275e9f9c6eb067e3441a22",
        size=23257,
        loader=loaders.wordsim353,
        verified="2026-11-07"
    ),
    Entry(
        name="blimp",
        tier="A",
        title="BLiMP",
        licence="CC BY 4.0",
        citation=(
            "Warstadt, A., Parrish, A., Liu, H., Mohananey, A., Peng, W., Wang, S.-F., & Bowman, S. R. (2020). "
            "BLiMP: The Benchmark of Linguistic Minimal Pairs for English. Transactions of the Association "
            "for Computational Linguistics, 8, 377-392."
        ),
        url="https://raw.githubusercontent.com/alexwarstadt/blimp/3e56b06fcabca9b30822fc66435fca6b1aa40bb1/BLiMP.zip",
        sha256="172a1a714d72e72a89e8384dd531483f3d0f8a9aee3d33dccaf7992d8cecbb73",
        size=2354164,
        loader=loaders.blimp,
        verified="2026-11-07"
    ),
    Entry(
        name="syntaxgym",
        tier="A",
        title="SyntaxGym test suites of Hu et al. (2020)",
        licence="MIT",
        citation=(
            (
            "Hu, J., Gauthier, J., Qian, P., Wilcox, E., & Levy, R. (2020). A systematic assessment of syntactic "
            "generalization in neural language models. Proceedings of ACL, 1725-1744."
        )
        ),
        url="https://github.com/cpllab/syntactic-generalization/archive/43f1f86b746254406dafc89bd17b9d4976607ef6.zip",
        sha256="22f6cb492dc5fc5a8e67542e663f42c6fae921ce050b31e91bf91e27389080b4",
        size=4062522,
        loader=loaders.syntaxgym,
        verified="2026-11-07"
    ),
    Entry(
        name="provo",
        tier="A",
        title="Provo Corpus",
        licence="CC BY 4.0",
        citation=(
            "Luke, S. G., & Christianson, K. (2018). The Provo Corpus: A large eye-tracking corpus with "
            "predictability norms. Behavior Research Methods, 50(2), 826-833."
        ),
        url="https://osf.io/download/a32be/",
        sha256="38aedcb29bc9171009916eb2bcc2375729f104a2a1005c64a563da94b611b9e7",
        size=69662713,
        loader=loaders.provo,
        verified="2026-11-07",
        file_name="Provo_Corpus-Eyetracking_Data.csv"
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
        licence="CC BY 4.0",
        citation=(
            "Hebart, M. N., Contier, O., Teichmann, L., Rockter, A. H., Zheng, C. Y., Kidder, A., Corriveau, "
            "A., Vaziri-Pashkam, M., & Baker, C. I. (2023). THINGS-data, a multimodal collection of "
            "large-scale datasets for investigating object representations in human brain and behavior. "
            "eLife, 12, e82580."
        ),
        url="https://osf.io/download/h2smy/",
        sha256="68f2bbc330b17537881f81b1aa806fc2179b7accecc3a44d54fb25ae3de7f3e5",
        size=360791944,
        loader=loaders.things_triplets,
        verified="2026-11-07",
        file_name="triplets_large_final_correctednc_correctedorder.csv"
    ),
    Entry(
        name="thingsplus",
        tier="A",
        title="THINGSplus norms",
        licence="CC BY",
        citation=(
            "Stoinski, L. M., Perkuhn, J., & Hebart, M. N. (2024). THINGSplus: New norms and metadata for the "
            "THINGS database of 1854 object concepts and 26,107 natural object images. Behavior Research "
            "Methods, 56, 1583-1603."
        ),
        url="https://osf.io/download/7cz69/",
        sha256="708a0f275162b783e31fe249cc92f01250bb160f3bc33dc287be33e1601d84b3",
        size=9519971,
        loader=loaders.thingsplus,
        verified="2026-11-07",
        file_name="_property-ratings.tsv"
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
        licence="Apache-2.0",
        citation=(
            "Binz, M., Akata, E., Bethge, M., Brändle, F., Callaway, F., Coda-Forno, J., ... Schulz, E. "
            "(2025). A foundation model to predict and capture human cognition. Nature, 644, 1002-1009."
        ),
        note="Its only file is 859 MB, and downloads over 500 MB haven't been verified for the registry."
    ),
    Entry(
        name="bigtom",
        tier="A",
        title="BigToM",
        licence="MIT",
        citation=(
            "Gandhi, K., Fränken, J.-P., Gerstenberg, T., & Goodman, N. D. (2023). Understanding social "
            "reasoning in language models with language models. Advances in Neural Information Processing "
            "Systems, Datasets and Benchmarks Track."
        ),
        url=(
            "https://raw.githubusercontent.com/cicl-stanford/procedural-evals-tom/"
            "fe647d680bddb69f738519313bed625f9e93b549/data/bigtom/bigtom.csv"
        ),
        sha256="a14884fbff5b61d3fa194ab166764b553ab795fcb1359e69d1282eadf3011a46",
        size=273557,
        loader=loaders.bigtom,
        verified="2026-11-07"
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
        note=(
            "Its authors don't share the data as a file, and the only machine-readable copy is a "
            "third-party transcription of the paper's table."
        )
    ),
    Entry(
        name="navarro2018",
        tier="A",
        title="Navarro et al. (2018) prior elicitation, MTurk sample",
        licence="CC BY 4.0",
        citation=(
            "Navarro, D. J., Perfors, A., Kary, A., Brown, S. D., & Donkin, C. (2018). When extremists win: "
            "Cultural transmission via iterated learning when populations are heterogeneous. Cognitive "
            "Science, 42(7), 2108-2149."
        ),
        url="https://osf.io/download/7ufj6/",
        sha256="0ab964a3b1ee18aaf4dd82174c616aeff89b802cd0b8975191b189732c7f6345",
        size=1548,
        loader=loaders.navarro2018,
        verified="2026-11-07",
        note=(
            "It is the prior elicitation the paper's simulations use. The iterated-learning "
            "chains aren't shared as data."
        ),
        file_name="data_MTurk_direct.csv"
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
        note="Its transcripts and word timings aren't publicly downloadable, since the shared stimuli are audio only."
    ),
    Entry(
        name="many_labs_2",
        tier="A",
        title="Many Labs 2, slate 1",
        licence="CC0 1.0",
        citation=(
            "Klein, R. A., Vianello, M., Hasselman, F., Adams, B. G., Adams, R. B., Jr., Alper, S., ... Nosek, "
            "B. A. (2018). Many Labs 2: Investigating variation in replicability across samples and settings. "
            "Advances in Methods and Practices in Psychological Science, 1(4), 443-490."
        ),
        url="https://osf.io/cwjp3/download",
        sha256="5c3dddf79a6e86dcbb9d7389e0c8a5a6f87d132966eafd319fe0bd58e1c19cd5",
        size=24922730,
        loader=loaders.many_labs_2,
        verified="2026-11-07",
        file_name="ML2_S1.csv"
    ),
    Entry(
        name="many_labs_2_slate_2",
        tier="A",
        title="Many Labs 2, slate 2",
        licence="CC0 1.0",
        citation=(
            "Klein, R. A., Vianello, M., Hasselman, F., Adams, B. G., Adams, R. B., Jr., Alper, S., ... Nosek, "
            "B. A. (2018). Many Labs 2: Investigating variation in replicability across samples and settings. "
            "Advances in Methods and Practices in Psychological Science, 1(4), 443-490."
        ),
        url="https://osf.io/jg9hc/download",
        sha256="842251429e2a27e64c2152ac48b34ff627e2e96723dd307f33830300b6c2f24e",
        size=43496574,
        loader=loaders.many_labs_2,
        verified="2026-11-07",
        file_name="ML2_S2.csv"
    ),
    Entry(
        name="meco",
        tier="A",
        title="MECO",
        licence="CC BY 4.0",
        citation=(
            "Siegelman, N., Schroeder, S., Acartürk, C., Ahn, H.-D., Alexeeva, S., Amenta, S., ... Kuperman, V. "
            "(2022). Expanding horizons of cross-linguistic research on reading: The Multilingual Eye-movement "
            "Corpus (MECO). Behavior Research Methods, 54, 2843-2863."
        ),
        note="Its word-level data is shared only as an R workspace, which cognarium can't read."
    ),
    Entry(
        name="swow",
        tier="B",
        title="Small World of Words (English)",
        licence="CC BY-NC-ND 3.0",
        citation=(
            "De Deyne, S., Navarro, D. J., Perfors, A., Brysbaert, M., & Storms, G. (2019). The \"Small World "
            "of Words\" English word association norms for over 12,000 cue words. Behavior Research Methods, "
            "51, 987-1006."
        ),
        note="Its download asks for your name and email, so cognarium can't fetch it for you."
    ),
    Entry(
        name="tomi",
        tier="B",
        title="ToMi",
        licence="CC BY-NC 4.0",
        citation=(
            "Le, M., Boureau, Y.-L., & Nickel, M. (2019). Revisiting the evaluation of theory of mind through "
            "question answering. Proceedings of EMNLP-IJCNLP, 5872-5877."
        ),
        url=(
            "https://raw.githubusercontent.com/facebookresearch/ToMi/"
            "dea2bca9b366c41cdab0bd717353cc02453e193e/tomi_balanced_story_types.zip"
        ),
        sha256="d8389ecf09749b4a49c56752a8ce44afac8c3d363391b5c29c8fa7032613ece6",
        size=403507,
        loader=loaders.tomi,
        verified="2026-11-07",
        note="Non-commercial use only. The repository states this licence for its code, and ships the data beside it."
    ),
    Entry(
        name="cifar10h",
        tier="B",
        title="CIFAR-10H",
        licence="CC BY-NC-SA 4.0",
        citation=(
            "Peterson, J. C., Battleday, R. M., Griffiths, T. L., & Russakovsky, O. (2019). Human uncertainty "
            "makes classification more robust. Proceedings of the IEEE/CVF International Conference on "
            "Computer Vision, 9617-9626."
        ),
        url=(
            "https://raw.githubusercontent.com/jcpeterson/cifar-10h/"
            "389f22d9d300ec71a4c44735d46f550c94b957ab/data/cifar10h-probs.npy"
        ),
        sha256="436ae98620e00073f9d08585ac870573b588853e7af9f54eb1647408114af77d",
        size=800128,
        loader=loaders.cifar10h,
        verified="2026-11-07",
        note="Non-commercial use only, and adaptations must be shared under the same licence."
    ),
    Entry(
        name="natural_stories",
        tier="B",
        title="Natural Stories",
        licence="CC BY-NC-SA 4.0",
        citation=(
            "Futrell, R., Gibson, E., Tily, H. J., Blank, I., Vishnevetsky, A., Piantadosi, S. T., & Fedorenko, "
            "E. (2021). The Natural Stories corpus: A reading-time corpus of English texts containing rare "
            "syntactic constructions. Language Resources and Evaluation, 55, 63-77."
        ),
        url=(
            "https://raw.githubusercontent.com/languageMIT/naturalstories/"
            "4700daad696e942f5aba23c957a7423d0de66612/naturalstories_RTS/processed_RTs.tsv"
        ),
        sha256="208ee6c451f3827734d5b2de32b91e2cf3bd76aa177c412cb426373fa7d50b22",
        size=96454912,
        loader=loaders.natural_stories,
        verified="2026-11-07",
        note="Non-commercial use only, and adaptations must be shared under the same licence."
    ),
    Entry(
        name="fantom",
        tier="B",
        title="FANToM",
        licence=NO_LICENCE,
        citation=(
            "Kim, H., Sclar, M., Zhou, X., Le Bras, R., Kim, G., Choi, Y., & Sap, M. (2023). FANToM: A "
            "benchmark for stress-testing machine theory of mind in interactions. Proceedings of EMNLP, "
            "14397-14413."
        ),
        url="https://storage.googleapis.com/ai2-mosaic-public/projects/fantom/fantom.tar.gz",
        sha256="1d08dfa0ea474c7f83b9bc7e3a7b466eab25194043489dd618b4c5223e1253a4",
        size=1040898,
        loader=loaders.fantom,
        verified="2026-11-07",
        note="Its README says the samples should only be used for evaluation."
    ),
    Entry(
        name="simlex999",
        tier="B",
        title="SimLex-999",
        licence=NO_LICENCE,
        citation=(
            "Hill, F., Reichart, R., & Korhonen, A. (2015). SimLex-999: Evaluating semantic models with "
            "(genuine) similarity estimation. Computational Linguistics, 41(4), 665-695."
        ),
        url="https://fh295.github.io/SimLex-999.zip",
        sha256="30f40eb18f41b08b65dfbc328f473e8b69ba444b1c9d053c1adde2324fc76315",
        size=16805,
        loader=loaders.simlex999,
        verified="2026-11-07",
        note="Its README asks you to cite the paper."
    ),
    Entry(
        name="wcs_chips",
        tier="B",
        title="World Color Survey chip coordinates",
        licence=NO_LICENCE,
        citation=(
            "Kay, P., Berlin, B., Maffi, L., Merrifield, W. R., & Cook, R. (2009). The World Color Survey. "
            "CSLI Publications."
        ),
        url="https://linguistics.berkeley.edu/wcs/data/cnum-maps/cnum-vhcm-lab-new.txt",
        sha256="82864ac378e051b6daa85e396f8e6f6d187c56d815938914b580e046dd95e3dc",
        size=12528,
        loader=loaders.wcs_chips,
        verified="2026-11-07",
        note="The archive asks that published work based on its data cite it."
    ),
    Entry(
        name="semantic_priming_project",
        tier="B",
        title="Semantic Priming Project",
        licence=NO_LICENCE,
        citation=(
            "Hutchison, K. A., Balota, D. A., Neely, J. H., Cortese, M. J., Cohen-Shikora, E. R., Tse, C.-S., "
            "Yap, M. J., Bengson, J. J., Niemeyer, D., & Buchanan, E. (2013). The semantic priming project. "
            "Behavior Research Methods, 45, 1099-1114."
        ),
        url="https://www.montana.edu/attmemlab/documents/all%20ldt%20subs_all%20trials3.xlsx",
        sha256="fbd5fb6ddf3f56885341df127cdaffba4458ed28999ab1c454a264fd8520d346",
        size=110512713,
        loader=loaders.semantic_priming_project,
        verified="2026-11-07",
        file_name="all_ldt_subs_all_trials3.xlsx"
    ),
    Entry(
        name="elp_trials",
        tier="B",
        title="English Lexicon Project, trial-level",
        licence=NO_LICENCE,
        citation=(
            "Balota, D. A., Yap, M. J., Cortese, M. J., Hutchison, K. A., Kessler, B., Loftis, B., Neely, J. "
            "H., Nelson, D. L., Simpson, G. B., & Treiman, R. (2007). The English Lexicon Project. Behavior "
            "Research Methods, 39, 445-459."
        ),
        url="https://osf.io/download/eu5ca/",
        sha256="f57f793896d38ccec369f25157685c9879c14c81bed513b7e0a510552614b210",
        size=36537458,
        loader=loaders.elp_trials,
        verified="2026-11-07",
        file_name="ldt_raw.zip"
    ),
    Entry(
        name="brysbaert_concreteness",
        tier="B",
        title="Brysbaert et al. (2014) concreteness ratings",
        licence=NO_LICENCE,
        citation=(
            "Brysbaert, M., Warriner, A. B., & Kuperman, V. (2014). Concreteness ratings for 40 thousand "
            "generally known English word lemmas. Behavior Research Methods, 46, 904-911."
        ),
        url=(
            "https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-013-0403-5/"
            "MediaObjects/13428_2013_403_MOESM1_ESM.xlsx"
        ),
        sha256="1673ead761e28833a40e82c0d20f10782955ced9366d600eafeefee0f2254545",
        size=2198394,
        loader=loaders.brysbaert_concreteness,
        verified="2026-11-07",
        note="It is the article's supplementary material, copyright 2013 Psychonomic Society."
    ),
    Entry(
        name="warriner",
        tier="B",
        title="Warriner et al. (2013) valence, arousal and dominance norms",
        licence=NO_LICENCE,
        citation=(
            "Warriner, A. B., Kuperman, V., & Brysbaert, M. (2013). Norms of valence, arousal, and dominance "
            "for 13,915 English lemmas. Behavior Research Methods, 45, 1191-1207."
        ),
        url=(
            "https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-012-0314-x/"
            "MediaObjects/13428_2012_314_MOESM1_ESM.zip"
        ),
        sha256="5a0db0437ef234219fb899caced5eb3195dbf9f84396637ab4db74c50ca441aa",
        size=1131281,
        loader=loaders.warriner,
        verified="2026-11-07",
        note="It is the article's supplementary material, copyright 2013 Psychonomic Society."
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
        loader=loaders.psych101,
        instructions=(
            "It is gated on the Hugging Face Hub: request access at "
            "https://huggingface.co/datasets/marcelbinz/Psych-101-test, download it, and pass path= to the file."
        )
    ),
    Entry(
        name="elp_items",
        tier="C",
        title="English Lexicon Project, item-level",
        licence="Terms given on registration",
        citation=(
            "Balota, D. A., Yap, M. J., Cortese, M. J., Hutchison, K. A., Kessler, B., Loftis, B., Neely, J. "
            "H., Nelson, D. L., Simpson, G. B., & Treiman, R. (2007). The English Lexicon Project. Behavior "
            "Research Methods, 39, 445-459."
        ),
        loader=loaders.elp_items,
        instructions=(
            "Register at https://elexicon.wustl.edu, export the item-level data as a CSV file that includes "
            "the Word column, and pass path= to it."
        )
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
        note=_NOT_OFFERED
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
