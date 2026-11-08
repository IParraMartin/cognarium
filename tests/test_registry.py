"""The dataset registry: tiers, downloads, checksums, caching and link checks."""

from __future__ import annotations

import hashlib
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from helpers import ScriptedSubject

import cognarium as cg
from cognarium.data import check, registry
from cognarium.data.registry import Entry

WORDS = b"word,rating\ncat,1.0\ndog,2.0\nfish,3.0\n"
PAIRS = (
    b"good,bad,phenomenon\n"
    b"The cats sleep.,The cats sleeps.,agreement\n"
    b"The dog barks.,The dog bark.,agreement\n"
    b"Who did you see?,Who did you see him?,islands\n"
)
SPEC = {
    "A": (
        "wordsim353",
        "blimp",
        "syntaxgym",
        "provo",
        "zuco",
        "things_triplets",
        "thingsplus",
        "geirhos_cue_conflict",
        "speech_commands",
        "librispeech",
        "psych101",
        "bigtom",
        "nosofsky1994",
        "navarro2018",
        "omniglot",
        "narratives",
        "many_labs_2",
        "meco"
    ),
    "B": (
        "swow",
        "tomi",
        "cifar10h",
        "natural_stories",
        "fantom",
        "simlex999",
        "semantic_priming_project",
        "elp_trials",
        "brysbaert_concreteness",
        "warriner"
    ),
    "C": (
        "timit",
        "common_voice",
        "things_images",
        "psych101_test",
        "elp_items"
    ),
    "D": (
        "men",
        "usf_norms",
        "kuperman_aoa",
        "drm_lists",
        "cogbench"
    )
}


class Files:
    """What the local server holds, and what it was asked for."""

    def __init__(self):
        self.files = {}
        self.cut = set()
        self.requests = []
        self.url = None
        self.stop = None


@pytest.fixture
def local_server():
    files = Files()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self._answer(body=True)

        def do_HEAD(self):
            self._answer(body=False)

        def _answer(self, body):
            files.requests.append(self.path)
            content = files.files.get(self.path)
            if content is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            if body:
                self.wfile.write(content[:len(content) // 2] if self.path in files.cut else content)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    files.url = f"http://127.0.0.1:{server.server_port}"
    stopped = []

    def stop():
        if not stopped:
            server.shutdown()
            server.server_close()
            stopped.append(True)

    files.stop = stop
    yield files
    stop()


def read_words(path, **options):
    return cg.data.from_csv(path, text="word", **options)


def read_pairs(path, **options):
    return cg.data.from_csv(path, **options)


def entry_for(
    server,
    monkeypatch,
    *,
    name="toy",
    content=WORDS,
    loader=read_words,
    **fields
):
    server.files[f"/{name}.csv"] = content
    settings = {
        "tier": "A",
        "title": "Toy words",
        "licence": "CC-BY-4.0",
        "citation": "Toy, A. (2026). A toy dataset.",
        "url": f"{server.url}/{name}.csv",
        "sha256": hashlib.sha256(content).hexdigest(),
        "size": len(content),
        "loader": loader,
        "verified": "2026-11-07",
        **fields
    }
    entry = Entry(name=name, **settings)
    monkeypatch.setitem(registry.REGISTRY, name, entry)
    return entry


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_load_downloads_checks_and_caches(
    local_server,
    tmp_path,
    monkeypatch,
    capsys
):
    entry_for(local_server, monkeypatch)
    first = cg.data.load("toy", cache_dir=tmp_path)
    assert "Toy, A. (2026)" in capsys.readouterr().out
    second = cg.data.load("toy", cache_dir=tmp_path)
    assert local_server.requests == ["/toy.csv"]
    assert first.to_pandas().equals(second.to_pandas())
    assert list(first.to_pandas()["word"]) == ["cat", "dog", "fish"]
    assert registry.REGISTRY["toy"].sha256 in first.source


def test_tampered_download_raises_and_leaves_nothing_behind(local_server, tmp_path, monkeypatch):
    entry_for(local_server, monkeypatch)
    local_server.files["/toy.csv"] = WORDS.replace(b"cat", b"cow")
    with pytest.raises(ValueError, match="checksum"):
        cg.data.load("toy", cache_dir=tmp_path)
    assert not any(path.is_file() for path in tmp_path.rglob("*"))


def test_truncated_download_raises(local_server, tmp_path, monkeypatch):
    entry_for(local_server, monkeypatch)
    local_server.files["/toy.csv"] = WORDS[:10]
    with pytest.raises(ValueError, match="checksum"):
        cg.data.load("toy", cache_dir=tmp_path)


def test_interrupted_download_leaves_a_retryable_cache(local_server, tmp_path, monkeypatch):
    entry_for(local_server, monkeypatch)
    local_server.cut.add("/toy.csv")
    with pytest.raises(ConnectionError, match="interrupted|reach"):
        cg.data.load("toy", cache_dir=tmp_path)
    assert not any(path.is_file() for path in tmp_path.rglob("*"))
    local_server.cut.clear()
    assert len(cg.data.load("toy", cache_dir=tmp_path)) == 3


def test_tampered_cache_file_is_refused_and_refetched(local_server, tmp_path, monkeypatch):
    entry = entry_for(local_server, monkeypatch)
    cached = registry.fetch(entry, cache_dir=tmp_path)
    cached.write_bytes(WORDS.replace(b"cat", b"cow"))
    dataset = cg.data.load("toy", cache_dir=tmp_path)
    assert list(dataset.to_pandas()["word"]) == ["cat", "dog", "fish"]
    assert local_server.requests == ["/toy.csv", "/toy.csv"]


def test_cached_file_works_with_the_network_down(local_server, tmp_path, monkeypatch):
    entry_for(local_server, monkeypatch)
    cg.data.load("toy", cache_dir=tmp_path)
    local_server.stop()
    assert len(cg.data.load("toy", cache_dir=tmp_path)) == 3


def test_network_down_with_no_cache_says_so(tmp_path, monkeypatch):
    entry = Entry(
        name="toy",
        tier="A",
        title="Toy words",
        licence="CC-BY-4.0",
        citation="Toy (2026).",
        url=f"http://127.0.0.1:{free_port()}/toy.csv",
        sha256=hashlib.sha256(WORDS).hexdigest(),
        size=len(WORDS),
        loader=read_words,
        verified="2026-11-07"
    )
    monkeypatch.setitem(registry.REGISTRY, "toy", entry)
    with pytest.raises(ConnectionError, match="reach .*no copy"):
        cg.data.load("toy", cache_dir=tmp_path)


def test_tier_b_needs_accepting_the_terms(local_server, tmp_path, monkeypatch):
    entry_for(
        local_server,
        monkeypatch,
        tier="B",
        licence="CC BY-NC 4.0",
        note="Non-commercial use only."
    )
    with pytest.raises(ValueError, match="accept_terms=True") as raised:
        cg.data.load("toy", cache_dir=tmp_path)
    assert "CC BY-NC 4.0" in str(raised.value)
    assert "Non-commercial use only." in str(raised.value)
    assert local_server.requests == []
    assert len(cg.data.load("toy", cache_dir=tmp_path, accept_terms=True)) == 3


def test_tier_b_with_no_licence_says_no_rights_are_granted(monkeypatch):
    entry = Entry(
        name="toy",
        tier="B",
        title="Toy words",
        licence=registry.NO_LICENCE,
        citation="Toy (2026).",
        url="http://127.0.0.1:9/toy.csv",
        sha256="0" * 64,
        size=1,
        loader=read_words,
        verified="2026-11-07"
    )
    monkeypatch.setitem(registry.REGISTRY, "toy", entry)
    with pytest.raises(ValueError, match="no rights are granted"):
        cg.data.load("toy")


def test_tier_c_gives_instructions_then_reads_a_local_path(tmp_path, monkeypatch):
    entry = Entry(
        name="toy",
        tier="C",
        title="Toy words",
        licence="Licensed by its distributor",
        citation="Toy (2026).",
        loader=read_words,
        instructions="Request a copy from the distributor, then pass path= to the CSV."
    )
    monkeypatch.setitem(registry.REGISTRY, "toy", entry)
    with pytest.raises(ValueError, match="Request a copy from the distributor"):
        cg.data.load("toy")
    local = tmp_path / "mine.csv"
    local.write_bytes(WORDS)
    dataset = cg.data.load("toy", path=local)
    assert list(dataset.to_pandas()["word"]) == ["cat", "dog", "fish"]
    assert "mine.csv" in dataset.source


def test_tier_d_is_not_offered():
    for name in SPEC["D"]:
        with pytest.raises(ValueError, match="isn't offered"):
            cg.data.load(name)
    listed = cg.data.available(include_disabled=True)
    assert not set(SPEC["D"]) & set(listed["name"])


def test_unverified_entry_says_why(monkeypatch):
    entry = Entry(
        name="toy",
        tier="A",
        title="Toy words",
        licence="CC-BY-4.0",
        citation="Toy (2026).",
        note="The download needs a login."
    )
    monkeypatch.setitem(registry.REGISTRY, "toy", entry)
    with pytest.raises(ValueError, match="The download needs a login."):
        cg.data.load("toy")


def test_unknown_names_are_listed():
    with pytest.raises(ValueError, match="available"):
        cg.data.load("no_such_dataset")


def test_available_lists_tiers_and_status(local_server, monkeypatch):
    entry_for(local_server, monkeypatch)
    monkeypatch.setitem(
        registry.REGISTRY,
        "toy_off",
        Entry(
            name="toy_off",
            tier="A",
            title="Off",
            licence="CC-BY-4.0",
            citation="Off (2026).",
            note="Not checked yet."
        )
    )
    monkeypatch.setitem(
        registry.REGISTRY,
        "toy_local",
        Entry(
            name="toy_local",
            tier="C",
            title="Local",
            licence="Licensed by its distributor",
            citation="Local (2026).",
            loader=read_words,
            instructions="Ask the distributor."
        )
    )
    shown = cg.data.available()
    assert list(shown.columns) == [
        "name",
        "tier",
        "title",
        "licence",
        "size",
        "status"
    ]
    assert shown.set_index("name").loc["toy", "status"] == "ready"
    assert "toy_off" not in set(shown["name"])
    every = cg.data.available(include_disabled=True).set_index("name")
    assert every.loc["toy_off", "status"] == "disabled"
    assert every.loc["toy_local", "status"] == "local copy"


def test_every_entry_is_complete():
    for name, entry in registry.REGISTRY.items():
        assert entry.name == name
        assert entry.tier in {"A", "B", "C", "D"}, name
        assert entry.title and entry.citation and entry.licence, name
        if entry.tier == "D" or (entry.tier in {"A", "B"} and entry.verified is None):
            assert entry.note, name
        if entry.tier in {"A", "B"} and entry.verified is not None:
            assert entry.url and entry.loader is not None, name
            assert entry.sha256 is not None and len(entry.sha256) == 64, name
            assert entry.size is not None and entry.size > 0, name
        if entry.tier == "C":
            assert entry.instructions, name
            assert entry.loader is not None or entry.note, name


def test_every_spec_dataset_has_an_entry_or_is_media():
    for tier, names in SPEC.items():
        for name in names:
            assert name in registry.REGISTRY, name
            found = registry.REGISTRY[name].tier
            assert found == tier or {found, tier} <= {"A", "B"}, name


def test_import_does_not_load_pooch():
    code = "import sys, cognarium, cognarium.data; cognarium.data.available(); print('pooch' in sys.modules)"
    found = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True
    )
    assert found.stdout.strip() == "False"


def test_link_check_reports_dead_links(local_server, monkeypatch, capsys):
    alive = entry_for(local_server, monkeypatch)
    dead = entry_for(local_server, monkeypatch, name="gone")
    del local_server.files["/gone.csv"]
    assert check.main({"toy": alive}) == 0
    assert check.main({"toy": alive, "gone": dead}) == 1
    printed = capsys.readouterr()
    assert "gone" in printed.out + printed.err
    assert "404" in printed.out + printed.err


class Acceptability(cg.Paradigm):
    item_columns = ("good", "bad")

    def trials(self, rng):
        for index, row in enumerate(self.item_rows(rng)):
            yield cg.Trial(
                cg.Stimulus(text=f"A: {row['good']}\nB: {row['bad']}\nWhich is better?"),
                options=["A", "B"],
                answer="A",
                item=index
            )


def test_a_paradigm_runs_from_a_registry_dataset(local_server, tmp_path, monkeypatch):
    entry = entry_for(
        local_server,
        monkeypatch,
        name="pairs",
        content=PAIRS,
        loader=read_pairs
    )
    paradigm = Acceptability(stimuli=cg.data.load("pairs", cache_dir=tmp_path / "cache"))
    results = cg.run(
        paradigm,
        ScriptedSubject(),
        out=tmp_path / "run",
        progress=False
    )
    assert len(results.to_pandas()) == 3
    assert entry.sha256 in results.manifest["paradigm"]["stimuli"]
