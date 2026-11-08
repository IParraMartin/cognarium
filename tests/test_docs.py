"""Every Python block in the docs runs, and every built-in paradigm has its documented section."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import cognarium as cg
from cognarium.paradigms import PARADIGMS

DOCS = Path(__file__).parent.parent / "docs"
PAGES = [*sorted(DOCS.glob("*.md")), *sorted((DOCS / "paradigms").glob("*.md"))]
BLOCK = re.compile(r"(<!-- no-run -->\s*\n)?```python\n(.*?)```", re.DOTALL)
HEADINGS = (
    "Maps onto",
    "What it can't capture",
    "Run it",
    "A real run"
)
NETWORK_ONLY = ("cg.data.load(", "push_to_hub(")


@pytest.fixture
def tiny_load(byte_model_and_tokenizer, monkeypatch):
    model, tokenizer = byte_model_and_tokenizer
    real = cg.load

    def load(name, **settings):
        return real(model, tokenizer=tokenizer, device="cpu")

    monkeypatch.setattr(cg, "load", load)


def blocks(page):
    return BLOCK.findall(page.read_text(encoding="utf-8"))


def section(page, title):
    text = page.read_text(encoding="utf-8")
    found = re.search(rf"^## {re.escape(title)}\n(.*?)(?=^## |\Z)", text, flags=re.DOTALL | re.MULTILINE)
    return None if found is None else found.group(1)


def test_there_are_pages_to_run():
    assert len(PAGES) >= 15
    assert all(blocks(page) for page in DOCS.glob("paradigms/*.md"))


@pytest.mark.parametrize("page", PAGES, ids=lambda page: str(page.relative_to(DOCS)))
def test_page_code_runs(
    page,
    tiny_load,
    tmp_path,
    monkeypatch
):
    monkeypatch.chdir(tmp_path)
    namespace = {"__name__": "__docs__"}
    for skipped, code in blocks(page):
        if skipped:
            continue
        exec(compile(code, str(page), "exec"), namespace)


def test_skipped_blocks_only_download_or_push():
    for page in PAGES:
        for skipped, code in blocks(page):
            if skipped:
                assert any(call in code for call in NETWORK_ONLY), f"{page.name} skips a block that runs offline"


@pytest.mark.parametrize("name", sorted(PARADIGMS))
def test_every_paradigm_has_a_section_and_a_real_run(name):
    paradigm = PARADIGMS[name]
    page = DOCS / "paradigms" / f"{paradigm.__module__.rsplit('.', 1)[-1]}.md"
    assert page.exists(), page
    found = section(page, paradigm.__name__)
    assert found is not None, f"{page.name} has no section for {paradigm.__name__}"
    for heading in HEADINGS:
        assert f"### {heading}\n" in found, f"{paradigm.__name__} lacks '{heading}'"
    assert f"--8<-- \"results/{name}.md\"" in found
    assert (DOCS / "results" / f"{name}.md").exists()


def result_rows(name):
    lines = (DOCS / "results" / f"{name}.md").read_text(encoding="utf-8").splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith("| "))
    header = [cell.strip() for cell in lines[start].strip("|").split("|")]
    rows = []
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        rows.append(dict(zip(header, [cell.strip() for cell in line.strip("|").split("|")], strict=True)))
    return rows


def test_the_home_screen_shows_the_runs_on_the_result_pages():
    screen = (DOCS / "javascripts" / "paradigm-runs.js").read_text(encoding="utf-8")
    runs = json.loads(screen[screen.index("["):screen.rindex("]") + 1])
    assert [run["name"] for run in runs] == list(PARADIGMS)
    for run in runs:
        rows = result_rows(run["name"])
        if "series" in run:
            fidelity = [row["value"] for row in rows if row["metric"] == "fidelity"]
            assert [f"{value:.4g}" for value in run["series"]["fidelity"]] == fidelity
            continue
        shown = [
            (effect["effect"], f"{effect['difference']:.4g}", str(effect["detected"]))
            for effect in run["effects"]
        ]
        assert shown == [(row["effect"], row["difference"], row["detected"]) for row in rows], run["name"]
