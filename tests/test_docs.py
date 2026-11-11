"""Every Python block in the docs runs, and every built-in paradigm has its documented section."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import cognarium as cg
from cognarium.paradigms import PARADIGMS

DOCS = Path(__file__).parent.parent / "docs"
PAGES = [*sorted(DOCS.glob("*.md")), *sorted((DOCS / "paradigms").glob("*.md"))]
BLOCK = re.compile(r"(<!-- no-run -->\s*\n)?```python\n(.*?)```", re.DOTALL)
HEADINGS = (
    "How it runs on a language model",
    "Maps onto",
    "What it can't capture",
    "Run it"
)
PREVIEW = re.compile(r"<!-- preview: (.*?) -->\s*\n```text\n(.*?)```", re.DOTALL)
NETWORK_ONLY = ("cg.data.load(", "push_to_hub(")


@pytest.fixture
def tiny_load(byte_model_and_tokenizer, omni_model_and_processor, monkeypatch):
    model, tokenizer = byte_model_and_tokenizer
    omni, processor = omni_model_and_processor
    real = cg.load

    def load(name, **settings):
        if settings.get("multimodal"):
            return real(omni, processor=processor, device="cpu")
        return real(model, tokenizer=tokenizer, device="cpu")

    monkeypatch.setattr(cg, "load", load)


def blocks(page):
    return BLOCK.findall(page.read_text(encoding="utf-8"))


def section(page, title):
    text = page.read_text(encoding="utf-8")
    found = re.search(rf"^## {re.escape(title)}\n(.*?)(?=^## |\Z)", text, flags=re.DOTALL | re.MULTILINE)
    return None if found is None else found.group(1)


def test_there_are_pages_to_run():
    assert len(PAGES) >= 11
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
def test_every_paradigm_has_a_documented_section(name):
    paradigm = PARADIGMS[name]
    page = DOCS / "paradigms" / f"{paradigm.__module__.rsplit('.', 1)[-1]}.md"
    assert page.exists(), page
    found = section(page, paradigm.__name__)
    assert found is not None, f"{page.name} has no section for {paradigm.__name__}"
    for heading in HEADINGS:
        assert f"### {heading}\n" in found, f"{paradigm.__name__} lacks '{heading}'"



def test_every_shown_prompt_is_what_the_paradigm_writes():
    shown = [match for page in PAGES for match in PREVIEW.findall(page.read_text(encoding="utf-8"))]
    assert len(shown) >= 3
    for expression, text in shown:
        assert eval(expression, {"cg": cg}) == text.rstrip("\n"), expression
