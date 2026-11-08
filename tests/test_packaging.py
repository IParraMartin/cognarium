from __future__ import annotations

import email
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def wheel(tmp_path_factory):
    if shutil.which("uv") is None:
        pytest.skip("uv isn't installed")
    out = tmp_path_factory.mktemp("dist")
    subprocess.run(
        [
            "uv",
            "build",
            "--wheel",
            "--out-dir",
            str(out),
            str(ROOT)
        ],
        check=True,
        capture_output=True
    )
    (path,) = out.glob("cognarium-*.whl")
    return path


@pytest.fixture(scope="module")
def sdist(tmp_path_factory):
    if shutil.which("uv") is None:
        pytest.skip("uv isn't installed")
    out = tmp_path_factory.mktemp("sdist")
    subprocess.run(
        [
            "uv",
            "build",
            "--sdist",
            "--out-dir",
            str(out),
            str(ROOT)
        ],
        check=True,
        capture_output=True
    )
    (path,) = out.glob("cognarium-*.tar.gz")
    return path


def names_in(wheel):
    with zipfile.ZipFile(wheel) as archive:
        return archive.namelist()


def metadata_of(wheel):
    with zipfile.ZipFile(wheel) as archive:
        (name,) = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        return email.message_from_bytes(archive.read(name))


def test_the_wheel_holds_only_the_library(wheel):
    names = names_in(wheel)
    assert not [name for name in names if "superpowers" in name]
    assert not [name for name in names if name.startswith(("docs/", "tests/", "examples/"))]
    assert not [name for name in names if "__pycache__" in name or name.endswith(".pyc")]
    assert "cognarium/data/registry.py" in names


def test_the_wheel_carries_release_metadata(wheel):
    metadata = metadata_of(wheel)
    assert metadata["Version"] == "0.1.0"
    assert metadata["License-Expression"] == "Apache-2.0"
    assert metadata["Keywords"]
    classifiers = metadata.get_all("Classifier")
    assert "Programming Language :: Python :: 3.11" in classifiers
    assert "Topic :: Scientific/Engineering :: Artificial Intelligence" in classifiers
    urls = dict(line.split(", ", 1) for line in metadata.get_all("Project-URL"))
    assert urls["Source"] == "https://github.com/IParraMartin/cognarium"
    assert "Documentation" in urls


def test_the_sdist_holds_only_files_the_repository_tracks(sdist):
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("this check needs the git checkout")
    listed = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True)
    tracked = set(listed.stdout.split())
    with tarfile.open(sdist) as archive:
        # every member sits under one cognarium-<version>/ folder
        files = {member.name.split("/", 1)[1] for member in archive.getmembers() if member.isfile()}
    # local working files, run outputs and builds must never ship
    assert sorted(files - tracked - {"PKG-INFO"}) == []
    assert "src/cognarium/__init__.py" in files
    assert "README.md" in files
