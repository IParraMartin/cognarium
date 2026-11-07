"""Downloading a registry file into the cache and checking it against its checksum."""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from cognarium.data.registry import Entry

_HUB = "hf://datasets/"
_TIMEOUT = 60


def fetch(entry: Entry, *, cache_dir: str | Path | None = None) -> Path:
    """Gives the path to an entry's file, downloading it only when the cache lacks it.

    A cached file is checked against the entry's checksum on every call, and a file that
    doesn't match is downloaded again. A download streams into a temporary file, which
    is checked before it replaces anything, so a failed or interrupted download leaves
    nothing behind and the next call simply tries again.

    Args:
        entry: A registry entry with a `url` and a `sha256`.
        cache_dir: Where downloads are kept. Defaults to the user cache folder that
            `pooch.os_cache("cognarium")` names. Each entry gets its own subfolder.

    Returns:
        The path to the checked file.

    Raises:
        ValueError: If the entry has no URL or checksum, or the download doesn't match
            its checksum. A file that doesn't match is removed.
        ConnectionError: If the URL can't be reached and there's no valid copy in the
            cache, the server refuses the request, or the download is interrupted.
    """
    import pooch
    import requests

    if entry.url is None or entry.sha256 is None:
        raise ValueError(f"{entry.title} has no URL and checksum to download it with.")
    url = resolve_url(entry.url)
    folder = Path(cache_dir) if cache_dir is not None else Path(pooch.os_cache("cognarium"))
    folder = folder / entry.name
    downloader = pooch.HTTPDownloader(
        progressbar=sys.stderr.isatty(),
        headers={"User-Agent": user_agent()},
        timeout=_TIMEOUT
    )
    try:
        found = pooch.retrieve(
            url,
            known_hash=f"sha256:{entry.sha256}",
            fname=entry.file_name or _file_name(url, entry.name),
            path=folder,
            downloader=downloader
        )
    except ValueError as error:
        raise ValueError(
            f"The download of {entry.title} doesn't match its checksum, so it was removed. The file at "
            f"{url} has changed since it was verified on {entry.verified}, or it was damaged on the way. "
            "Try again, and report it if it keeps happening."
        ) from error
    except requests.exceptions.HTTPError as error:
        status = error.response.status_code if error.response is not None else "an error"
        raise ConnectionError(f"{url} answered {status} when downloading {entry.title}.") from error
    except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ContentDecodingError) as error:
        raise ConnectionError(
            f"The download of {entry.title} from {url} was interrupted. Nothing was kept, so loading it "
            "again retries."
        ) from error
    except requests.exceptions.RequestException as error:
        raise ConnectionError(
            f"Couldn't reach {url} to download {entry.title}, and there's no copy in {folder}. Check the "
            "connection, or download the file yourself and pass path=."
        ) from error
    return Path(found)


def resolve_url(url: str) -> str:
    """Turns a pinned Hugging Face address into the URL that serves the file.

    Args:
        url: A direct URL, or `hf://datasets/<owner>/<repo>/<file>@<commit>`.

    Returns:
        The direct URL, unchanged, or the Hub's URL for that file at that commit.

    Raises:
        ValueError: If a Hugging Face address lacks its file or its commit.
    """
    if not url.startswith(_HUB):
        return url
    location, _, commit = url[len(_HUB):].rpartition("@")
    parts = location.split("/")
    if not commit or len(parts) < 3:
        raise ValueError(f"A Hugging Face address needs hf://datasets/<owner>/<repo>/<file>@<commit>, got {url}.")
    repo, file = "/".join(parts[:2]), "/".join(parts[2:])
    return f"https://huggingface.co/datasets/{repo}/resolve/{commit}/{file}"


def user_agent() -> str:
    """Names the library, and nothing about its user, to the servers it downloads from.

    Returns:
        The client name with the installed version.
    """
    try:
        return f"cognarium/{version('cognarium')}"
    except PackageNotFoundError:
        return "cognarium"


def _file_name(url: str, name: str) -> str:
    # an OSF link ends in /download, which says nothing about the file
    last = PurePosixPath(urlparse(url).path).name
    return last if "." in last else name
