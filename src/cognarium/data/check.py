"""Checks that every verified dataset's link still answers: python -m cognarium.data.check."""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from collections.abc import Mapping

from cognarium.data.download import resolve_url, user_agent
from cognarium.data.registry import REGISTRY, Entry

_TIMEOUT = 30
_HEAD_REFUSED = frozenset({403, 405, 501})


def check_links(entries: Mapping[str, Entry] | None = None) -> dict[str, str]:
    """Asks the server of each verified entry whether its file is still there.

    Each request names only the library as its client. A server that refuses a `HEAD`
    request is asked for the file's first byte instead, so nothing large is downloaded.

    Args:
        entries: The entries to check. Defaults to the whole registry.

    Returns:
        The entries whose links failed, mapped to what went wrong. Empty when every link
        answers.
    """
    entries = REGISTRY if entries is None else entries
    failures = {}
    for name, entry in entries.items():
        if entry.verified is None or entry.url is None:
            continue
        problem = _problem(resolve_url(entry.url))
        if problem is not None:
            failures[name] = problem
    return failures


def main(entries: Mapping[str, Entry] | None = None) -> int:
    """Checks the links and prints the ones that failed.

    Args:
        entries: The entries to check. Defaults to the whole registry.

    Returns:
        0 when every link answers, and 1 otherwise.
    """
    failures = check_links(entries)
    for name, problem in failures.items():
        print(f"{name}: {problem}", file=sys.stderr)
    if failures:
        return 1
    print("Every dataset link answers.")
    return 0


def _problem(url: str) -> str | None:
    headers = {"User-Agent": user_agent()}
    try:
        _open(urllib.request.Request(url, method="HEAD", headers=headers))
        return None
    except urllib.error.HTTPError as error:
        if error.code not in _HEAD_REFUSED:
            return f"HTTP {error.code} from {url}"
    except (urllib.error.URLError, OSError) as error:
        return f"{error} from {url}"
    try:
        _open(urllib.request.Request(url, headers={**headers, "Range": "bytes=0-0"}))
        return None
    except urllib.error.HTTPError as error:
        return f"HTTP {error.code} from {url}"
    except (urllib.error.URLError, OSError) as error:
        return f"{error} from {url}"


def _open(request: urllib.request.Request) -> None:
    with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
        response.read(1)


if __name__ == "__main__":
    sys.exit(main())
