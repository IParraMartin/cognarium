"""Turning files into datasets and trials, and fetching datasets from the registry."""

from __future__ import annotations

from cognarium.data.convert import Dataset, from_csv, from_json
from cognarium.data.registry import Entry, available, load

__all__ = [
    "Dataset",
    "Entry",
    "available",
    "from_csv",
    "from_json",
    "load"
]
