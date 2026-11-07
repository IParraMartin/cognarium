from __future__ import annotations

import pytest
import torch

from cognarium.device import resolve_device, resolve_dtype


def test_auto_prefers_cuda_then_mps_then_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto").type == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto").type == "mps"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("auto").type == "cpu"


def test_unavailable_device_raises_instead_of_falling_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="cuda:1"):
        resolve_device("cuda:1")


def test_dtype_auto_uses_preference_where_supported():
    assert resolve_dtype("auto", torch.device("cpu"), prefer=torch.float64) == torch.float64
    assert resolve_dtype("auto", torch.device("mps"), prefer=torch.float64) == torch.float32


def test_float64_on_mps_raises():
    with pytest.raises(ValueError, match="float64"):
        resolve_dtype("float64", torch.device("mps"))


def test_dtype_accepts_names():
    assert resolve_dtype("float16", torch.device("cpu")) == torch.float16
