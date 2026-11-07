from __future__ import annotations

import pytest
import torch


@pytest.fixture(
    params=[
        "cpu",
        pytest.param("mps", marks=pytest.mark.mps),
        pytest.param("cuda", marks=pytest.mark.cuda)
    ]
)
def device(request: pytest.FixtureRequest) -> str:
    """Yields each device this machine has, skipping the others."""
    name = request.param
    if name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("mps is not available")
    if name == "cuda" and not torch.cuda.is_available():
        pytest.skip("cuda is not available")
    return name
