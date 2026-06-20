"""Shared test fixtures."""

from __future__ import annotations

import pytest

import sandbox.models  # noqa: F401  (registers models)
from sandbox.models.wright_fisher import WFParams


@pytest.fixture
def wf_params_factory():
    """Factory turning a plain dict into WFParams (the core stays params-agnostic)."""
    return lambda d: WFParams(**d)
