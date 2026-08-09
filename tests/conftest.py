"""Shared test fixtures, and the suite's CPU-politeness policy."""

from __future__ import annotations

import contextlib
import os
import sys

import pytest

import sandbox.models  # noqa: F401  (registers models)
from sandbox.models.wright_fisher import WFParams


def _lower_process_priority() -> None:
    """Drop this process below normal scheduling priority.

    The validation suite is the real check, and the convergence tests are
    CPU-bound for minutes at a time — run in parallel (``-n 4``) that is four
    cores pinned for the whole run. At normal priority that visibly starves
    interactive work on the same machine, so the suite deliberately yields to
    everything else: it takes slightly longer when the machine is busy and full
    speed when it is idle.

    This module is imported by every worker process, so each one lowers itself.
    Best-effort by design — a platform that refuses must not fail the suite.
    """
    with contextlib.suppress(Exception):
        if sys.platform == "win32":
            import ctypes

            below_normal = 0x00004000  # BELOW_NORMAL_PRIORITY_CLASS
            kernel32 = ctypes.windll.kernel32
            kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), below_normal)
        else:
            os.nice(10)


_lower_process_priority()


@pytest.fixture
def wf_params_factory():
    """Factory turning a plain dict into WFParams (the core stays params-agnostic)."""
    return lambda d: WFParams(**d)
