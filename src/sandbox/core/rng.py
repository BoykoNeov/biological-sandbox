"""Seeded RNG plumbing — reproducibility lives here.

The rule that matters: independent stochastic replicates must use *independently
seeded* generators, and the way to get that right is ``SeedSequence.spawn`` —
**not** ``default_rng(seed + i)``. Adjacent integer seeds can produce correlated
streams, which would quietly bias replicate statistics (a fixation fraction that
looks fine but is subtly wrong). Spawning derives well-separated child seeds from
a parent ``SeedSequence``, so replicates are both independent and reproducible
from the single top-level ``seed``.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator, SeedSequence


def make_rng(seed: int) -> Generator:
    """A single reproducible generator from an integer seed."""
    return np.random.default_rng(seed)


def spawn_rngs(seed: int, n: int) -> list[Generator]:
    """``n`` independent, reproducible generators derived from one ``seed``.

    Uses ``SeedSequence(seed).spawn(n)`` so the streams are statistically
    independent — the correct primitive for stochastic replicates.
    """
    parent = SeedSequence(seed)
    return [np.random.default_rng(child) for child in parent.spawn(n)]
