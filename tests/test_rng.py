"""RNG plumbing: reproducible and independent replicate streams."""

from __future__ import annotations

import numpy as np

from sandbox.core.rng import make_rng, spawn_rngs


def test_make_rng_is_reproducible():
    a = make_rng(42).standard_normal(1000)
    b = make_rng(42).standard_normal(1000)
    assert np.array_equal(a, b)


def test_spawn_is_reproducible_from_seed():
    first = [r.standard_normal(100) for r in spawn_rngs(7, 5)]
    second = [r.standard_normal(100) for r in spawn_rngs(7, 5)]
    for x, y in zip(first, second, strict=True):
        assert np.array_equal(x, y)


def test_spawned_streams_are_independent():
    draws = [r.standard_normal(5000) for r in spawn_rngs(7, 4)]
    # Distinct streams: no two replicate draws are identical, and pairwise
    # correlations are small (the failure mode `default_rng(seed + i)` risks).
    for i in range(len(draws)):
        for j in range(i + 1, len(draws)):
            assert not np.array_equal(draws[i], draws[j])
            corr = np.corrcoef(draws[i], draws[j])[0, 1]
            assert abs(corr) < 0.1
