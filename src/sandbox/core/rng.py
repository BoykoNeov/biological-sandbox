"""Seeded RNG plumbing — reproducibility lives here.

Independent stochastic replicates must use independently-seeded generators, and
the robust, composable way to get that is ``SeedSequence.spawn`` rather than
hand-constructing generators from adjacent integer seeds (``default_rng(seed +
i)``). A ``SeedSequence`` runs its entropy through a hashing step before seeding
the generator, so child seeds derived by ``spawn`` — and even nearby integer
seeds — produce well-separated, statistically independent streams. Spawning is
preferred because it composes: a parent sequence can spawn per-sweep-point
children, each of which spawns per-replicate children, all reproducible from one
top-level ``seed`` with no chance of two branches colliding.

(The folklore that "adjacent seeds correlate" is really a legacy ``RandomState``
/ raw-MT19937 hazard; the modern ``Generator`` + ``SeedSequence`` machinery
mitigates it. We still spawn rather than add offsets — for composability and to
keep the seeding tree explicit, not because ``seed + i`` is corrupt.)
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator, SeedSequence


def make_rng(seed: int) -> Generator:
    """A single reproducible generator from an integer seed."""
    return np.random.default_rng(seed)


def spawn_rngs(seed: int | SeedSequence, n: int) -> list[Generator]:
    """``n`` independent, reproducible generators.

    ``seed`` may be an integer (top of a fresh seeding tree) or an existing
    ``SeedSequence`` (a branch of one) — so callers can build a hierarchy, e.g.
    spawn one ``SeedSequence`` per sweep point and then spawn ``n`` replicate
    generators from each, with no risk of collisions between branches.
    """
    parent = seed if isinstance(seed, SeedSequence) else SeedSequence(seed)
    return [np.random.default_rng(child) for child in parent.spawn(n)]
