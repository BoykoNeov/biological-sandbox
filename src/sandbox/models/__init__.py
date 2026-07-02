"""Models. Importing this package registers every built-in model by name.

The validated core lives here directly. The speculative arc, when it exists,
lives quarantined under ``models/ecosystem/`` — a sibling, never entangled with
the validated models. That directory boundary is the structural expression of
the verifiable/exploratory split.
"""

from __future__ import annotations

from sandbox.models import (
    birth_death,  # noqa: F401  (import triggers registration)
    isomerization,  # noqa: F401  (import triggers registration)
    wright_fisher,  # noqa: F401  (import triggers registration)
)
from sandbox.models.birth_death import BirthDeath, BirthDeathParams, BirthDeathState
from sandbox.models.isomerization import (
    Isomerization,
    IsomerizationParams,
    IsomerizationState,
)
from sandbox.models.wright_fisher import WFParams, WFState, WrightFisher

__all__ = [
    "BirthDeath",
    "BirthDeathParams",
    "BirthDeathState",
    "Isomerization",
    "IsomerizationParams",
    "IsomerizationState",
    "WFParams",
    "WFState",
    "WrightFisher",
]
