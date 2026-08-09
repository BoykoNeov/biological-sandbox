"""Models. Importing this package registers every built-in model by name.

The validated core lives here directly. The speculative arc, when it exists,
lives quarantined under ``models/ecosystem/`` — a sibling, never entangled with
the validated models. That directory boundary is the structural expression of
the verifiable/exploratory split.
"""

from __future__ import annotations

from sandbox.models import (
    birth_death,  # noqa: F401  (import triggers registration)
    hh_voltage_clamp,  # noqa: F401  (import triggers registration)
    hodgkin_huxley,  # noqa: F401  (import triggers registration)
    isomerization,  # noqa: F401  (import triggers registration)
    repressilator,  # noqa: F401  (import triggers registration)
    wright_fisher,  # noqa: F401  (import triggers registration)
)
from sandbox.models.birth_death import BirthDeath, BirthDeathParams, BirthDeathState
from sandbox.models.hh_voltage_clamp import (
    HHVoltageClamp,
    HHVoltageClampParams,
    HHVoltageClampState,
)
from sandbox.models.hodgkin_huxley import HHParams, HHState, HodgkinHuxley
from sandbox.models.isomerization import (
    Isomerization,
    IsomerizationParams,
    IsomerizationState,
)
from sandbox.models.repressilator import (
    Repressilator,
    RepressilatorParams,
    RepressilatorState,
)
from sandbox.models.wright_fisher import WFParams, WFState, WrightFisher

__all__ = [
    "BirthDeath",
    "BirthDeathParams",
    "BirthDeathState",
    "HHParams",
    "HHState",
    "HHVoltageClamp",
    "HHVoltageClampParams",
    "HHVoltageClampState",
    "HodgkinHuxley",
    "Isomerization",
    "IsomerizationParams",
    "IsomerizationState",
    "Repressilator",
    "RepressilatorParams",
    "RepressilatorState",
    "WFParams",
    "WFState",
    "WrightFisher",
]
