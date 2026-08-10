"""Models. Importing this package registers every built-in model by name.

The validated core lives here directly. The speculative arc, when it exists,
lives quarantined under ``models/ecosystem/`` — a sibling, never entangled with
the validated models. That directory boundary is the structural expression of
the verifiable/exploratory split.
"""

from __future__ import annotations

from sandbox.models import (
    adaptive_dynamics,  # noqa: F401  (import triggers registration)
    birth_death,  # noqa: F401  (import triggers registration)
    daisyworld,  # noqa: F401  (import triggers registration)
    glv,  # noqa: F401  (import triggers registration)
    glv_stochastic,  # noqa: F401  (import triggers registration)
    gray_scott,  # noqa: F401  (import triggers registration)
    hh_stochastic,  # noqa: F401  (import triggers registration)
    hh_voltage_clamp,  # noqa: F401  (import triggers registration)
    hodgkin_huxley,  # noqa: F401  (import triggers registration)
    isomerization,  # noqa: F401  (import triggers registration)
    lotka_volterra,  # noqa: F401  (import triggers registration)
    repressilator,  # noqa: F401  (import triggers registration)
    trait_branching,  # noqa: F401  (import triggers registration)
    wright_fisher,  # noqa: F401  (import triggers registration)
)
from sandbox.models.adaptive_dynamics import (
    AdaptiveDynamics,
    AdaptiveDynamicsParams,
    AdaptiveDynamicsState,
)
from sandbox.models.birth_death import BirthDeath, BirthDeathParams, BirthDeathState
from sandbox.models.daisyworld import Daisyworld, DaisyworldParams, DaisyworldState
from sandbox.models.glv import GLV, GLVParams, GLVState
from sandbox.models.glv_stochastic import (
    GLVStochastic,
    GLVStochasticParams,
    GLVStochasticState,
)
from sandbox.models.gray_scott import GrayScott, GrayScottParams, GrayScottState
from sandbox.models.hh_stochastic import (
    HHStochastic,
    HHStochasticParams,
    HHStochasticState,
)
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
from sandbox.models.lotka_volterra import LotkaVolterra, LVParams, LVState
from sandbox.models.repressilator import (
    Repressilator,
    RepressilatorParams,
    RepressilatorState,
)
from sandbox.models.trait_branching import (
    TraitBranching,
    TraitBranchingParams,
    TraitBranchingState,
)
from sandbox.models.wright_fisher import WFParams, WFState, WrightFisher

__all__ = [
    "AdaptiveDynamics",
    "AdaptiveDynamicsParams",
    "AdaptiveDynamicsState",
    "BirthDeath",
    "BirthDeathParams",
    "BirthDeathState",
    "Daisyworld",
    "DaisyworldParams",
    "DaisyworldState",
    "GLV",
    "GLVParams",
    "GLVState",
    "GLVStochastic",
    "GLVStochasticParams",
    "GLVStochasticState",
    "GrayScott",
    "GrayScottParams",
    "GrayScottState",
    "HHParams",
    "HHState",
    "HHStochastic",
    "HHStochasticParams",
    "HHStochasticState",
    "HHVoltageClamp",
    "HHVoltageClampParams",
    "HHVoltageClampState",
    "HodgkinHuxley",
    "Isomerization",
    "IsomerizationParams",
    "IsomerizationState",
    "LVParams",
    "LVState",
    "LotkaVolterra",
    "Repressilator",
    "RepressilatorParams",
    "RepressilatorState",
    "TraitBranching",
    "TraitBranchingParams",
    "TraitBranchingState",
    "WFParams",
    "WFState",
    "WrightFisher",
]
