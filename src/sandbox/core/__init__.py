"""Core: the protocol and the shared services built on top of it."""

from __future__ import annotations

from sandbox.core.protocol import (
    Experiment,
    Model,
    Result,
    State,
    TerminableModel,
    ValidatableModel,
)
from sandbox.core.recorder import Trajectory, run_replicate
from sandbox.core.registry import available, get_model, register
from sandbox.core.rng import make_rng, spawn_rngs
from sandbox.core.sweep import run_experiment
from sandbox.core.validation import Check, ValidationReport, validate

__all__ = [
    "Check",
    "Experiment",
    "Model",
    "Result",
    "State",
    "TerminableModel",
    "Trajectory",
    "ValidatableModel",
    "ValidationReport",
    "available",
    "get_model",
    "make_rng",
    "register",
    "run_experiment",
    "run_replicate",
    "spawn_rngs",
    "validate",
]
