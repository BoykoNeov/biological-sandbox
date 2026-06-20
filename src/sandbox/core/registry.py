"""A tiny name -> model registry.

An ``Experiment`` refers to its model by string so that experiments are
serializable. The registry resolves that string to a stateless model instance.
"""

from __future__ import annotations

from sandbox.core.protocol import Model

_REGISTRY: dict[str, Model] = {}


def register(name: str, model: Model) -> Model:
    """Register a model instance under ``name``. Returns the model for convenience."""
    if name in _REGISTRY:
        raise ValueError(f"model already registered: {name!r}")
    _REGISTRY[name] = model
    return model


def get_model(name: str) -> Model:
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"unknown model {name!r}; registered: {known}") from None


def available() -> list[str]:
    return sorted(_REGISTRY)
