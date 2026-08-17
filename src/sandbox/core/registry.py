"""A tiny name -> model registry.

An ``Experiment`` refers to its model by string so that experiments are
serializable. The registry resolves that string to a stateless model instance.

**And to a params factory, which is the half that was missing.** Every service
here takes a ``params_factory`` from its *caller* — ``lambda d: WFParams(**d)``
— so that the core stays agnostic about params shapes. That works when a human
writes the call. It does not work for a front-end handed a JSON ``Experiment``
and nothing else: there is no caller to supply the factory, and an experiment
that names its model by string but needs an out-of-band Python callable to be
runnable is only half serializable. Registering the params type closes that,
and the callers that already pass their own factory are unaffected.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sandbox.core.protocol import Model

_REGISTRY: dict[str, Model] = {}
_PARAMS_TYPES: dict[str, type] = {}


def register(name: str, model: Model, params_type: type | None = None) -> Model:
    """Register a model instance under ``name``. Returns the model for convenience.

    ``params_type`` is the model's params dataclass. It is optional because
    test-only models registered inside the suite (broken-model "teeth") reuse a
    real model's params and have no interest in JSON round-tripping; omitting it
    costs only :func:`get_params_factory`, which raises a specific error rather
    than a confusing one.
    """
    if name in _REGISTRY:
        raise ValueError(f"model already registered: {name!r}")
    _REGISTRY[name] = model
    if params_type is not None:
        _PARAMS_TYPES[name] = params_type
    return model


def get_model(name: str) -> Model:
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"unknown model {name!r}; registered: {known}") from None


def get_params_type(name: str) -> type:
    """The params dataclass registered for ``name``."""
    get_model(name)  # raises the informative "unknown model" error first
    try:
        return _PARAMS_TYPES[name]
    except KeyError:
        raise KeyError(
            f"model {name!r} is registered without a params type, so its params "
            "cannot be built from a plain dict; pass params_type= to register(), "
            "or supply a params_factory at the call site"
        ) from None


def get_params_factory(name: str) -> Callable[[dict[str, Any]], Any]:
    """A ``dict -> params`` factory for ``name``, for callers holding only JSON.

    The factory is ``params_type(**d)`` and nothing more: every params dataclass
    in this project takes plain JSON-representable values (floats, ints, strings,
    and tuples of them), and coerces sequences in ``__post_init__``. Anything
    needing more than that should not be silently smoothed over here.
    """
    params_type = get_params_type(name)
    return lambda d: params_type(**d)


def available() -> list[str]:
    return sorted(_REGISTRY)
