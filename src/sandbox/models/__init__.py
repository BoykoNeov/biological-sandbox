"""Models. Importing this package registers every built-in model by name.

The validated core lives here directly. The speculative arc, when it exists,
lives quarantined under ``models/ecosystem/`` — a sibling, never entangled with
the validated models. That directory boundary is the structural expression of
the verifiable/exploratory split.
"""

from __future__ import annotations

from sandbox.models import wright_fisher  # noqa: F401  (import triggers registration)
from sandbox.models.wright_fisher import WFParams, WFState, WrightFisher

__all__ = ["WFParams", "WFState", "WrightFisher"]
