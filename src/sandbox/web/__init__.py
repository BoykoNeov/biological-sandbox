"""The browser front-end's Python side.

One module of substance — :mod:`sandbox.web.bridge` — plus a colormap. It is a
shared service in exactly the sense non-negotiable #1 means: it consumes only
the protocol surface (``initial_state`` / ``step`` / ``observables`` /
``is_terminal`` / ``fields`` / ``analytic_predictions`` / ``deterministic_rhs``)
and never reaches inside a concrete state. That is the same relationship the
Recorder has, and the reason a front-end can be added without touching a model.

**No numerics live in JavaScript.** The JS side owns the DOM, the canvas and the
message loop; everything numerical happens here, in the same code the native
test suite validates. See ``docs/plans/phase4-plan.md`` on why a second
implementation of a validated model — in JS or in a shader — is the cost this
branch exists to avoid.
"""

from __future__ import annotations
