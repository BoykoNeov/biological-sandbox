"""Visualization — pluggable backends built on the Recorder's trajectories.

A backend consumes :class:`~sandbox.core.recorder.Trajectory` objects (and
nothing model-specific), so the same plotting code serves every model. The
matplotlib backend is the default; a browser/shader backend can be added later
without touching the core (see HANDOFF.md §4 on the browser-vs-local fork).
"""

from __future__ import annotations
