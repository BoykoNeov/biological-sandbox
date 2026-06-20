"""Matplotlib backend.

Kept import-guarded: matplotlib is an optional extra (``pip install
biological-sandbox[viz]``), so the core and its tests never depend on it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sandbox.core.recorder import Trajectory


def _require_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional dep
        raise ModuleNotFoundError(
            "matplotlib is required for visualization. Install with: "
            "uv sync --extra viz   (or: pip install 'biological-sandbox[viz]')"
        ) from exc
    return plt


def plot_replicates(
    trajectories: Sequence[Trajectory],
    observable: str,
    *,
    deterministic: tuple[Sequence[float], Sequence[float]] | None = None,
    title: str | None = None,
    ax: Any = None,
    alpha: float = 0.15,
) -> Any:
    """Overlay an observable across stochastic replicates, with an optional
    deterministic limit drawn on top.

    This is the project's central teaching figure: many faint stochastic
    trajectories, the deterministic limit as a bold line, and (as the system
    grows) the cloud tightening around that line. Returns the matplotlib Axes.
    """
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    for traj in trajectories:
        times, series = traj.as_arrays()
        if observable in series:
            ax.plot(times, series[observable], color="C0", alpha=alpha, linewidth=0.8)

    if deterministic is not None:
        det_t, det_y = deterministic
        ax.plot(det_t, det_y, color="C3", linewidth=2.5, label="deterministic limit")
        ax.legend()

    ax.set_xlabel("time")
    ax.set_ylabel(observable)
    if title:
        ax.set_title(title)
    return ax
