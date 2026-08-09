"""Matplotlib backend.

Kept import-guarded: matplotlib is an optional extra (``pip install
biological-sandbox[viz]``), so the core and its tests never depend on it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

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

    **Units line up with no rescaling.** ``observables()`` returns
    *concentrations* ``x = n/Omega`` (the project-wide convention, see
    ``core/protocol.py``) and ``deterministic_rhs`` is integrated in that same
    concentration space, so ``deterministic`` is passed through as-is. Plotting
    raw counts against the ODE would be wrong by a factor of ``Omega``.

    ``deterministic`` is a **single** ``(times, values)`` pair, so a multi-species
    ODE solution ``y`` of shape ``(n_t, S)`` must be sliced to the column matching
    ``observable`` — derive that index from the model's ordered observable keys
    (e.g. ``OBSERVABLE_KEYS.index(observable)``), never hardcode it. On a
    symmetric limit cycle the wrong column is the right *shape* at the wrong
    *phase*, which looks plausible in the figure.

    An ``observable`` absent from a trajectory is skipped silently (replicates
    may record different keys), so a typo yields a figure with only the
    deterministic line drawn.
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


# Artist gids are part of this helper's contract: callers (and its tests) locate the
# fit, the reference law and the two point sets by gid rather than by index into
# ax.lines (errorbar adds several artists, so positional indexing is brittle) or by
# legend text (which is user-facing prose and may be reworded).
FIT_GID = "fit"
GUIDE_GID = "guide"
FITTED_POINTS_GID = "points-fitted"
EXCLUDED_POINTS_GID = "points-excluded"


def plot_convergence(
    omegas: Sequence[float],
    discrepancy: Sequence[float],
    *,
    sem: Sequence[float] | None = None,
    fit_mask: Sequence[bool] | None = None,
    slope: float | None = None,
    slope_se: float | None = None,
    expected_slope: float = -0.5,
    title: str | None = None,
    ax: Any = None,
) -> Any:
    """Log-log plot of the system-size discrepancy ``D(Omega)`` against the ``-1/2`` law.

    The companion figure to :func:`plot_replicates`: where that one shows the
    stochastic cloud tightening around the deterministic limit, this one *measures*
    how fast it tightens. On log-log axes the Kurtz/van-Kampen prediction
    ``D ~ Omega^{-1/2}`` is a straight line of slope ``-1/2``, so the check is
    visual — does the data run parallel to the guide line?

    Takes plain arrays (not a
    :class:`~sandbox.core.convergence.ConvergenceReport`) so it can be exercised on
    synthetic data without running an SSA sweep; a caller with a report unpacks it
    in one call.

    Parameters
    ----------
    omegas, discrepancy:
        The swept system sizes and ``D(Omega)`` (``report.omegas`` /
        ``report.discrepancy``). Both must be strictly positive — log-log axes.
    sem:
        Optional standard error per point (``report.discrepancy_sem``), drawn as
        error bars.
    fit_mask:
        Which points entered the log-log fit (``report.fit_mask``). Excluded points
        are still **drawn**, as hollow markers: the low-``Omega`` phase-saturation
        knee is evidence about where the law stops applying, and silently dropping
        it from the figure would hide exactly what the mask is documenting.
    slope, slope_se:
        The measured fit (``report.slope`` / ``report.slope_se``) shown in the
        legend. If ``slope`` is omitted it is refitted by OLS over the masked
        points, so the helper is usable on bare arrays.
    expected_slope:
        The predicted exponent, drawn as a dashed guide line.

    Both the fit line and the guide line are anchored at the **centroid of the
    fitted points** in log-log space (i.e. the geometric mean of the masked
    ``Omega`` and ``D``). Anchoring at the first point instead would pin the guide
    to the saturated knee and make a passing check look like a failing one.
    """
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    om = np.asarray(list(omegas), dtype=float)
    d = np.asarray(list(discrepancy), dtype=float)
    if om.size != d.size:
        raise ValueError(f"omegas and discrepancy differ in length: {om.size} vs {d.size}")
    if np.any(om <= 0.0) or np.any(d <= 0.0):
        raise ValueError("omegas and discrepancy must be strictly positive for a log-log plot")

    mask = np.ones(om.size, dtype=bool) if fit_mask is None else np.asarray(list(fit_mask), bool)
    if not mask.any():
        raise ValueError("fit_mask excludes every point; nothing to fit")

    err = None
    if sem is not None:
        s = np.asarray(list(sem), dtype=float)
        # Keep the lower whisker inside the positive half-plane: a symmetric bar
        # wider than the value itself has no place on a log axis.
        err = np.vstack([np.minimum(s, 0.9 * d), s])

    def _points(sel: np.ndarray, label: str, gid: str, filled: bool) -> None:
        if not sel.any():
            return
        style = {"color": "C0"} if filled else {"markerfacecolor": "none", "color": "C7"}
        container = ax.errorbar(
            om[sel],
            d[sel],
            yerr=None if err is None else err[:, sel],
            fmt="o",
            markersize=6,
            capsize=3,
            linestyle="none",
            label=label,
            **style,
        )
        # errorbar puts the label on the container, so the data line itself carries
        # no identifying label — tag it directly.
        container.lines[0].set_gid(gid)

    _points(mask, "D(Omega), fitted", FITTED_POINTS_GID, filled=True)
    _points(~mask, "D(Omega), excluded from fit", EXCLUDED_POINTS_GID, filled=False)

    log_om, log_d = np.log(om[mask]), np.log(d[mask])
    if slope is None:
        slope = float(np.polyfit(log_om, log_d, 1)[0]) if log_om.size >= 2 else float("nan")
    # Centroid anchoring: both lines pass through the geometric mean of the fitted
    # points, so the guide is a fair visual comparison for the fit.
    cx, cy = float(log_om.mean()), float(log_d.mean())

    def _line(exponent: float, span: np.ndarray, label: str, gid: str, **style: Any) -> None:
        x = np.array([span.min(), span.max()], dtype=float)
        # gid is the stable handle (legend text is user-facing and may be reworded).
        ax.plot(x, np.exp(cy + exponent * (np.log(x) - cx)), label=label, gid=gid, **style)

    fit_label = f"fit: slope = {slope:.3f}"
    if slope_se is not None:
        fit_label += f" +/- {slope_se:.3f}"
    _line(slope, om[mask], fit_label, FIT_GID, color="C0", linewidth=2.0)
    _line(
        expected_slope,
        om,
        f"predicted: slope = {expected_slope:+.2f}",
        GUIDE_GID,
        color="C3",
        linewidth=1.5,
        linestyle="--",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("system size Omega")
    ax.set_ylabel("discrepancy D(Omega)")
    ax.legend(loc="best", fontsize=9)
    if title:
        ax.set_title(title)
    return ax
