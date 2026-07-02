"""Convergence pathway — the project's *second* validation track.

Phase 0 and the exact-closed-form Phase-1 models (``birth_death``,
``isomerization``) are validated by :func:`sandbox.core.validation.validate`: the
replicate mean of a *final scalar* observable matches a closed form. But the
headline Phase-1 model — the repressilator — has **no scalar closed form**, and
its deterministic limit is a *limit cycle*, not a fixed point. It cannot be
validated by matching an ensemble mean to a number. This module is the rigorous
alternative.

**Kurtz's law of large numbers for density-dependent jump processes.** As the
system size ``Omega`` (reaction volume; molecule counts scale with it) grows, a
**single** scaled SSA trajectory ``X_Omega(t)/Omega`` converges *uniformly on any
finite horizon* ``[0, T]`` to the deterministic ODE solution ``x(t)``. The
fluctuations around it are ``O(Omega^{-1/2})`` (van Kampen system-size expansion
/ functional CLT). So the checkable, quantitative claim is:

    D(Omega) = mean over replicates of  < |X_Omega(t)/Omega - x(t)| >_t
    scales as  Omega^{-1/2}   -->   log-log slope of D vs Omega  ~=  -1/2.

**The one trap this module exists to avoid.** The discrepancy is the *mean of the
per-replicate errors*, **never** the error of the ensemble-averaged trajectory.
Independent replicates of an oscillator undergo *phase diffusion* — they drift out
of sync, so their ensemble mean damps toward the time-average of the cycle while
every individual replicate keeps oscillating at full amplitude. Matching that
damped mean to the ODE would compare two different things and produce a green
check that proves nothing. The per-replicate discrepancy is factored into the
pure helper :func:`_per_replicate_discrepancy` precisely so it can be unit-tested
in isolation — the machinery's core correctness must not rest on the fuzzier
slope check alone.

**Where the -1/2 slope flattens (so we fit only the clean middle regime).**

* *Low-Omega knee (phase saturation).* Phase-diffusion variance grows ``~ T/Omega``;
  if ``T`` is not ``<< Omega`` replicates fully dephase and ``D(Omega)`` saturates
  at ``O(amplitude)`` — flat, not ``Omega^{-1/2}``. Keep ``T`` to ~1-2 periods and
  ``Omega`` large enough that ``T << Omega`` across the sweep; pass a ``fit_mask``
  to exclude any saturated low-``Omega`` points.
* *High-Omega floor (reference error).* The RK4 ODE is the *reference*; if its
  integration error is not ``<< D(Omega)`` at the largest ``Omega`` the discrepancy
  floors and the slope flattens at the top. We Richardson-check the reference
  (halve ``dt``, confirm the ODE moves ``<<`` the smallest ``D``) and fold that
  into pass/fail as ``reference_ok``.

**Pass/fail is a slope confidence interval, never a hardcoded epsilon** — matching
the ValidationSuite's philosophy. The slope standard error is the *larger* of a
bootstrap-over-replicates estimate and the ordinary-least-squares fit SE (the
bootstrap resamples replicate noise but not the small-``K`` fit uncertainty, so
with only 4-6 ``Omega`` points it can be deceptively tight; the OLS SE catches
that). We assert the slope is *consistent with* ``-1/2`` (``|slope + 1/2| <=
z * SE``) **and** *significantly negative* (``slope + z * SE < 0``, so a flat or
positive slope fails — the teeth). Monotonicity of ``D(Omega)`` is kept only as a
soft printed diagnostic (replicate noise makes adjacent points cross by chance).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from sandbox.core.ode import integrate_rk4
from sandbox.core.protocol import DeterministicLimitModel, Experiment, Result
from sandbox.core.registry import get_model
from sandbox.core.sweep import run_experiment


def _sample_on_grid(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Step-interpolate an SSA sample series onto ``grid`` (hold last event value).

    A Gillespie trajectory is piecewise-constant: a count holds until the next
    event fires, so the physically faithful interpolation is "value of the most
    recent event at or before ``t``", not linear. For each grid time we take the
    last recorded index with ``times[i] <= t`` via ``searchsorted(side='right')``.

    ``times`` is the event-time sequence (monotonic, ``times[0] == 0``); ``grid``
    lies within ``[0, times[-1]]`` (the caller checks the horizon was reached), so
    the clip to a valid index only guards floating-point dust at the endpoints.
    """
    idx = np.searchsorted(times, grid, side="right") - 1
    idx = np.clip(idx, 0, times.size - 1)
    return values[idx]


def _per_replicate_discrepancy(
    times: np.ndarray,
    series: dict[str, np.ndarray],
    grid: np.ndarray,
    ode_on_grid: np.ndarray,
    keys: Sequence[str],
) -> float:
    """Time- and species-averaged ``|X_Omega(t)/Omega - x(t)|`` for ONE replicate.

    This is the whole pathway's load-bearing computation, kept pure and tiny so it
    can be unit-tested against hand-worked trajectories. It operates on a *single*
    replicate's recorded ``(times, series)``; the caller averages the returned
    scalars **over replicates**. Averaging here — before the caller's replicate
    mean — is the correct order; averaging the trajectories first and then taking
    one discrepancy is the phase-diffusion trap this module is built to avoid.

    ``ode_on_grid`` has shape ``(n_grid, S)`` with column ``s`` corresponding to
    ``keys[s]``; the stochastic series are pulled in the same key order so the two
    line up column-for-column. Returns the mean of ``|stochastic - ode|`` over both
    time (grid points) and species (columns) — an L1 discrepancy.
    """
    stoch = np.stack([_sample_on_grid(times, series[k], grid) for k in keys], axis=1)
    return float(np.abs(stoch - ode_on_grid).mean())


def _fit_slope(log_omega: np.ndarray, log_d: np.ndarray) -> tuple[float, float]:
    """Ordinary-least-squares log-log slope and its standard error.

    Returns ``(slope, slope_se)`` where ``slope_se`` comes from the fit covariance
    (residual scatter around the line). With only a handful of ``Omega`` points
    this SE is noisy but it is the honest small-sample fit uncertainty, which the
    bootstrap-over-replicates alone does not capture.
    """
    n = log_omega.size
    if n < 3:
        # polyfit's covariance needs n > order + 2 = 3; below that report no SE.
        slope = float(np.polyfit(log_omega, log_d, 1)[0]) if n >= 2 else float("nan")
        return slope, float("inf")
    coeffs, cov = np.polyfit(log_omega, log_d, 1, cov=True)
    return float(coeffs[0]), float(np.sqrt(cov[0, 0]))


def _bootstrap_slope_se(
    per_replicate: np.ndarray,
    log_omega: np.ndarray,
    mask: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> float:
    """Slope SE from resampling replicates within each ``Omega`` point.

    ``per_replicate`` has shape ``(K, R)``. For each bootstrap iteration we resample
    the ``R`` replicate discrepancies (with replacement) *independently per Omega*
    — they are independent SSA runs — recompute ``D(Omega)``, refit the masked
    log-log slope, and collect it. The SE is the std of those slopes. A dedicated
    seeded generator keeps this reproducible (a single auxiliary resampling stream,
    not per-replicate simulation seeds, so ``default_rng(seed)`` is appropriate).
    """
    rng = np.random.default_rng(seed)
    k, r = per_replicate.shape
    log_om = log_omega[mask]
    slopes = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        # (K, R) resample indices — independent columns drawn per row.
        idx = rng.integers(0, r, size=(k, r))
        resampled = np.take_along_axis(per_replicate, idx, axis=1)
        d_b = resampled.mean(axis=1)[mask]
        slopes[b] = np.polyfit(log_om, np.log(d_b), 1)[0]
    return float(slopes.std(ddof=1))


@dataclass(frozen=True)
class ConvergenceReport:
    """Result of a system-size convergence check (the ``Omega^{-1/2}`` scaling law).

    ``omegas`` / ``discrepancy`` are the swept system sizes and their per-``Omega``
    discrepancies ``D(Omega)`` (mean over replicates). ``per_replicate`` (shape
    ``(K, R)``) is exposed so callers can verify the mean is taken *per replicate*
    (``discrepancy == per_replicate.mean(axis=1)``), and for their own resampling.
    ``slope`` / ``slope_se`` / ``slope_ci`` describe the log-log fit on the
    ``fit_mask`` subset; ``passed`` folds in slope-consistency-with ``-1/2``,
    slope-significantly-negative, and ``reference_ok`` (the Richardson check).
    """

    omegas: np.ndarray
    discrepancy: np.ndarray
    discrepancy_sem: np.ndarray
    per_replicate: np.ndarray
    fit_mask: np.ndarray
    slope: float
    slope_se: float
    slope_ci: tuple[float, float]
    expected_slope: float
    z: float
    richardson_delta: float
    reference_ok: bool
    monotone: bool
    consistent: bool
    significant: bool
    passed: bool

    def __str__(self) -> str:  # ASCII only (Windows cp1252 console)
        head = f"Convergence check: {'PASS' if self.passed else 'FAIL'}"
        lines = [head]
        for om, d, sem, used in zip(
            self.omegas, self.discrepancy, self.discrepancy_sem, self.fit_mask, strict=True
        ):
            flag = "fit" if used else "   "
            lines.append(f"  [{flag}] Omega={om:>10.4g}  D(Omega)={d:.5g} +/- {sem:.2g}")
        lines.append(
            f"  slope = {self.slope:.4f} +/- {self.slope_se:.4f} "
            f"(expected {self.expected_slope:+.2f}); "
            f"CI[{self.z:g} SE] = [{self.slope_ci[0]:.3f}, {self.slope_ci[1]:.3f}]"
        )
        lines.append(
            f"  consistent_with_{self.expected_slope:+.2f}={self.consistent}  "
            f"significantly_negative={self.significant}  "
            f"reference_ok={self.reference_ok} (Richardson delta={self.richardson_delta:.3g})  "
            f"monotone={self.monotone}"
        )
        return "\n".join(lines)


def _integrate_on_grid(
    rhs: Callable[[np.ndarray], np.ndarray],
    c0: np.ndarray,
    t_max: float,
    dt: float,
    grid: np.ndarray,
) -> np.ndarray:
    """Integrate the ODE at step ``dt`` and interpolate each species onto ``grid``.

    Linear interpolation is used here (unlike the SSA step-interpolation): the ODE
    is smooth, so ``np.interp`` on a dense RK4 trajectory has negligible error.
    Returns shape ``(n_grid, S)``.
    """
    t_ode, y_ode = integrate_rk4(rhs, c0, t_max, dt)
    return np.stack([np.interp(grid, t_ode, y_ode[:, s]) for s in range(y_ode.shape[1])], axis=1)


def convergence_report(
    model_name: str,
    base_params: dict[str, Any],
    params_factory: Callable[[dict[str, Any]], Any],
    *,
    omegas: Sequence[float],
    t_max: float,
    dt: float,
    replicates: int,
    n_grid: int = 200,
    observable_keys: Sequence[str] | None = None,
    omega_key: str = "Omega",
    t_max_key: str = "t_max",
    seed: int = 0,
    z: float = 2.0,
    n_bootstrap: int = 1000,
    fit_mask: Sequence[bool] | None = None,
    max_steps: int = 1_000_000,
    record_every: int = 1,
    expected_slope: float = -0.5,
    reference_floor_frac: float = 0.1,
) -> ConvergenceReport:
    """Check that ``D(Omega)`` scales as ``Omega^{-1/2}`` for a deterministic-limit model.

    Runs seeded SSA replicates at each system size in ``omegas`` (horizon ``t_max``),
    compares each *single* scaled trajectory to the ODE limit, and fits the log-log
    slope of the per-``Omega`` discrepancy — asserting it is consistent with
    ``expected_slope`` (``-1/2``) and significantly negative.

    Parameters
    ----------
    model_name, base_params, params_factory:
        As in :func:`~sandbox.core.validation.validate`. ``base_params`` is the
        plain-number param dict; ``omega_key`` and ``t_max_key`` are overridden per
        run (the Omega sweep and the horizon), so any values for them in
        ``base_params`` are ignored.
    omegas, t_max, dt, replicates:
        System sizes to sweep, the finite horizon ``T``, the ODE integration step,
        and replicates per size. ``dt`` must be small enough that the Richardson
        check passes (its error ``<<`` the smallest ``D``).
    n_grid:
        Number of uniform sample times on ``[0, t_max]`` for the time-average.
    observable_keys:
        Ordered observable names whose column order matches the model's
        ``deterministic_rhs`` / ``initial_concentrations`` vector. Defaults to the
        model's ``observables()`` key order (which matches the ODE order for the
        models authored in this project); **pass it explicitly** for many-species
        models to avoid relying on dict-insertion order.
    z:
        Confidence multiplier for the slope tolerance (``tolerance = z * slope_se``).
    fit_mask:
        Optional boolean mask (length ``len(omegas)``) selecting which points enter
        the log-log fit — exclude a saturated low-``Omega`` knee or a floored
        high-``Omega`` point. Defaults to all points.
    record_every:
        Event sub-sampling for the recorder. **Keep at 1 for a slope check** —
        sub-sampling improves interpolation fidelity *with* ``Omega`` (more events
        per unit time at large ``Omega``), biasing the slope more negative. Raise it
        only for viz, where the trajectory shape, not the slope, is what matters.

    Raises
    ------
    TypeError
        If the model does not implement ``DeterministicLimitModel``.
    ValueError
        If any replicate failed to reach ``t_max`` within ``max_steps`` (its
        time-average would cover a short horizon and bias ``D``), mirroring the
        anti-bias guard in :func:`~sandbox.core.validation.validate`.
    """
    model = get_model(model_name)
    if not isinstance(model, DeterministicLimitModel):
        raise TypeError(
            f"model {model_name!r} does not implement DeterministicLimitModel "
            "(deterministic_rhs / initial_concentrations); it has no ODE limit to "
            "converge to and cannot be checked by this pathway"
        )

    omegas_arr = np.asarray(list(omegas), dtype=float)
    grid = np.linspace(0.0, t_max, n_grid)

    # The ODE is Omega-independent (concentration space), so a single integration
    # serves every system size. Build a representative params object for it.
    params_ode = params_factory({**base_params, omega_key: float(omegas_arr[0]), t_max_key: t_max})
    rhs = model.deterministic_rhs(params_ode)
    c0 = np.asarray(model.initial_concentrations(params_ode), dtype=float)

    if observable_keys is None:
        init = model.initial_state(params_ode, np.random.default_rng(0))
        observable_keys = tuple(model.observables(init).keys())
    keys = tuple(observable_keys)
    if len(keys) != c0.size:
        raise ValueError(
            f"observable_keys has {len(keys)} entries but the ODE vector has {c0.size} "
            "components; they must correspond one-to-one and in order"
        )

    ode_on_grid = _integrate_on_grid(rhs, c0, t_max, dt, grid)
    # Richardson: halve dt; the reference must move much less than the smallest D.
    ode_on_grid_fine = _integrate_on_grid(rhs, c0, t_max, dt / 2.0, grid)
    richardson_delta = float(np.abs(ode_on_grid - ode_on_grid_fine).max())

    experiment = Experiment(
        model=model_name,
        params={**base_params, t_max_key: t_max},
        replicates=replicates,
        observables=keys,
        seed=seed,
        max_steps=max_steps,
        record_every=record_every,
        sweep={omega_key: [float(om) for om in omegas_arr]},
    )
    result: Result = run_experiment(experiment, params_factory)

    k = omegas_arr.size
    per_replicate = np.empty((k, replicates), dtype=float)
    for p in range(k):
        trajectories = result.trajectories[p]
        truncated = sum(1 for traj in trajectories if not traj.terminated)
        if truncated:
            raise ValueError(
                f"{truncated}/{len(trajectories)} replicates at Omega={omegas_arr[p]:g} hit "
                f"max_steps (={max_steps}) without reaching t_max (={t_max}); the time-average "
                "would cover a truncated horizon and bias D(Omega). Increase max_steps."
            )
        for r, traj in enumerate(trajectories):
            times, series = traj.as_arrays()
            per_replicate[p, r] = _per_replicate_discrepancy(times, series, grid, ode_on_grid, keys)

    discrepancy = per_replicate.mean(axis=1)
    discrepancy_sem = per_replicate.std(axis=1, ddof=1) / np.sqrt(replicates)

    mask = np.ones(k, dtype=bool) if fit_mask is None else np.asarray(list(fit_mask), dtype=bool)
    log_omega = np.log(omegas_arr)
    slope, slope_se_ols = _fit_slope(log_omega[mask], np.log(discrepancy[mask]))
    slope_se_boot = _bootstrap_slope_se(per_replicate, log_omega, mask, n_bootstrap, seed)
    # The larger SE is the conservative one: bootstrap misses small-K fit scatter,
    # OLS misses replicate resampling noise; taking the max guards both blind spots.
    slope_se = max(slope_se_ols, slope_se_boot)
    slope_ci = (slope - z * slope_se, slope + z * slope_se)

    reference_ok = richardson_delta < reference_floor_frac * float(discrepancy.min())
    monotone = bool(np.all(np.diff(discrepancy[mask]) < 0.0))
    consistent = abs(slope - expected_slope) <= z * slope_se
    significant = slope + z * slope_se < 0.0
    passed = consistent and significant and reference_ok

    return ConvergenceReport(
        omegas=omegas_arr,
        discrepancy=discrepancy,
        discrepancy_sem=discrepancy_sem,
        per_replicate=per_replicate,
        fit_mask=mask,
        slope=slope,
        slope_se=slope_se,
        slope_ci=slope_ci,
        expected_slope=expected_slope,
        z=z,
        richardson_delta=richardson_delta,
        reference_ok=reference_ok,
        monotone=monotone,
        consistent=consistent,
        significant=significant,
        passed=passed,
    )
