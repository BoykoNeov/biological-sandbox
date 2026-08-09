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

**Two things Phase 2 had to add, both about the same failure mode.** A
fixed-``dt`` continuous model (Hodgkin-Huxley channel noise) is not an SSA, and
two of this module's Gillespie-shaped assumptions turn into *``Omega``-independent
floors* under it — the exact thing that flattens a slope while looking like
physics:

* *Step-hold sampling.* :func:`_sample_on_grid` holds the last event value, which
  is right for a piecewise-constant SSA trajectory and wrong for a smooth one:
  its error ``~ |dx/dt| dt_record`` does not shrink with ``Omega``. Rather than
  add an interpolation mode, callers may pass an explicit ``grid`` that is a
  *subset of the recorded times* — which a model recording at ``step_index * dt``
  can guarantee bit-for-bit — and set ``require_exact_grid`` so the assumption is
  **checked against every replicate's actual times** instead of trusted.
* *The reference check was one-sided.* Richardson halves the **ODE**'s ``dt``, but
  the stochastic side has its own step size and its own ``Omega``-independent
  discretization bias. ``stochastic_dt_key`` adds the symmetric check: re-run the
  largest ``Omega`` at half the model's own step and confirm ``D`` does not move.

``compare_keys`` is the third addition and is not about floors: a channel-state
model has no ``m``, ``h``, ``n`` to compare — only ``V`` — so the discrepancy must
be able to use a *subset* of the ODE's columns. It restricts the Richardson
maximum to the same columns, or a millivolt-scale quantity would be gated on a
dimensionless gate's error.

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
    idx = _grid_indices(times, grid)
    return values[idx]


def _grid_indices(times: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Index of the last recorded time at or before each grid time."""
    idx = np.searchsorted(times, grid, side="right") - 1
    return np.clip(idx, 0, times.size - 1)


def check_grid_is_exact(times: np.ndarray, grid: np.ndarray) -> None:
    """Raise unless every grid time is *exactly* a recorded time.

    The point of the check is that the step-hold in :func:`_sample_on_grid` then
    does nothing at all: it returns the recorded value rather than an older one,
    so the sampling contributes no error to ``D(Omega)`` — not "a small error",
    none. A model that writes ``state.t = step_index * dt`` (counted, never
    accumulated) makes this achievable bit-for-bit, because a caller building
    ``np.arange(0, n_steps + 1, j) * dt`` performs the identical multiplication.

    It is a *check*, not an assumption, because the failure is silent and its
    symptom is a flattened slope rather than an error: the step-hold's error is
    ``Omega``-independent, so it would sit under ``D(Omega)`` as a floor. One ulp
    of drift is enough — a grid time a single bit below a recorded time sends
    ``searchsorted`` back a whole step.
    """
    idx = _grid_indices(times, grid)
    sampled = times[idx]
    bad = np.flatnonzero(sampled != grid)
    if bad.size:
        first = int(bad[0])
        raise ValueError(
            f"{bad.size}/{grid.size} grid times are not exactly recorded times "
            f"(first: grid[{first}]={grid[first]!r} sampled as {sampled[first]!r}, "
            f"a gap of {grid[first] - sampled[first]:.3g}). The step-hold sampling "
            "would then contribute an Omega-independent error that floors D(Omega) "
            "and flattens the slope. Build the grid from the same arithmetic the "
            "model uses for state.t, e.g. np.arange(0, n_steps + 1, j) * dt."
        )


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
    # The stochastic-side mirror of Richardson: how far D at the largest Omega
    # moved when the *model's own* step was halved, and whether that shift is
    # small enough to rule out an Omega-independent discretization floor.
    # ``nan``/``True`` when the check was not requested (``stochastic_dt_key=None``).
    stochastic_delta: float
    stochastic_floor_ok: bool
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
        if np.isfinite(self.stochastic_delta):
            lines.append(
                f"  stochastic_floor_ok={self.stochastic_floor_ok} "
                f"(halving the model's own dt moved D by {self.stochastic_delta:.3g})"
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
    grid: Sequence[float] | None = None,
    require_exact_grid: bool = False,
    observable_keys: Sequence[str] | None = None,
    compare_keys: Sequence[str] | None = None,
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
    stochastic_dt_key: str | None = None,
    stochastic_floor_frac: float = 0.25,
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
        Ignored when ``grid`` is given.
    grid, require_exact_grid:
        An explicit array of sample times, and whether to *verify* that every one
        of them is exactly a recorded time in **every replicate**. Use both for a
        fixed-``dt`` continuous model, where the default step-hold sampling would
        otherwise contribute an ``Omega``-independent floor — see
        :func:`check_grid_is_exact`. Verification is per replicate against the
        real recorded times, not once against a constructed array, because a
        model that ends a run early would silently break the alignment.
    observable_keys:
        Ordered observable names whose column order matches the model's
        ``deterministic_rhs`` / ``initial_concentrations`` vector. Defaults to the
        model's ``observables()`` key order (which matches the ODE order for the
        models authored in this project); **pass it explicitly** for many-species
        models to avoid relying on dict-insertion order.
    compare_keys:
        Subset of ``observable_keys`` actually used for the discrepancy; defaults
        to all of them. Needed when a model cannot report every component of its
        own limit — a channel-state Hodgkin-Huxley has occupancy counts whose
        limits are ``m^3 h`` and ``n^4``, not ``m``, ``h``, ``n``, so only ``V``
        is comparable. It is also the right knob for scale: ``_per_replicate_``
        ``discrepancy`` averages ``|stoch - ode|`` with equal weight, so ``V``
        (~100 mV) would swamp dimensionless gates, and a mixed-scale L1 blend
        across millivolts and pure numbers is unprincipled rather than merely
        imprecise. The Richardson maximum is restricted to the same columns.
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
    stochastic_dt_key, stochastic_floor_frac:
        Name of the *model's own* numerical step in ``base_params``, enabling the
        stochastic-side mirror of the Richardson check: the largest ``Omega`` is
        re-run at half that step and ``D`` must not move. Only meaningful for a
        model that has such a step at all (a Gillespie SSA does not — it samples
        its own waiting time, so there is nothing to halve). The shift passes if
        it is either below ``stochastic_floor_frac`` of ``D``, **or** statistically
        indistinguishable from zero at ``z`` combined standard errors — the two
        runs use independent streams, so some of any observed shift is just
        replicate noise, and demanding a small *absolute* shift would turn a
        statistical fluctuation into a failure.

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
    sample_grid = (
        np.linspace(0.0, t_max, n_grid) if grid is None else np.asarray(list(grid), dtype=float)
    )

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

    compare = tuple(compare_keys) if compare_keys is not None else keys
    unknown = [key for key in compare if key not in keys]
    if unknown:
        raise ValueError(
            f"compare_keys {unknown} are not in observable_keys {list(keys)}; they name "
            "the ODE columns to compare, so every one must be a component of the limit"
        )
    columns = np.array([keys.index(key) for key in compare], dtype=int)

    ode_on_grid = _integrate_on_grid(rhs, c0, t_max, dt, sample_grid)[:, columns]
    # Richardson: halve dt; the reference must move much less than the smallest D.
    # Restricted to the compared columns for the same reason the discrepancy is:
    # a millivolt-scale check must not be gated on a dimensionless gate's error.
    ode_on_grid_fine = _integrate_on_grid(rhs, c0, t_max, dt / 2.0, sample_grid)[:, columns]
    richardson_delta = float(np.abs(ode_on_grid - ode_on_grid_fine).max())

    def _run(sweep_values: list[float], params: dict[str, Any], steps: int) -> Result:
        return run_experiment(
            Experiment(
                model=model_name,
                params=params,
                replicates=replicates,
                observables=compare,
                seed=seed,
                max_steps=steps,
                record_every=record_every,
                sweep={omega_key: sweep_values},
            ),
            params_factory,
        )

    def _discrepancies(result: Result, point: int, omega: float, steps: int) -> np.ndarray:
        trajectories = result.trajectories[point]
        truncated = sum(1 for traj in trajectories if not traj.terminated)
        if truncated:
            raise ValueError(
                f"{truncated}/{len(trajectories)} replicates at Omega={omega:g} hit "
                f"max_steps (={steps}) without reaching t_max (={t_max}); the time-average "
                "would cover a truncated horizon and bias D(Omega). Increase max_steps."
            )
        out = np.empty(len(trajectories), dtype=float)
        for r, traj in enumerate(trajectories):
            times, series = traj.as_arrays()
            if require_exact_grid:
                check_grid_is_exact(times, sample_grid)
            out[r] = _per_replicate_discrepancy(times, series, sample_grid, ode_on_grid, compare)
        return out

    result = _run([float(om) for om in omegas_arr], {**base_params, t_max_key: t_max}, max_steps)

    k = omegas_arr.size
    per_replicate = np.empty((k, replicates), dtype=float)
    for p in range(k):
        per_replicate[p] = _discrepancies(result, p, float(omegas_arr[p]), max_steps)

    discrepancy = per_replicate.mean(axis=1)
    discrepancy_sem = per_replicate.std(axis=1, ddof=1) / np.sqrt(replicates)

    # The stochastic-side mirror of Richardson, at the largest Omega: that is where
    # D is smallest, so an Omega-independent discretization bias bites there first.
    stochastic_delta, stochastic_floor_ok = float("nan"), True
    if stochastic_dt_key is not None:
        coarse_dt = float(base_params[stochastic_dt_key])
        fine = _run(
            [float(omegas_arr[-1])],
            {**base_params, t_max_key: t_max, stochastic_dt_key: coarse_dt / 2.0},
            2 * max_steps,
        )
        fine_per_replicate = _discrepancies(fine, 0, float(omegas_arr[-1]), 2 * max_steps)
        fine_mean = float(fine_per_replicate.mean())
        fine_sem = float(fine_per_replicate.std(ddof=1) / np.sqrt(replicates))
        stochastic_delta = abs(fine_mean - float(discrepancy[-1]))
        stochastic_floor_ok = stochastic_delta <= max(
            stochastic_floor_frac * float(discrepancy[-1]),
            z * float(np.hypot(fine_sem, discrepancy_sem[-1])),
        )

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
    passed = consistent and significant and reference_ok and stochastic_floor_ok

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
        stochastic_delta=stochastic_delta,
        stochastic_floor_ok=stochastic_floor_ok,
        monotone=monotone,
        consistent=consistent,
        significant=significant,
        passed=passed,
    )
