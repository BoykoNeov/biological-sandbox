"""The third validation track: which *pattern* a run selects, not how fast it grows.

``core.validation`` asserts that a replicate mean matches a predicted scalar within
``z * SE``. ``core.convergence`` asserts that a discrepancy scales as a power law.
Neither can express the claim this module exists for, and the reason is measured
rather than stylistic.

**Why a mean-vs-prediction check is the wrong instrument here.** A periodic box
admits only wavenumbers ``q = 2 pi j / L`` with integer ``j``, so the wavelength a
run selects is *quantized*. The ensemble mean over replicates is therefore a mean of
integers, which has no reason to equal any continuous prediction exactly — and
measurement confirms it does not: at one resolution the deviation from the
continuous maximiser **grows** with replicate count (1.08, 0.98, 1.81, 2.75 SE at
R = 8, 16, 32, 48) because the mean holds still while the error bar shrinks under
it, and at another resolution the better target is the nearest *integer* instead.
Asserting ``|mean - predicted| <= z * SE`` there would pass only because ``R`` is
small. See ``docs/plans/phase2c-schnakenberg-measurement.md`` §7.

**What is assertable instead is a discrimination.** The prediction competes against
named alternatives — the same calculation done with the continuum operator instead
of the stencil, the centre of the unstable band, its edges — and the claim is that
the measurement is closer to the prediction than to any of them **by a margin
exceeding ``z * SE``**. That is a statistical tolerance, derived from the spread of
the runs rather than typed, and it is robust to a small unexplained offset in a way
that an equality claim is not.

Two non-vacuity guards come with it, because a discrimination between hypotheses is
worthless if the hypotheses were never in doubt:

* **Every replicate must land inside the linearly unstable band.** Outside it,
  nothing can grow, so a "selected" mode there would mean the measurement is not
  measuring selection.
* **The selected mode must not track the initial condition.** If the winner is
  whichever mode the random start happened to load most, the prediction is
  irrelevant and the agreement is a coincidence. The report compares the spread of
  the *initial* dominant modes with the spread of the *final* ones; measured, the
  initial modes ranged over the whole spectrum while every run ended within one mode
  of the same answer.

This module consumes only the protocol surface, as ``Recorder`` and ``sweep`` do: it
reads two named scalar observables and takes the prediction, the competitors and the
band from its **caller**. It knows nothing about dispersion relations, stencils or
any model's concrete state — the model-specific arithmetic stays in the model.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from sandbox.core.protocol import Experiment, Result
from sandbox.core.sweep import run_experiment


@dataclass(frozen=True)
class HypothesisCheck:
    """One competing prediction, and whether the measurement excludes it."""

    name: str
    mode: float
    gap: float  # |mean - this hypothesis|
    margin_se: float  # (gap - gap_to_prediction) / sem
    passed: bool

    def __str__(self) -> str:  # ASCII only (Windows cp1252 console)
        verdict = "EXCLUDED" if self.passed else "not excluded"
        return (
            f"[{verdict:>12}] {self.name}: predicts {self.mode:.4f}, "
            f"|measured - it| = {self.gap:.4f}, margin {self.margin_se:+.2f} SE"
        )


@dataclass(frozen=True)
class SelectionReport:
    """The measured selected mode, and every hypothesis it was tested against."""

    model: str
    predicted_mode: float
    measured_mean: float
    measured_sd: float
    sem: float
    n: int
    z: float
    band: tuple[int, int]
    selected_modes: tuple[float, ...]
    initial_modes: tuple[float, ...]
    efolds: float | None
    checks: tuple[HypothesisCheck, ...] = ()
    all_inside_band: bool = True
    initial_spread: float = 0.0
    selected_spread: float = 0.0
    initials_outside_band: int = 0

    @property
    def gap_to_prediction(self) -> float:
        return abs(self.measured_mean - self.predicted_mode)

    @property
    def gap_in_se(self) -> float:
        """How far the prediction sits from the measurement, in standard errors.

        Reported and **not** asserted: this is the quantity that fails as ``R``
        grows, which is why this module asserts margins instead. Read it as a
        description of how small the unexplained offset is, not as a pass mark.
        """
        return self.gap_to_prediction / self.sem if self.sem > 0.0 else float("inf")

    @property
    def spread_ratio(self) -> float:
        """Spread of the selected modes over spread of the initially-loaded ones.

        Small means the growth rate decided the outcome; near 1 would mean the
        initial condition did.
        """
        return (
            self.selected_spread / self.initial_spread
            if self.initial_spread > 0.0
            else float("nan")
        )

    @property
    def passed(self) -> bool:
        return self.all_inside_band and all(c.passed for c in self.checks)

    def __str__(self) -> str:  # ASCII only (Windows cp1252 console)
        lines = [
            f"selection report for {self.model!r}: "
            f"{'PASS' if self.passed else 'FAIL'} (z = {self.z:g})",
            f"  predicted mode      {self.predicted_mode:.4f}",
            f"  measured mean       {self.measured_mean:.4f} +- {self.sem:.4f} "
            f"(sd {self.measured_sd:.4f}, n = {self.n})",
            f"  gap to prediction   {self.gap_to_prediction:.4f} = {self.gap_in_se:.2f} SE "
            f"(reported, not asserted)",
            f"  unstable band       {self.band[0]}-{self.band[1]}, "
            f"every replicate inside: {self.all_inside_band}",
            f"  initial vs selected spread  {self.initial_spread:.3f} -> "
            f"{self.selected_spread:.3f} (ratio {self.spread_ratio:.4f}), "
            f"{self.initials_outside_band}/{self.n} initial modes were outside the band",
        ]
        if self.efolds is not None:
            lines.append(f"  horizon             {self.efolds:.1f} e-folds of the fastest mode")
        lines.extend(f"  {check}" for check in self.checks)
        return "\n".join(lines)


def _mean_sd_sem(values: Sequence[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    if array.size < 2:
        raise ValueError(
            f"a selected-mode distribution needs at least 2 replicates, got {array.size}; "
            "with one there is no standard error and every margin is infinite"
        )
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    return mean, sd, sd / math.sqrt(array.size)


def selection_report(
    model_name: str,
    params: dict[str, Any],
    params_factory: Callable[[dict[str, Any]], Any],
    *,
    predicted_mode: float,
    competitors: Mapping[str, float],
    band: tuple[int, int],
    replicates: int,
    seed: int = 0,
    z: float = 4.0,
    max_steps: int = 1_000_000,
    record_every: int = 1,
    mode_key: str = "dominant_mode",
    initial_mode_key: str = "initial_mode",
    efolds: float | None = None,
    min_efolds: float = 20.0,
    sem_floor: float = 1e-12,
) -> SelectionReport:
    """Measure which mode a random start selects, and test it against ``competitors``.

    Parameters
    ----------
    model_name, params, params_factory:
        As in :func:`~sandbox.core.validation.validate`.
    predicted_mode:
        The mode the model predicts, as a possibly non-integer number. The caller
        computes it — this module deliberately knows nothing about how.
    competitors:
        ``{name: mode}`` for each hypothesis the measurement must exclude. An empty
        mapping is refused: with nothing to exclude, a "discrimination" report would
        pass by having no content, which is this project's oldest failure mode.
    band:
        ``(lowest, highest)`` unstable integer mode, inclusive. Every replicate must
        select inside it.
    replicates:
        How many seeded runs. The standard error falls as ``1/sqrt(replicates)``, so
        this sets how sharp the discrimination is — and, unlike an equality check, a
        larger value here makes the report *stronger* rather than eventually false.
    efolds, min_efolds:
        The horizon in e-folds of the fastest mode, and the minimum below which this
        refuses to report. A pattern whose wavenumber is still drifting has not
        selected anything yet; the default of 20 was swept at 12, 16, 20, 24 and 30.
        Pass ``efolds=None`` to skip the check, which is only appropriate when the
        caller has established settling some other way.
    """
    if not competitors:
        raise ValueError(
            "selection_report needs at least one competing hypothesis: the claim is "
            "that the measurement excludes the alternatives, and with no alternatives "
            "there is nothing to exclude and the report would pass vacuously"
        )
    if efolds is not None and efolds < min_efolds:
        raise ValueError(
            f"the horizon is {efolds:.2f} e-folds of the fastest mode, below the "
            f"minimum {min_efolds:g}: the selected wavenumber is still drifting there, "
            "so what would be reported is a transient, not a selection. Increase t_max"
        )

    experiment = Experiment(
        model=model_name,
        params=dict(params),
        replicates=replicates,
        seed=seed,
        max_steps=max_steps,
        record_every=record_every,
    )
    result: Result = run_experiment(experiment, params_factory)
    finals = result.final_observables[0]  # single param point, no sweep

    missing = [key for key in (mode_key, initial_mode_key) if any(key not in f for f in finals)]
    if missing:
        raise KeyError(
            f"model {model_name!r} does not report {missing} as observables; "
            "selection_report reads the selected mode and the initially-loaded mode "
            "by name, through the protocol surface only"
        )

    selected = [float(f[mode_key]) for f in finals]
    initial = [float(f[initial_mode_key]) for f in finals]
    if any(not math.isfinite(value) for value in selected):
        raise ValueError(
            f"{mode_key} is non-finite in at least one replicate's final observables; "
            "a selected mode of nan means the run recorded no pattern (t = 0?), and "
            "averaging it would produce a number that is not a measurement"
        )

    mean, sd, sem = _mean_sd_sem(selected)
    sem_eff = max(sem, sem_floor)
    gap_prediction = abs(mean - predicted_mode)

    checks = []
    for name, mode in competitors.items():
        gap = abs(mean - float(mode))
        margin = (gap - gap_prediction) / sem_eff
        checks.append(
            HypothesisCheck(
                name=name,
                mode=float(mode),
                gap=gap,
                margin_se=margin,
                passed=margin >= z,
            )
        )

    low, high = band
    selected_array = np.asarray(selected, dtype=float)
    initial_array = np.asarray(initial, dtype=float)
    return SelectionReport(
        model=model_name,
        predicted_mode=float(predicted_mode),
        measured_mean=mean,
        measured_sd=sd,
        sem=sem,
        n=len(selected),
        z=z,
        band=(int(low), int(high)),
        selected_modes=tuple(selected),
        initial_modes=tuple(initial),
        efolds=efolds,
        checks=tuple(checks),
        all_inside_band=bool(((selected_array >= low) & (selected_array <= high)).all()),
        initial_spread=float(initial_array.std(ddof=1)) if len(initial) > 1 else 0.0,
        selected_spread=sd,
        initials_outside_band=int(((initial_array < low) | (initial_array > high)).sum()),
    )
