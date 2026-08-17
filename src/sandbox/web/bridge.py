"""The worker-side adapter: an ``Experiment`` in, streamed observables out.

A native caller runs an experiment by calling :func:`~sandbox.core.sweep.run_experiment`
and waiting. A browser cannot wait: the run must be *interruptible* — advanced a
chunk at a time, drawn between chunks — or the page shows nothing until it is
over, which for the repressilator is ~15 s of blank canvas. That is the only
thing this module adds. Everything else it defers to the existing services.

**The invariant that makes it trustworthy: chunking changes nothing.** A run
advanced 4 096 steps at a time and the same run advanced one step at a time
produce byte-identical trajectories *and* leave the generator in the same state,
because both drive the same :class:`~sandbox.core.recorder.ReplicateRunner`. And
the replicate RNGs are spawned through the *same two-level chain* that
:func:`run_experiment` uses (``SeedSequence(seed).spawn(1)`` for the single
param point, then one child per replicate), so a browser run of a given
``Experiment`` is the same run as a native one — not merely a similar one.
``tests/test_web_bridge.py`` asserts both, on the trajectory and on
``rng.bit_generator.state``: a matching final trajectory can hide a compensating
reordering of draws, which is the trap Phase 3e recorded.

**What crosses the boundary.** Plain JSON for everything except field pixels,
which cross as raw RGBA bytes (the measurements price a transferable 8 MB buffer
at 0.41-0.78 ms against 9.7-2812 ms for a structured-clone copy). JSON was
chosen over PyProxy objects for the scalar path because 4 096 boundary crossings
measured cheaper than the noise on a single 115 ms workload — the crossing is
not what costs, so the readable option is free.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from typing import Any

import numpy as np
from numpy.random import SeedSequence

import sandbox.models  # noqa: F401  (importing registers the built-in models)
from sandbox.core.ode import integrate_rk4
from sandbox.core.protocol import (
    DeterministicLimitModel,
    Experiment,
    FieldModel,
    TerminableModel,
    ValidatableModel,
)
from sandbox.core.recorder import ReplicateRunner
from sandbox.core.registry import available, get_model, get_params_factory, get_params_type
from sandbox.core.rng import spawn_rngs
from sandbox.core.validation import validate
from sandbox.web.colormap import COLORMAPS, to_rgba

__all__ = [
    "SESSION",
    "Run",
    "Session",
    "default_params",
    "describe_models",
    "deterministic_limit",
    "experiment_from_spec",
    "fingerprint_spec",
    "validate_spec",
]


def experiment_from_spec(spec: dict[str, Any]) -> Experiment:
    """Build an :class:`Experiment` from a front-end spec dict.

    The spec *is* an ``Experiment`` — same keys, same meaning — with two
    restrictions. A sweep is refused because a sweep is a batch job with nothing
    to watch, and the front-end's whole point is watching one run go; and
    ``observables`` is ignored because the front-end streams every observable the
    model reports and decides what to draw on the JS side.
    """
    spec = {k: v for k, v in spec.items() if k not in ("limit", "observables")}
    experiment = Experiment.from_dict(spec)
    if experiment.sweep:
        raise ValueError(
            "the front-end runs a single param point, not a sweep; sweeps are a "
            "batch workload and stay native (see docs/plans/phase4-plan.md)"
        )
    if experiment.replicates < 1:
        raise ValueError(f"replicates must be >= 1, got {experiment.replicates}")
    return experiment


def _params_for(experiment: Experiment) -> Any:
    return get_params_factory(experiment.model)(dict(experiment.params))


def _spawn_replicate_rngs(experiment: Experiment) -> list[Any]:
    """The replicate generators, spawned exactly as ``run_experiment`` spawns them.

    Two levels, not one. ``run_experiment`` spawns one ``SeedSequence`` per sweep
    point and *then* one generator per replicate; a single param point is still a
    sweep of length one, so skipping the first spawn would give different streams
    from the same seed. The browser would then disagree with native for a reason
    that looks like a WebAssembly floating-point difference and is not.
    """
    point_seeds = SeedSequence(experiment.seed).spawn(1)
    return spawn_rngs(point_seeds[0], experiment.replicates)


def _observable_keys(model: Any, params: Any) -> tuple[str, ...]:
    """The model's observable names, in order, without consuming a step.

    ``initial_state`` may draw from the generator, so it gets a throwaway one —
    the run's own generators must not be touched by a metadata query.
    """
    probe = model.initial_state(params, np.random.default_rng(0))
    return tuple(model.observables(probe).keys())


def _finite(value: float) -> float | None:
    """JSON has no NaN or Infinity. Non-finite values cross as ``null``."""
    value = float(value)
    return value if math.isfinite(value) else None


class Run:
    """One live experiment: ``replicates`` runners advanced in lockstep.

    Lockstep rather than one-at-a-time because the picture is an *overlay* —
    replicates scattered around their deterministic limit — and a chart in which
    replicate 0 is finished while replicate 3 has not started is not that
    picture. The cost is nil: the runners are independent, so advancing them
    round-robin visits exactly the same states in exactly the same order as
    running each to completion.
    """

    def __init__(self, run_id: str, experiment: Experiment) -> None:
        self.run_id = run_id
        self.experiment = experiment
        self.model = get_model(experiment.model)
        self.params = _params_for(experiment)
        self.observable_keys = _observable_keys(self.model, self.params)

        self.runners = [
            ReplicateRunner(
                self.model,
                self.params,
                rng,
                max_steps=experiment.max_steps,
                record_every=experiment.record_every,
            )
            for rng in _spawn_replicate_rngs(experiment)
        ]
        # How many recorded points of each replicate the front-end already holds.
        self._drained = [0] * len(self.runners)
        # Seconds spent inside advance(), for a like-with-like speed comparison.
        self.compute_seconds = 0.0

    @property
    def finished(self) -> bool:
        return all(runner.finished for runner in self.runners)

    def advance(self, n_steps: int) -> dict[str, Any]:
        """Advance every replicate by up to ``n_steps``; report what happened.

        The elapsed time is accumulated because it is the only figure that
        compares like with like across the worker and the main thread: a
        wall-clock window around a worker run also contains message round trips,
        JSON draining and progress posting, so comparing windows would price the
        plumbing and call it the simulation.
        """
        started = time.perf_counter()
        taken = [runner.advance(n_steps) for runner in self.runners]
        self.compute_seconds += time.perf_counter() - started
        return {
            "run_id": self.run_id,
            "compute_seconds": self.compute_seconds,
            "steps_taken": taken,
            "steps": [runner.steps for runner in self.runners],
            "t": [_finite(runner.state.t) for runner in self.runners],
            "terminal": [runner.terminal for runner in self.runners],
            "finished": self.finished,
            "pending": [
                len(runner.trajectory.times) - drained
                for runner, drained in zip(self.runners, self._drained, strict=True)
            ],
        }

    def drain(self) -> dict[str, Any]:
        """Everything recorded since the last drain, and nothing twice.

        The front-end appends what it gets, so a point re-sent is a point drawn
        twice; the cursor is per replicate and only ever moves forward.
        """
        replicates = []
        for index, runner in enumerate(self.runners):
            traj = runner.trajectory
            start = self._drained[index]
            stop = len(traj.times)
            replicates.append(
                {
                    "from": start,
                    "t": [_finite(v) for v in traj.times[start:stop]],
                    "series": {
                        key: [_finite(v) for v in values[start:stop]]
                        for key, values in traj.series.items()
                    },
                }
            )
            self._drained[index] = stop
        return {"run_id": self.run_id, "replicates": replicates}

    def status(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model": self.experiment.model,
            "replicates": len(self.runners),
            "max_steps": self.experiment.max_steps,
            "record_every": self.experiment.record_every,
            "observable_keys": list(self.observable_keys),
            "compute_seconds": self.compute_seconds,
            "steps": [runner.steps for runner in self.runners],
            "t": [_finite(runner.state.t) for runner in self.runners],
            "terminal": [runner.terminal for runner in self.runners],
            "finished": self.finished,
            "recorded": [len(runner.trajectory.times) for runner in self.runners],
        }

    def field_names(self) -> tuple[str, ...]:
        if not isinstance(self.model, FieldModel):
            return ()
        return tuple(self.model.fields(self.runners[0].state).keys())

    def field_rgba(
        self,
        name: str,
        *,
        replicate: int = 0,
        cmap: str = "ember",
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """One named field as an ``(h*w*4,)`` RGBA byte array, plus its metadata.

        The metadata carries the colour range actually used, because an image
        drawn with a silently autoscaled range makes a claim about *shape* while
        hiding a claim about *magnitude* — a pattern collapsing to nothing looks
        identical to one at full strength if the scale follows it down. The
        caller is expected to display the range next to the picture.
        """
        if not isinstance(self.model, FieldModel):
            raise TypeError(
                f"model {self.experiment.model!r} has no fields(); it is not a FieldModel"
            )
        fields = self.model.fields(self.runners[replicate].state)
        if name not in fields:
            raise KeyError(f"unknown field {name!r}; this model has {sorted(fields)}")
        field = np.asarray(fields[name], dtype=float)
        rgba, used_min, used_max = to_rgba(field, cmap=cmap, vmin=vmin, vmax=vmax)
        meta = {
            "run_id": self.run_id,
            "field": name,
            "replicate": replicate,
            "width": int(field.shape[1]),
            "height": int(field.shape[0]),
            "cmap": cmap,
            "vmin": _finite(used_min),
            "vmax": _finite(used_max),
            "t": _finite(self.runners[replicate].state.t),
        }
        return rgba, meta


class Session:
    """The worker's set of live runs, keyed by an opaque id."""

    def __init__(self) -> None:
        self._runs: dict[str, Run] = {}
        self._next = 0

    def create(self, spec: dict[str, Any]) -> dict[str, Any]:
        experiment = experiment_from_spec(spec)
        self._next += 1
        run_id = f"run-{self._next}"
        run = Run(run_id, experiment)
        self._runs[run_id] = run
        status = run.status()
        status["fields"] = list(run.field_names())
        return status

    def get(self, run_id: str) -> Run:
        try:
            return self._runs[run_id]
        except KeyError:
            known = ", ".join(sorted(self._runs)) or "(none)"
            raise KeyError(f"unknown run {run_id!r}; live runs: {known}") from None

    def close(self, run_id: str) -> dict[str, Any]:
        self._runs.pop(run_id, None)
        return {"run_id": run_id, "closed": True, "live": sorted(self._runs)}

    def close_all(self) -> None:
        self._runs.clear()


#: The worker's single session. A worker is one thread with one interpreter, so
#: module-level state is the honest expression of it.
SESSION = Session()


def describe_models() -> dict[str, Any]:
    """Every registered model and what the front-end can do with it.

    The capability flags are ``isinstance`` checks against the protocol classes,
    so this list cannot drift from what the models actually implement — it is
    derived, not maintained.
    """
    models = []
    for name in available():
        model = get_model(name)
        try:
            get_params_type(name)
            has_params = True
        except KeyError:
            has_params = False
        models.append(
            {
                "name": name,
                "validatable": isinstance(model, ValidatableModel),
                "terminable": isinstance(model, TerminableModel),
                "deterministic_limit": isinstance(model, DeterministicLimitModel),
                "fields": isinstance(model, FieldModel),
                "params_from_json": has_params,
            }
        )
    return {"models": models, "colormaps": sorted(COLORMAPS)}


def deterministic_limit(spec: dict[str, Any]) -> dict[str, Any]:
    """The ODE the stochastic run collapses into, integrated with RK4.

    ``spec["limit"]`` supplies ``t_max`` and ``dt`` explicitly rather than being
    dug out of the params. Models spell their horizon differently (``t_max``
    here, a step count there), and a front-end that guessed a param name would
    work for the models it was written against and fail silently on the next one.
    """
    experiment = experiment_from_spec(spec)
    model = get_model(experiment.model)
    if not isinstance(model, DeterministicLimitModel):
        return {
            "model": experiment.model,
            "available": False,
            "reason": (
                f"model {experiment.model!r} does not implement DeterministicLimitModel "
                "(deterministic_rhs / initial_concentrations), so it declares no ODE limit"
            ),
        }

    limit = dict(spec.get("limit") or {})
    if "t_max" not in limit:
        raise ValueError("spec['limit'] must supply t_max (the horizon to integrate to)")
    t_max = float(limit["t_max"])
    dt = float(limit.get("dt", t_max / 2000.0))

    params = _params_for(experiment)
    rhs = model.deterministic_rhs(params)
    c0 = np.asarray(model.initial_concentrations(params), dtype=float)
    keys = _observable_keys(model, params)
    if len(keys) != c0.size:
        raise ValueError(
            f"model {experiment.model!r} reports {len(keys)} observables but its ODE "
            f"vector has {c0.size} components; the front-end overlays them column by "
            "column and in order, so they must correspond one-to-one"
        )

    t, y = integrate_rk4(rhs, c0, t_max, dt)
    return {
        "model": experiment.model,
        "available": True,
        "t_max": t_max,
        "dt": dt,
        "t": [_finite(v) for v in t.tolist()],
        "series": {key: [_finite(v) for v in y[:, i].tolist()] for i, key in enumerate(keys)},
    }


def validate_spec(spec: dict[str, Any], *, z: float = 4.0) -> dict[str, Any]:
    """Run the ValidationSuite on this spec and report the verdict.

    A model without ``analytic_predictions`` is **not** an error here, and the
    refusal is the interesting output rather than a gap to paper over: the
    repressilator's deterministic limit is a limit cycle, so there is no scalar
    to match an ensemble mean against, and native ``validate()`` raises for
    exactly that reason. The front-end reports the refusal *with its reason*, and
    matching that refusal is what "the browser verdict matches native" means for
    that model — a pass mark quoting a tolerance would be meaningless for it.
    """
    experiment = experiment_from_spec(spec)
    model = get_model(experiment.model)
    if not isinstance(model, ValidatableModel):
        return {
            "model": experiment.model,
            "validatable": False,
            "passed": None,
            "reason": (
                f"model {experiment.model!r} has no analytic_predictions, so the "
                "ValidationSuite has nothing to check it against; its checkable claim "
                "is convergence to its deterministic limit, which is a sweep and stays "
                "native (see docs/plans/phase4-plan.md)"
            ),
            "checks": [],
        }

    report = validate(experiment, get_params_factory(experiment.model), z=z)
    return {
        "model": report.model,
        "validatable": True,
        "passed": report.passed,
        "reason": None,
        "checks": [
            {
                "name": check.name,
                "predicted": _finite(check.predicted),
                "measured": _finite(check.measured),
                "sem": _finite(check.sem),
                "tolerance": _finite(check.tolerance),
                "z": _finite(check.z),
                "z_score": _finite(check.z_score),
                "n": check.n,
                "passed": check.passed,
                "line": str(check),
            }
            for check in report.checks
        ],
    }


def discrepancy_to_limit(spec: dict[str, Any], *, n_grid: int = 200) -> dict[str, Any]:
    """How far this run's replicates sit from the ODE they collapse into.

    Reuses :mod:`sandbox.core.convergence`'s own ``_per_replicate_discrepancy``
    rather than computing something similar here. The underscore is stepped over
    deliberately: that function is the load-bearing computation of the whole
    convergence pathway, unit-tested against hand-worked trajectories, and it
    averages over time and species **per replicate before** the replicate mean —
    which is the correct order and the one that avoids the phase-diffusion trap
    the module was built around. A second implementation on this side would be a
    lookalike that could drift, and the front-end's job is to show the project's
    quantities, not new ones.

    **What this is and is not.** It is a demonstration that the discrepancy falls
    as ``Omega`` grows. It is *not* the validated claim: that is the log-log slope
    of ``D(Omega)`` against ``-1/2`` with a statistical standard error, which
    needs a sweep over many system sizes and many replicates each, is minutes of
    CPU, and stays native. The returned dict says so in ``claim``.
    """
    from sandbox.core.convergence import (  # local: keeps the import graph shallow
        _integrate_on_grid,
        _per_replicate_discrepancy,
    )

    experiment = experiment_from_spec(spec)
    model = get_model(experiment.model)
    if not isinstance(model, DeterministicLimitModel):
        return {"available": False, "reason": f"model {experiment.model!r} declares no ODE limit"}

    limit_spec = dict(spec.get("limit") or {})
    if "t_max" not in limit_spec:
        raise ValueError("spec['limit'] must supply t_max")
    t_max = float(limit_spec["t_max"])
    dt = float(limit_spec.get("dt", t_max / 2000.0))

    params = _params_for(experiment)
    keys = _observable_keys(model, params)
    grid = np.linspace(0.0, t_max, int(n_grid))
    ode_on_grid = _integrate_on_grid(
        model.deterministic_rhs(params),
        np.asarray(model.initial_concentrations(params), dtype=float),
        t_max,
        dt,
        grid,
    )

    run = Run("discrepancy", experiment)
    while not run.finished:
        run.advance(8192)

    # A replicate that never reached the horizon would have its last value held
    # across the rest of the grid, quietly turning a truncation into a
    # discrepancy. Reported rather than silently averaged in.
    truncated = [i for i, r in enumerate(run.runners) if r.state.t < t_max]
    values = []
    for runner in run.runners:
        times, series = runner.trajectory.as_arrays()
        values.append(_per_replicate_discrepancy(times, series, grid, ode_on_grid, keys))

    mean = float(np.mean(values)) if values else math.nan
    sem = float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else math.inf
    return {
        "available": True,
        "model": experiment.model,
        "D": _finite(mean),
        "sem": _finite(sem),
        "per_replicate": [_finite(v) for v in values],
        "replicates": len(values),
        "n_grid": int(n_grid),
        "t_max": t_max,
        "truncated": truncated,
        "claim": (
            "A demonstration, not the validated claim. What is validated is the "
            "log-log slope of D(Omega) against -1/2 with a statistical standard "
            "error; that needs a sweep over many system sizes, is minutes of CPU, "
            "and stays native."
        ),
    }


def colormap_strip(
    name: str, width: int = 256, height: int = 1
) -> tuple[np.ndarray, dict[str, Any]]:
    """A horizontal ramp through ``name``, as RGBA — the scale bar for a field.

    Produced here, by the same :func:`~sandbox.web.colormap.to_rgba` the field
    image goes through, rather than reconstructed in JavaScript. A scale bar
    drawn independently is a second implementation of the mapping, and the one
    thing a scale bar must never do is disagree with the picture it labels.
    """
    ramp = np.tile(np.linspace(0.0, 1.0, int(width)), (int(height), 1))
    rgba, lo, hi = to_rgba(ramp, cmap=name, vmin=0.0, vmax=1.0)
    return rgba, {
        "cmap": name,
        "width": int(width),
        "height": int(height),
        "vmin": lo,
        "vmax": hi,
    }


def fingerprint_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """sha256 of each replicate's trajectory, for a browser-vs-native comparison.

    **Informational, and deliberately not a pass mark.** Six of the project's
    models happen to reproduce bit-for-bit under WebAssembly and the rest differ
    in the last couple of digits, because IEEE-754 requires correctly-rounded
    ``+ - * /`` and ``sqrt`` but says nothing about ``exp``, ``log`` or ``tanh``,
    whose implementations are a property of the toolchain. Recorded fingerprints
    are **native** anchors; a browser run is validated statistically, through
    ``analytic_predictions`` and the ValidationSuite, which is how every model in
    this project is defined as correct anyway. Nothing here may be built on this
    matching — it is reported because a *change* in whether it matches is worth
    noticing.
    """
    experiment = experiment_from_spec(spec)
    run = Run("fingerprint", experiment)
    while not run.finished:
        run.advance(4096)

    digests = []
    for runner in run.runners:
        h = hashlib.sha256()
        times, series = runner.trajectory.as_arrays()
        h.update(times.tobytes())
        for key in sorted(series):
            h.update(key.encode())
            h.update(series[key].tobytes())
        h.update(b"terminated" if runner.trajectory.terminated else b"truncated")
        digests.append(h.hexdigest())

    return {
        "model": experiment.model,
        "replicates": len(digests),
        "digests": digests,
        "recorded": [len(runner.trajectory.times) for runner in run.runners],
        "steps": [runner.steps for runner in run.runners],
    }


def default_params(model_name: str) -> dict[str, Any]:
    """The registered params dataclass's defaults, for pre-filling a form.

    Fields without a default are reported as ``None``, so the UI can tell "the
    model does not care" from "you must choose".
    """
    params_type = get_params_type(model_name)
    out: dict[str, Any] = {}
    for f in dataclasses.fields(params_type):
        if f.default is not dataclasses.MISSING:
            out[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            out[f.name] = f.default_factory()  # type: ignore[misc]
        else:
            out[f.name] = None
    return json.loads(json.dumps(out))
