"""The browser bridge, checked natively.

The bridge's whole job is to make a run *interruptible* without changing it, so
almost every test here is a form of one question: **does chunking change the
answer?** It is asked against the trajectory, against the generator state, and
against a deliberately broken chunker that must fail — because a trajectory that
matches at the end can still hide a compensating reordering of draws (Phase 3e),
and a chunking test that only ever chunks by multiples of ``record_every`` cannot
see a phase bug (the "sweep the constant the probe lands on" lesson, which this
project has now been caught by four times).

The second question is whether a browser run *is the native run*. That reduces
to the RNG spawn chain: ``run_experiment`` spawns one ``SeedSequence`` per sweep
point and only then one generator per replicate, so a bridge that spawned
directly from the seed would produce a different — equally valid, equally
reproducible, and completely incomparable — run. The chain is written out
longhand below rather than imported from the bridge, because a test that calls
the code it is checking proves nothing.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np
import pytest
from numpy.random import SeedSequence

import sandbox.models  # noqa: F401  (registers models)
from sandbox.core.protocol import Experiment
from sandbox.core.recorder import ReplicateRunner, Trajectory, run_replicate
from sandbox.core.registry import (
    available,
    get_model,
    get_params_factory,
    get_params_type,
    register,
)
from sandbox.core.rng import spawn_rngs
from sandbox.core.sweep import run_experiment
from sandbox.core.validation import validate
from sandbox.web import bridge
from sandbox.web.colormap import (
    DIVERGING,
    SEQUENTIAL,
    lut,
    relative_luminance,
    to_rgba,
)

# --------------------------------------------------------------------------
# Specs. Small enough that the whole module stays off the suite's critical path
# (two multi-minute convergence tests), which is why none of them is a headline
# configuration.
# --------------------------------------------------------------------------

WF_SPEC = {
    "model": "wright_fisher",
    "params": {"N": 200, "p0": 0.3},
    "replicates": 6,
    "seed": 12345,
    "max_steps": 4000,
    "record_every": 1,
}

REPRESSILATOR_SPEC = {
    "model": "repressilator",
    "params": {
        "alpha": 216.0,
        "alpha0": 0.216,
        "n_hill": 2.0,
        "beta": 1.0,
        "Omega": 50.0,
        "t_max": 3.0,
    },
    "replicates": 3,
    "seed": 99,
    "max_steps": 6000,
    "record_every": 5,
    "limit": {"t_max": 3.0, "dt": 0.005},
}

GRAY_SCOTT_SPEC = {
    "model": "gray_scott",
    "params": {"n": 32, "dt": 0.5, "t_max": 50.0, "initial": "mode", "mode_j": 3},
    "replicates": 1,
    "seed": 3,
    "max_steps": 100,
    "record_every": 10,
}

# Params classes whose required fields mean they cannot be built with no
# arguments. Listed explicitly so the JSON round-trip below covers all 14
# registered models rather than only the 9 that happen to have full defaults.
_REQUIRED_PARAMS: dict[str, dict] = {
    "birth_death": {"k": 1.0, "gamma": 0.5, "Omega": 100.0, "t_max": 5.0},
    "glv_stochastic": {"Omega": 100.0, "t_max": 5.0},
    "hh_voltage_clamp": {"v_clamp": -20.0},
    "isomerization": {"k1": 1.0, "k2": 2.0, "Omega": 100.0, "t_max": 5.0, "c_tot": 1.0},
    "repressilator": REPRESSILATOR_SPEC["params"],
    "wright_fisher": {"N": 100, "p0": 0.4},
}


def fingerprint(traj: Trajectory) -> str:
    """sha256 of times + every series + the termination flag."""
    h = hashlib.sha256()
    times, series = traj.as_arrays()
    h.update(times.tobytes())
    for key in sorted(series):
        h.update(key.encode())
        h.update(series[key].tobytes())
    h.update(b"terminated" if traj.terminated else b"truncated")
    return h.hexdigest()


def native_replicate_rngs(experiment: Experiment) -> list[np.random.Generator]:
    """The two-level spawn chain from ``run_experiment``, written out longhand.

    Deliberately *not* ``bridge._spawn_replicate_rngs``: this is the reference
    the bridge is being checked against, so it must be an independent
    transcription of ``sweep.run_experiment``'s seeding, not a call into the
    thing under test.
    """
    sweep_points = 1  # a single param point is still a sweep of length one
    point_seeds = SeedSequence(experiment.seed).spawn(sweep_points)
    return spawn_rngs(point_seeds[0], experiment.replicates)


def generator_state(rng: np.random.Generator) -> dict:
    return dict(rng.bit_generator.state["state"])


# --------------------------------------------------------------------------
# 1. The bridge run IS the native run
# --------------------------------------------------------------------------


@pytest.mark.parametrize("spec", [WF_SPEC, REPRESSILATOR_SPEC], ids=["wf", "repressilator"])
def test_bridge_run_is_bit_identical_to_run_experiment(spec):
    """Same spec, same trajectories — every replicate, byte for byte.

    This is the test everything else in Phase 4 rests on. Until it passes, a
    browser-vs-native comparison is comparing two different runs, and a
    disagreement would look like a WebAssembly floating-point problem while
    being a seeding problem.
    """
    experiment = bridge.experiment_from_spec(spec)
    native = run_experiment(experiment, get_params_factory(experiment.model))

    run = bridge.Run("test", experiment)
    while not run.finished:
        run.advance(97)  # deliberately not a divisor of anything

    assert len(run.runners) == len(native.trajectories[0])
    for runner, native_traj in zip(run.runners, native.trajectories[0], strict=True):
        assert fingerprint(runner.trajectory) == fingerprint(native_traj)


@pytest.mark.parametrize("spec", [WF_SPEC, REPRESSILATOR_SPEC], ids=["wf", "repressilator"])
def test_bridge_leaves_the_generators_where_native_leaves_them(spec):
    """The pin is on the generator state, not only the trajectory.

    A matching final trajectory can hide a compensating reordering of draws — a
    stream consumed in a different order that happens to land on the same
    numbers. Phase 3e pinned two implementations of one jump process this way
    for exactly that reason.
    """
    experiment = bridge.experiment_from_spec(spec)
    factory = get_params_factory(experiment.model)
    params = factory(dict(experiment.params))
    model = get_model(experiment.model)

    reference_rngs = native_replicate_rngs(experiment)
    for rng in reference_rngs:
        run_replicate(
            model,
            params,
            rng,
            max_steps=experiment.max_steps,
            record_every=experiment.record_every,
        )

    run = bridge.Run("test", experiment)
    while not run.finished:
        run.advance(13)

    for runner, rng in zip(run.runners, reference_rngs, strict=True):
        assert generator_state(runner.rng) == generator_state(rng)


def test_a_one_level_spawn_would_not_have_matched():
    """The non-vacuity guard on the two tests above.

    If ``spawn_rngs(seed, R)`` and the two-level chain gave the same streams, the
    pin would pass no matter which one the bridge used and would be checking
    nothing. They do not, and this records that they do not.
    """
    experiment = bridge.experiment_from_spec(WF_SPEC)
    two_level = native_replicate_rngs(experiment)
    one_level = spawn_rngs(experiment.seed, experiment.replicates)
    assert generator_state(two_level[0]) != generator_state(one_level[0])


# --------------------------------------------------------------------------
# 2. Chunking changes nothing
# --------------------------------------------------------------------------


def _run_in_chunks(spec: dict, chunk: int) -> tuple[list[str], list[dict]]:
    experiment = bridge.experiment_from_spec(spec)
    run = bridge.Run("test", experiment)
    guard = 0
    while not run.finished:
        run.advance(chunk)
        guard += 1
        assert guard < 100_000, "advance() is not making progress"
    return (
        [fingerprint(runner.trajectory) for runner in run.runners],
        [generator_state(runner.rng) for runner in run.runners],
    )


@pytest.mark.parametrize("chunk", [1, 3, 7, 64, 1000, 4096])
@pytest.mark.parametrize("record_every", [1, 5], ids=["every1", "every5"])
def test_chunk_size_does_not_move_the_trajectory(chunk, record_every):
    """Six chunk sizes against two recording strides, and 8 of the 12 pairs do
    not divide evenly.

    That is the point of the grid. ``record_every`` is counted against the
    *cumulative* step index, so a chunked stepper that restarted the count each
    chunk would record a different set of states — and would still look correct
    at every chunk size that happens to be a multiple of the stride.
    """
    spec = {**REPRESSILATOR_SPEC, "record_every": record_every}
    reference = _run_in_chunks(spec, 10_000)
    assert _run_in_chunks(spec, chunk) == reference


def test_a_per_chunk_recording_counter_would_have_been_caught():
    """The tooth for the test above: break the stride and it must go red.

    A stepper that counts ``record_every`` within each chunk instead of across
    the run records a different set of states whenever a chunk boundary falls
    mid-stride. If that mutant passed, the grid above would be decoration.
    """

    class PerChunkRunner(ReplicateRunner):
        def advance(self, n_steps: int) -> int:
            model, rng, traj = self.model, self.rng, self.trajectory
            is_terminal = self._is_terminal
            state, step = self.state, self.steps
            terminal_now = self.terminal
            budget = min(step + n_steps, self.max_steps)
            started_at = step
            within_chunk = 0  # <-- the mutation: a per-chunk counter
            while step < budget and not terminal_now:
                state = model.step(state, rng)
                step += 1
                within_chunk += 1
                terminal_now = bool(is_terminal and is_terminal(state))
                if within_chunk % self.record_every == 0 or terminal_now:
                    traj.record(state.t, model.observables(state))
            self.state, self.steps, self.terminal = state, step, terminal_now
            traj.terminated = terminal_now
            return step - started_at

    experiment = bridge.experiment_from_spec({**REPRESSILATOR_SPEC, "record_every": 5})
    params = get_params_factory(experiment.model)(dict(experiment.params))
    model = get_model(experiment.model)
    rng_good, rng_bad = native_replicate_rngs(experiment)[0], native_replicate_rngs(experiment)[0]

    good = ReplicateRunner(model, params, rng_good, max_steps=2000, record_every=5)
    bad = PerChunkRunner(model, params, rng_bad, max_steps=2000, record_every=5)
    for _ in range(200):
        good.advance(7)  # 7 is not a multiple of 5 — the whole point
        bad.advance(7)

    assert fingerprint(good.trajectory) != fingerprint(bad.trajectory)
    # ...and the draws are identical, so only the *recording* differs. Without
    # this the tooth could be biting a different bug than the one it names.
    assert generator_state(good.rng) == generator_state(bad.rng)


def test_advance_reports_how_many_steps_it_actually_took():
    experiment = bridge.experiment_from_spec({**REPRESSILATOR_SPEC, "max_steps": 100})
    run = bridge.Run("test", experiment)
    first = run.advance(60)
    assert first["steps_taken"] == [60] * len(run.runners)
    second = run.advance(60)  # only 40 of the budget left
    assert second["steps_taken"] == [40] * len(run.runners)
    assert run.advance(60)["steps_taken"] == [0] * len(run.runners)
    assert run.finished


def test_a_terminal_model_stops_early_and_says_so():
    experiment = bridge.experiment_from_spec(WF_SPEC)
    run = bridge.Run("test", experiment)
    while not run.finished:
        run.advance(50)
    status = run.status()
    assert all(status["terminal"]), "Wright-Fisher must absorb well inside 4000 steps"
    assert all(steps < experiment.max_steps for steps in status["steps"])


# --------------------------------------------------------------------------
# 3. Draining: everything once, in order
# --------------------------------------------------------------------------


def test_drains_concatenate_into_exactly_the_trajectory():
    """No point sent twice, none skipped, and the order preserved.

    The front-end appends what it is given, so a duplicated point is a point
    drawn twice and a dropped one is a gap — neither of which shows up as an
    error, only as a slightly wrong picture.
    """
    experiment = bridge.experiment_from_spec(REPRESSILATOR_SPEC)
    run = bridge.Run("test", experiment)

    collected_t: list[list[float]] = [[] for _ in run.runners]
    collected: list[dict[str, list[float]]] = [{} for _ in run.runners]
    while not run.finished:
        run.advance(211)
        payload = run.drain()
        for index, part in enumerate(payload["replicates"]):
            assert part["from"] == len(collected_t[index])
            collected_t[index].extend(part["t"])
            for key, values in part["series"].items():
                collected[index].setdefault(key, []).extend(values)
    run.drain()  # a drain after the end must be empty, not a repeat

    for index, runner in enumerate(run.runners):
        assert collected_t[index] == runner.trajectory.times
        assert collected[index] == runner.trajectory.series


def test_a_drain_with_nothing_new_is_empty():
    run = bridge.Run("test", bridge.experiment_from_spec(REPRESSILATOR_SPEC))
    run.advance(100)
    assert any(part["t"] for part in run.drain()["replicates"])
    second = run.drain()
    assert all(part["t"] == [] for part in second["replicates"])
    assert all(not any(part["series"].values()) for part in second["replicates"])


# --------------------------------------------------------------------------
# 4. Everything that crosses the boundary is real JSON
# --------------------------------------------------------------------------


def _strict_json(payload) -> str:
    """``allow_nan=False`` because ``NaN`` is not JSON and ``JSON.parse`` rejects it.

    Python's default emits a bare ``NaN`` token happily, so a payload containing
    one round-trips in tests and dies in the browser.
    """
    return json.dumps(payload, allow_nan=False)


def test_every_payload_is_strictly_valid_json():
    run_status = bridge.SESSION.create(REPRESSILATOR_SPEC)
    run_id = run_status["run_id"]
    try:
        payloads = [
            bridge.describe_models(),
            run_status,
            bridge.SESSION.get(run_id).advance(500),
            bridge.SESSION.get(run_id).drain(),
            bridge.SESSION.get(run_id).status(),
            bridge.deterministic_limit(REPRESSILATOR_SPEC),
            bridge.validate_spec(REPRESSILATOR_SPEC),
            bridge.validate_spec(WF_SPEC),
            bridge.default_params("repressilator"),
        ]
        for payload in payloads:
            assert _strict_json(payload)
    finally:
        bridge.SESSION.close(run_id)


@dataclasses.dataclass(frozen=True)
class _NaNParams:
    """Params for the NaN model below — no fields, and that is the point."""


@dataclasses.dataclass(frozen=True)
class _NaNState:
    t: float


class _NaNModel:
    """A model whose observables include a NaN. Registered under a ``_test_`` name."""

    def initial_state(self, params: _NaNParams, rng) -> _NaNState:
        return _NaNState(t=0.0)

    def step(self, state: _NaNState, rng) -> _NaNState:
        return _NaNState(t=state.t + 1.0)

    def observables(self, state: _NaNState) -> dict[str, float]:
        return {"finite": 1.0, "broken": float("nan")}


register("_test_web_nan", _NaNModel(), _NaNParams)


def test_non_finite_observables_cross_as_null_not_as_a_broken_token():
    """A model producing a NaN must not take the message channel down with it.

    ``json.dumps`` emits a bare ``NaN`` token by default, which is not JSON and
    which ``JSON.parse`` rejects — so the payload would round-trip in Python and
    die at the boundary. Non-finite values cross as ``null`` instead.
    """
    run = bridge.Run(
        "test",
        bridge.experiment_from_spec(
            {"model": "_test_web_nan", "params": {}, "max_steps": 3, "seed": 0}
        ),
    )
    run.advance(3)
    payload = run.drain()
    assert _strict_json(payload)
    series = payload["replicates"][0]["series"]
    assert series["broken"] == [None] * 4  # t = 0 plus three steps
    assert series["finite"] == [1.0] * 4


# --------------------------------------------------------------------------
# 5. Validation: the verdict, and the refusal
# --------------------------------------------------------------------------


def test_validate_spec_reproduces_native_validate_exactly():
    """Not "also passes" — the same numbers.

    "The browser verdict matches native" is only a check if the two are the same
    measurement. A browser run that passed its own, differently-seeded validation
    would prove that the browser can validate something, not that it validates
    this.
    """
    spec = {**WF_SPEC, "replicates": 300, "max_steps": 20_000}
    experiment = bridge.experiment_from_spec(spec)
    native = validate(experiment, get_params_factory(experiment.model))
    reported = bridge.validate_spec(spec)

    assert reported["validatable"] is True
    assert reported["passed"] is native.passed is True
    assert len(reported["checks"]) == len(native.checks) == 1
    check, native_check = reported["checks"][0], native.checks[0]
    assert check["name"] == native_check.name
    assert check["measured"] == native_check.measured
    assert check["predicted"] == native_check.predicted
    assert check["sem"] == native_check.sem
    assert check["n"] == native_check.n
    assert check["line"] == str(native_check)


def test_validate_spec_refuses_the_repressilator_the_way_native_does():
    """The refusal is the verdict, and it has to match too.

    The repressilator's deterministic limit is a limit cycle, so there is no
    scalar for an ensemble mean to match and native ``validate()`` raises. The
    bridge reports that refusal with its reason instead of raising, because a
    front-end that died on a legitimate "this model is checked a different way"
    would be worse than useless. What must not happen is a *number*.
    """
    experiment = bridge.experiment_from_spec(REPRESSILATOR_SPEC)
    with pytest.raises(TypeError, match="analytic_predictions"):
        validate(experiment, get_params_factory(experiment.model))

    reported = bridge.validate_spec(REPRESSILATOR_SPEC)
    assert reported["validatable"] is False
    assert reported["passed"] is None
    assert reported["checks"] == []
    assert "analytic_predictions" in reported["reason"]


# --------------------------------------------------------------------------
# 6. The deterministic limit
# --------------------------------------------------------------------------


def test_deterministic_limit_matches_a_direct_integration():
    from sandbox.core.ode import integrate_rk4

    limit = bridge.deterministic_limit(REPRESSILATOR_SPEC)
    assert limit["available"] is True

    model = get_model("repressilator")
    params = get_params_factory("repressilator")(dict(REPRESSILATOR_SPEC["params"]))
    t, y = integrate_rk4(
        model.deterministic_rhs(params),
        model.initial_concentrations(params),
        REPRESSILATOR_SPEC["limit"]["t_max"],
        REPRESSILATOR_SPEC["limit"]["dt"],
    )
    assert limit["t"] == t.tolist()
    keys = list(limit["series"])
    assert keys == list(model.observables(model.initial_state(params, np.random.default_rng(0))))
    for column, key in enumerate(keys):
        assert limit["series"][key] == y[:, column].tolist()


def test_the_limit_is_not_a_flat_line():
    """The empty-figure guard, applied to the data instead of the picture.

    The browser slice rendered a valid PNG with correct axes and no data because
    a filter tested ``startswith("m")`` against series named ``x_m1``. The same
    failure here would be a limit overlay that is present, correctly shaped, and
    constant.
    """
    limit = bridge.deterministic_limit(REPRESSILATOR_SPEC)
    spans = {key: max(v) - min(v) for key, v in limit["series"].items()}
    assert all(span > 1e-6 for span in spans.values()), spans


def test_a_model_without_an_ode_limit_says_so_rather_than_guessing():
    limit = bridge.deterministic_limit({**WF_SPEC, "limit": {"t_max": 10.0}})
    assert limit["available"] is False
    assert "DeterministicLimitModel" in limit["reason"]


def test_the_limit_needs_its_horizon_stated():
    with pytest.raises(ValueError, match="t_max"):
        bridge.deterministic_limit({k: v for k, v in REPRESSILATOR_SPEC.items() if k != "limit"})


# --------------------------------------------------------------------------
# 7. Specs, the registry, and JSON params
# --------------------------------------------------------------------------


def test_a_sweep_is_refused_rather_than_silently_run():
    with pytest.raises(ValueError, match="single param point"):
        bridge.experiment_from_spec({**WF_SPEC, "sweep": {"N": [100, 200]}})


def test_every_registered_model_can_build_its_params_from_json():
    """The completeness check: a model registered without a params type fails here.

    Test-only models (the broken-model "teeth" other modules register) are
    excluded by their leading underscore — they reuse a real model's params and
    never cross a JSON boundary.
    """
    for name in available():
        if name.startswith("_"):
            continue
        params_type = get_params_type(name)
        values = _REQUIRED_PARAMS.get(name, {})
        params = params_type(**values)
        round_tripped = json.loads(json.dumps(dataclasses.asdict(params)))
        assert params_type(**round_tripped) == params, name


def test_describe_models_flags_are_derived_from_the_protocol():
    described = {entry["name"]: entry for entry in bridge.describe_models()["models"]}
    assert described["wright_fisher"]["validatable"] is True
    assert described["wright_fisher"]["terminable"] is True
    assert described["wright_fisher"]["deterministic_limit"] is False
    assert described["wright_fisher"]["fields"] is False
    assert described["repressilator"]["validatable"] is False
    assert described["repressilator"]["deterministic_limit"] is True
    assert described["gray_scott"]["fields"] is True


def test_default_params_distinguishes_no_default_from_a_default():
    defaults = bridge.default_params("repressilator")
    assert defaults["alpha"] is None  # required: the caller must choose
    assert defaults["p2_0"] == 5.0  # defaulted: the model does not care


# --------------------------------------------------------------------------
# 8. Colormaps and field bytes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(SEQUENTIAL))
def test_sequential_colormaps_rise_monotonically_in_luminance(name):
    """A reader must rank two patches the same way in colour and in greyscale."""
    luminance = relative_luminance(lut(name))
    steps = np.diff(luminance)
    assert steps.min() >= 0.0, f"{name} dips at index {int(np.argmin(steps))}"
    assert luminance[-1] - luminance[0] > 0.8, "the map barely changes brightness at all"


@pytest.mark.parametrize("name", sorted(DIVERGING))
def test_diverging_colormaps_are_not_monotone_and_must_not_be(name):
    """The non-vacuity guard on the test above.

    If every map in the module were monotone, the check would be passing for
    free. A diverging map is light in the middle and dark at both ends by
    construction, so it fails that check — and being *offered* under a name that
    says so is the point.
    """
    luminance = relative_luminance(lut(name))
    assert np.diff(luminance).min() < 0.0
    assert luminance[128] > max(luminance[0], luminance[-1]) + 0.5


def test_to_rgba_shape_dtype_and_reported_range():
    field = np.linspace(0.0, 1.0, 24 * 16).reshape(24, 16)
    rgba, lo, hi = to_rgba(field, cmap="ember")
    assert rgba.shape == (24 * 16 * 4,)
    assert rgba.dtype == np.uint8
    assert (lo, hi) == (0.0, 1.0)
    assert (rgba[3::4] == 255).all(), "alpha must be opaque everywhere"
    # The ends of the field land on the ends of the map.
    assert tuple(rgba[:3]) == tuple(lut("ember")[0])
    assert tuple(rgba[-4:-1]) == tuple(lut("ember")[255])


def test_an_explicit_range_is_used_and_reported_not_ignored():
    field = np.linspace(0.0, 1.0, 64).reshape(8, 8)
    rgba, lo, hi = to_rgba(field, cmap="ember", vmin=-1.0, vmax=2.0)
    assert (lo, hi) == (-1.0, 2.0)
    # Nothing reaches either end of the map, because nothing reaches the range.
    index_of = {tuple(row): i for i, row in enumerate(lut("ember").tolist())}
    used = {index_of[tuple(rgba[i : i + 3].tolist())] for i in range(0, rgba.size, 4)}
    assert min(used) > 0 and max(used) < 255


def test_a_constant_field_maps_to_the_middle_rather_than_to_an_end():
    rgba, lo, hi = to_rgba(np.full((4, 4), 0.37), cmap="ice")
    assert lo == hi == 0.37
    assert tuple(rgba[:3]) == tuple(lut("ice")[128])


def test_non_finite_cells_do_not_index_the_table_at_random():
    field = np.array([[0.0, np.nan], [np.inf, 1.0]])
    rgba, _, _ = to_rgba(field, cmap="slate", vmin=0.0, vmax=1.0)
    assert tuple(rgba[4:7]) == tuple(lut("slate")[0])  # nan -> low end
    assert tuple(rgba[8:11]) == tuple(lut("slate")[255])  # +inf -> high end


def test_field_rgba_carries_a_real_picture_and_its_scale():
    """Not "the buffer is the right size" — that a decayed field is visibly flat.

    A field renderer that returned a correctly-shaped buffer of one colour would
    satisfy every structural assertion. What separates a picture from a blank is
    that the byte stream has variety in it, and that the range it was drawn
    against is reported alongside.
    """
    run = bridge.Run("test", bridge.experiment_from_spec(GRAY_SCOTT_SPEC))
    run.advance(100)
    assert run.field_names() == ("u", "v")

    rgba, meta = run.field_rgba("v", cmap="ember")
    assert rgba.shape == (32 * 32 * 4,)
    assert (meta["width"], meta["height"]) == (32, 32)
    assert meta["vmin"] < meta["vmax"], "a seeded mode must not have gone flat"
    assert len(set(rgba[0::4].tolist())) > 4, "the image is essentially one colour"


def test_field_rgba_refuses_a_model_that_has_no_fields():
    run = bridge.Run("test", bridge.experiment_from_spec(WF_SPEC))
    with pytest.raises(TypeError, match="FieldModel"):
        run.field_rgba("u")


def test_field_rgba_names_the_fields_it_does_have():
    run = bridge.Run("test", bridge.experiment_from_spec(GRAY_SCOTT_SPEC))
    with pytest.raises(KeyError, match="'u', 'v'"):
        run.field_rgba("w")


# --------------------------------------------------------------------------
# 9. The session
# --------------------------------------------------------------------------


def test_the_session_hands_back_runs_and_forgets_closed_ones():
    session = bridge.Session()
    first = session.create(WF_SPEC)["run_id"]
    second = session.create(REPRESSILATOR_SPEC)["run_id"]
    assert first != second
    assert session.get(first) is not session.get(second)
    session.close(first)
    with pytest.raises(KeyError, match="unknown run"):
        session.get(first)
    assert session.get(second).experiment.model == "repressilator"


def test_create_reports_the_fields_a_model_offers():
    session = bridge.Session()
    assert session.create(GRAY_SCOTT_SPEC)["fields"] == ["u", "v"]
    assert session.create(WF_SPEC)["fields"] == []


def test_two_runs_of_one_spec_agree():
    """Reproducibility (non-negotiable #3) through the front-end path too."""
    session = bridge.Session()
    ids = [session.create(REPRESSILATOR_SPEC)["run_id"] for _ in range(2)]
    for run_id in ids:
        session.get(run_id).advance(1500)
    first, second = (session.get(run_id) for run_id in ids)
    assert [fingerprint(r.trajectory) for r in first.runners] == [
        fingerprint(r.trajectory) for r in second.runners
    ]


# --------------------------------------------------------------------------
# 10. The three functions that shipped without tests
#
# fingerprint_spec, colormap_strip and discrepancy_to_limit landed in three
# separate commits and the test count never moved -- which is a suite-level
# version of the failure this file is otherwise about. The last of them turned
# out to carry a real trap.
# --------------------------------------------------------------------------


def test_fingerprint_is_stable_across_calls_and_moves_with_the_seed():
    first = bridge.fingerprint_spec(REPRESSILATOR_SPEC)
    second = bridge.fingerprint_spec(REPRESSILATOR_SPEC)
    assert first["digests"] == second["digests"]
    assert len(first["digests"]) == REPRESSILATOR_SPEC["replicates"]
    # Distinct replicates must not share a stream. If they did, the fingerprint
    # would be stable, reproducible and measuring one run four times.
    assert len(set(first["digests"])) == len(first["digests"])

    moved = bridge.fingerprint_spec({**REPRESSILATOR_SPEC, "seed": 1234})
    assert moved["digests"] != first["digests"]


def test_the_fingerprint_is_of_the_same_run_the_bridge_would_stream():
    """It must not be a second, differently-driven run that happens to agree."""
    experiment = bridge.experiment_from_spec(REPRESSILATOR_SPEC)
    run = bridge.Run("reference", experiment)
    while not run.finished:
        run.advance(37)
    expected = [fingerprint(runner.trajectory) for runner in run.runners]
    assert bridge.fingerprint_spec(REPRESSILATOR_SPEC)["digests"] == expected


def test_colormap_strip_agrees_with_the_field_path_it_labels():
    """The scale bar cannot be allowed to disagree with the picture it labels.

    It is drawn from bytes the bridge produced through ``to_rgba`` — the same
    call the field image goes through — so this asserts the two really are one
    code path rather than two that currently match.
    """
    rgba, meta = bridge.colormap_strip("ember", width=256, height=3)
    assert rgba.shape == (256 * 3 * 4,)
    assert (meta["vmin"], meta["vmax"]) == (0.0, 1.0)

    ramp = np.tile(np.linspace(0.0, 1.0, 256), (3, 1))
    direct, _, _ = to_rgba(ramp, cmap="ember", vmin=0.0, vmax=1.0)
    assert np.array_equal(rgba, direct)

    # Every row of the strip is the same ramp, and it spans the whole table.
    assert tuple(rgba[:3]) == tuple(lut("ember")[0])
    assert tuple(rgba[256 * 4 - 4 : 256 * 4 - 1]) == tuple(lut("ember")[255])


DISCREPANCY_SPEC = {
    "model": "repressilator",
    "params": {
        "alpha": 216.0,
        "alpha0": 0.216,
        "n_hill": 2.0,
        "beta": 1.0,
        "Omega": 6.0,
        "t_max": 3.0,
    },
    "replicates": 3,
    "seed": 4,
    "max_steps": 40000,
    "record_every": 1,
    "limit": {"t_max": 3.0, "dt": 0.005},
}


def test_discrepancy_reports_a_finite_positive_distance():
    report = bridge.discrepancy_to_limit(DISCREPANCY_SPEC, n_grid=60)
    assert report["available"] is True
    assert report["truncated"] == []
    assert report["D"] > 0.0
    assert len(report["per_replicate"]) == DISCREPANCY_SPEC["replicates"]
    # Distinct replicates give distinct distances; identical ones would mean the
    # measurement is reading one trajectory several times.
    assert len(set(report["per_replicate"])) == len(report["per_replicate"])
    assert _strict_json(report)


def test_discrepancy_refuses_a_recording_too_coarse_for_its_grid():
    """The trap, and why it has to raise rather than return a plausible number.

    ``_per_replicate_discrepancy`` samples by step-hold, which is exact for an
    SSA only while the recording is denser than the comparison grid. Coarser than
    that and every grid point reads a stale value — and that error is
    **Omega-independent**, so it does not cancel between system sizes. It sits
    under D as a floor and flattens the scaling, which is the one thing the
    measurement exists to show. Measured on the repressilator at Omega = 5
    against a 200-point grid, D read 8.72 with 15 578 recorded points and
    8.83 / 10.28 / 13.00 / 21.15 with 79 / 17 / 9 / 5.
    """
    with pytest.raises(ValueError, match="Omega-INDEPENDENT"):
        bridge.discrepancy_to_limit({**DISCREPANCY_SPEC, "record_every": 5000}, n_grid=60)


def test_discrepancy_reports_replicates_that_never_reached_the_horizon():
    """A truncated replicate holds its last value across the rest of the grid.

    That turns "the run stopped" into "the run was far from the limit" — a large
    discrepancy with no error anywhere. It is reported rather than averaged in
    silently.
    """
    report = bridge.discrepancy_to_limit({**DISCREPANCY_SPEC, "max_steps": 500}, n_grid=60)
    assert report["truncated"] == [0, 1, 2]


def test_discrepancy_refuses_a_model_with_no_limit_to_be_far_from():
    report = bridge.discrepancy_to_limit({**WF_SPEC, "limit": {"t_max": 5.0}})
    assert report["available"] is False
    assert "no ODE limit" in report["reason"]


def test_compute_seconds_accumulates_and_only_counts_stepping():
    run = bridge.Run("test", bridge.experiment_from_spec(REPRESSILATOR_SPEC))
    assert run.compute_seconds == 0.0
    first = run.advance(500)["compute_seconds"]
    assert first > 0.0
    # Draining is not stepping, and must not be charged as it: the whole point of
    # the figure is to compare like with like across the worker and main thread.
    run.drain()
    assert run.status()["compute_seconds"] == first
    assert run.advance(500)["compute_seconds"] > first
