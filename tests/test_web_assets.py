"""The front-end's static assets, checked natively — no browser required.

Every bug this module exists to catch was found by opening a browser and looking,
which is the expensive way to find any of them:

- A **backtick** inside the Python that ``worker.js`` embeds in a JS template
  literal ends the string, and the whole worker fails to parse with a syntax
  error pointing at a line nowhere near the cause. The page then hangs on
  ``starting Python in a background thread ...`` with nothing in the console,
  because the failure killed the module script before it could log.
- A **preset naming a model that cannot run it**: a params field missing, a
  horizon absent, a model not registered.
- A **preset declaring an ODE limit that cannot be integrated.** The bridge
  overlays the limit column by column and refuses anything that does not
  correspond one-to-one with the observables, so a model reporting *derived*
  observables alongside its state (``glv_stochastic`` reports ``total_biomass``
  and ``n_survivors`` on top of three species, against a three-component ODE)
  can never have one drawn. The refusal is correct; declaring the block anyway
  is the error.
- A **step budget below what the model needs.** This is the one with teeth: a
  run that stops early leaves the deterministic limit alone on the right of the
  frame, and that reads as perfect agreement rather than as absence. The project
  has shipped that figure once already.

The presets are data (``web/presets.json``) rather than page code for the same
reason ``web/conformance.json`` is: keeping the specs out of both programs is
what makes the browser and the test suite provably the same question.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import sandbox.models  # noqa: F401  (registers models)
from sandbox.core.registry import available, get_params_factory
from sandbox.web import bridge

WEB = Path(__file__).resolve().parents[1] / "web"
PRESETS = json.loads((WEB / "presets.json").read_text(encoding="utf-8"))["presets"]
PRESET_NAMES = sorted(PRESETS)


def _embedded_python(source: str) -> list[str]:
    """Every ``py.runPython(`...`)`` block in a JS file, as source strings.

    Deliberately a dumb split on the backtick rather than a JS parse: the failure
    being guarded against IS a stray backtick, and a lenient parser that recovered
    from one would defeat the check.
    """
    blocks = []
    for chunk in source.split("py.runPython(`")[1:]:
        end = chunk.find("`")
        assert end != -1, "an embedded runPython block is not closed by a backtick"
        blocks.append(chunk[:end])
    return blocks


def test_the_python_embedded_in_the_worker_compiles():
    """A syntax error in here does not fail loudly — it hangs the page.

    The shim is a Python string inside a JavaScript template literal inside a Web
    Worker, so a mistake in it surfaces as a boot that never finishes, three
    contexts away from the cause.
    """
    blocks = _embedded_python((WEB / "worker.js").read_text(encoding="utf-8"))
    assert blocks, "no embedded Python found in worker.js — has it been restructured?"
    for index, block in enumerate(blocks):
        compile(block, f"<worker.js runPython block {index}>", "exec")


def test_the_embedded_python_defines_every_entry_point_the_worker_calls():
    """``pyCall('web_x')`` with no ``def web_x`` is a runtime error per command.

    Cheap to check here and otherwise only found by pressing the button that uses
    it, which is how the field path stayed broken: no page exercised it.
    """
    source = (WEB / "worker.js").read_text(encoding="utf-8")
    defined = set()
    for block in _embedded_python(source):
        for line in block.splitlines():
            if line.startswith("def web_"):
                defined.add(line[len("def ") :].split("(")[0])

    called = set()
    for chunk in source.split('pyCall("')[1:]:
        called.add(chunk[: chunk.find('"')])

    assert called, "no pyCall sites found in worker.js — has it been restructured?"
    assert called <= defined, f"called but never defined: {sorted(called - defined)}"


@pytest.mark.parametrize("name", PRESET_NAMES)
def test_every_preset_names_a_registered_model_and_builds_its_params(name):
    assert name in available(), f"preset {name!r} names a model that is not registered"
    get_params_factory(name)(dict(PRESETS[name]["params"]))


@pytest.mark.parametrize("name", PRESET_NAMES)
def test_every_preset_actually_runs(name):
    """A short run, not the full one: this asks whether it *starts*.

    The full cost is measured separately by ``web/serve.py --measure-presets``,
    which is where the recorded step counts come from. Running all fourteen to
    termination here would put ~20 s on a suite whose timing this project has
    already been burned by three times.
    """
    preset = PRESETS[name]
    run = bridge.Run(
        "test",
        bridge.experiment_from_spec(
            {
                "model": name,
                "params": preset["params"],
                "replicates": 1,
                "seed": preset.get("seed", 0),
                "max_steps": 200,
                "record_every": 1,
            }
        ),
    )
    run.advance(200)
    status = run.status()
    assert status["steps"][0] > 0, f"{name} took no steps at all"
    assert status["recorded"][0] > 1, f"{name} recorded nothing beyond its initial state"


@pytest.mark.parametrize("name", PRESET_NAMES)
def test_a_declared_limit_can_actually_be_integrated(name):
    """Declaring a limit the bridge will refuse is the error, not the refusal.

    ``glv_stochastic`` is the case: five observables against a three-component
    ODE, because two of them are derived. Its preset therefore declares no limit,
    and this is what keeps that decision from being quietly undone.
    """
    preset = PRESETS[name]
    if "limit" not in preset:
        pytest.skip(f"{name} declares no limit block")
    report = bridge.deterministic_limit(
        {
            "model": name,
            "params": preset["params"],
            "replicates": 1,
            "seed": preset.get("seed", 0),
            "limit": preset["limit"],
        }
    )
    assert report["available"] is True, report.get("reason")
    assert len(report["t"]) > 1


@pytest.mark.parametrize("name", PRESET_NAMES)
def test_the_step_budget_is_above_the_measured_cost(name):
    """The budget must exceed what the model was measured to need.

    Not a style rule. A budget that runs out mid-run stops the replicates early
    and leaves the deterministic limit alone on the right-hand side of the frame,
    which a reader takes as agreement — the specific wrong figure this project
    shipped once and now sizes budgets to avoid.
    """
    preset = PRESETS[name]
    measured = preset["measured_steps"]
    budget = preset["max_steps"]
    assert budget > measured, (
        f"{name}: budget {budget:,} does not clear its measured {measured:,} steps, "
        "so the run stops early and the picture reads as agreement"
    )


@pytest.mark.parametrize("name", PRESET_NAMES)
def test_the_recorded_density_throws_nothing_away_it_needs(name):
    """Either keep every point, or keep enough of them to draw a curve.

    The first draft of this asserted a flat floor of a hundred recorded points and
    ``adaptive_dynamics`` failed it at 57 — correctly, and the *test* was wrong.
    Its trait-substitution sequence is a jump process with 57 events in the whole
    run, so at ``record_every = 1`` it is already keeping everything that exists.
    A floor on the number of points is a claim about the model; what a preset can
    actually be held to is a claim about the *thinning*, which is the only part it
    chooses.
    """
    preset = PRESETS[name]
    every = preset["record_every"]
    points = preset["measured_steps"] / every
    assert every == 1 or points >= 100, (
        f"{name}: record_every={every} thins a {preset['measured_steps']:,}-step run "
        f"down to {points:.0f} recorded points, which is too few to draw"
    )


def test_the_presets_cover_every_built_in_model():
    """A model with no preset still runs on its defaults, but with no measured budget.

    The page says so out loud when that happens. This is here so that adding a
    model is a decision about the picker rather than a silent omission from it.

    **Against the package, not against the registry.** The first version compared
    the presets to ``available()`` and passed alone while failing in the full
    suite: other test modules register deliberately-broken "teeth" models into the
    same global registry, so ``available()`` is larger there than it is here. What
    the picker owes a preset to is the *built-in* set, which is the model modules
    in ``sandbox.models`` — and that is derived from the package rather than from
    whatever has been registered by the time this happens to run.

    The set is the **intersection** of the two, which is what makes it robust in
    both directions at once: a teeth model registered by another test module is
    not a module of this package, and ``gillespie`` and ``hh_rates`` are modules
    of this package that register no model — they are the SSA engine and the rate
    functions. Neither belongs in the picker, and neither survives the
    intersection.
    """
    import pkgutil

    modules = {
        module.name for module in pkgutil.iter_modules(sandbox.models.__path__) if not module.ispkg
    }
    built_in = modules & set(available())
    assert sorted(PRESETS) == sorted(built_in)


# ---------------------------------------------------------------------------
# The on-demand figure export
#
# Guarded, because matplotlib is an optional extra and the core must not depend
# on it -- which is the same reason it is absent from the browser bundle and
# staged separately.
# ---------------------------------------------------------------------------

matplotlib = pytest.importorskip("matplotlib", reason="viz extra not installed")
matplotlib.use("Agg")

import matplotlib.image as mpimg  # noqa: E402  (must follow the backend selection)
import numpy as np  # noqa: E402

FIGURE_SPEC = {
    "model": "repressilator",
    "params": {
        "alpha": 216.0,
        "alpha0": 0.216,
        "n_hill": 2.0,
        "beta": 1.0,
        "Omega": 8.0,
        "t_max": 3.0,
    },
    "replicates": 3,
    "seed": 11,
    "max_steps": 40_000,
    "record_every": 5,
    "limit": {"t_max": 3.0, "dt": 0.01},
}


@pytest.fixture
def finished_run():
    status = bridge.SESSION.create(FIGURE_SPEC)
    run = bridge.SESSION.get(status["run_id"])
    while not run.finished:
        run.advance(8192)
    yield status["run_id"]
    bridge.SESSION.close(status["run_id"])


def _ink(png_bytes: np.ndarray) -> dict[str, int]:
    """Count reddish and bluish pixels in a rendered PNG.

    This is the whole point of the test below and not decoration. The failure this
    project actually shipped was a **valid PNG with correct axes and no data**,
    from a series filter that matched nothing. Every structural assertion passed
    it: the file existed, it opened, it had the right size and the right labels.
    Only the ink was missing. So the ink is what gets counted -- blue for the
    stochastic replicates (matplotlib's C0) and red for the deterministic limit
    (C3), which are the two things the figure claims to show.

    The test is on the *difference* between channels, not on absolute levels. A
    first draft required ``b > 0.5 and r < 0.65`` and found only 106 blue pixels
    in a figure full of them: the replicates are drawn at alpha 0.35, so they
    blend toward white and land near ``(0.77, 0.85, 0.93)`` -- unmistakably blue
    to the eye and failing an absolute red-channel bound.

    The thresholds sit in a **measured** gap rather than a guessed one. Rendering
    the same panel four ways gives blue/red counts of ``2412/3271`` (both drawn),
    ``0/3457`` (a run that recorded nothing -- the historical failure exactly),
    ``3845/0`` (no limit) and ``0/0`` (neither). The empty cases are not merely
    small, they are zero.
    """
    import io

    image = mpimg.imread(io.BytesIO(png_bytes.tobytes()))
    r, b = image[..., 0], image[..., 2]
    return {
        "red": int(np.count_nonzero(r - b > 0.20)),
        "blue": int(np.count_nonzero(b - r > 0.05)),
    }


def test_the_exported_figure_actually_contains_the_data(finished_run):
    """A valid PNG with correct axes and no data is the failure to beat."""
    limit = bridge.deterministic_limit(FIGURE_SPEC)
    data, meta = bridge.figure_png(finished_run, ["x_m1"], limit=limit)

    assert data.tobytes()[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    assert meta["bytes"] > 1000
    ink = _ink(data)
    assert ink["blue"] > 500, f"the stochastic replicates are not drawn: {ink}"
    assert ink["red"] > 500, f"the deterministic limit is not drawn: {ink}"


def test_a_run_that_recorded_nothing_produces_a_visibly_empty_figure():
    """The tooth for the test above, and the project's actual historical bug.

    A run created but never advanced has no recorded points, so the panel is axes
    and a limit line and nothing else. Every structural property of a good figure
    holds: it is a valid PNG, the axes are right, the limit is drawn, the labels
    are correct. Only the data is missing — and the ink count is what notices,
    reading exactly zero rather than merely small.
    """
    status = bridge.SESSION.create(FIGURE_SPEC)
    try:
        limit = bridge.deterministic_limit(FIGURE_SPEC)
        data, meta = bridge.figure_png(status["run_id"], ["x_m1"], limit=limit)
        assert data.tobytes()[:8] == b"\x89PNG\r\n\x1a\n"  # a perfectly valid PNG
        assert meta["limit_drawn_for"] == ["x_m1"]  # with the limit really drawn
        ink = _ink(data)
        assert ink["red"] > 500, "the limit should still be drawn"
        assert ink["blue"] == 0, f"there is no replicate data, so there is no blue: {ink}"
    finally:
        bridge.SESSION.close(status["run_id"])


def test_the_figure_reports_which_panels_got_a_limit_rather_than_which_asked(finished_run):
    """A caption saying "with the limit" over a panel without one is the recurring bug."""
    limit = bridge.deterministic_limit(FIGURE_SPEC)
    _, with_limit = bridge.figure_png(finished_run, ["x_m1", "x_p3"], limit=limit)
    assert with_limit["limit_drawn_for"] == ["x_m1", "x_p3"]

    _, without = bridge.figure_png(finished_run, ["x_m1"], limit=None)
    assert without["limit_drawn_for"] == []


def test_the_limit_columns_are_matched_by_name_not_by_position(finished_run):
    """On a symmetric limit cycle the wrong column is the right shape at the wrong phase.

    Which is why it would look entirely plausible. Reversing the order of the
    limit's series must change nothing at all; an implementation indexing into the
    ODE's columns would render a different picture and still look fine.
    """
    limit = bridge.deterministic_limit(FIGURE_SPEC)
    shuffled = {**limit, "series": dict(reversed(list(limit["series"].items())))}

    straight, _ = bridge.figure_png(finished_run, ["x_m1"], limit=limit)
    reversed_, _ = bridge.figure_png(finished_run, ["x_m1"], limit=shuffled)
    assert straight.tobytes() == reversed_.tobytes()


def test_an_unknown_observable_is_refused_with_the_list_of_real_ones(finished_run):
    with pytest.raises(KeyError, match="x_m1"):
        bridge.figure_png(finished_run, ["m1"])


def test_the_import_is_priced_separately_from_the_drawing(finished_run):
    """Two different questions: what the first figure costs, and what the next one does.

    A single number is dominated by the import and misprices both -- and the
    import is the entire reason matplotlib is staged rather than bundled.
    """
    _, meta = bridge.figure_png(finished_run, ["x_m1"])
    assert meta["import_seconds"] >= 0.0
    assert meta["render_seconds"] > 0.0


def test_a_stopped_run_carries_its_own_warning_into_the_png():
    """A PNG leaves the page without the page's caption, so it must carry this itself.

    The pages say whether every replicate reached the horizon, and separate "you
    stopped it" from "the budget ran out". A figure exported from such a run is a
    standalone file with none of that: the replicates end part-way, the limit runs
    to the full horizon, and the right of every panel is the limit alone. That is
    verbatim the wrong figure this project already shipped once — *"the right of
    the frame showed the limit alone and read as perfect agreement"* — and an
    export re-opens it unless the figure says so in its own ink.
    """
    truncated = {**FIGURE_SPEC, "max_steps": 300}  # nowhere near the horizon
    status = bridge.SESSION.create(truncated)
    run = bridge.SESSION.get(status["run_id"])
    run.advance(300)
    try:
        limit = bridge.deterministic_limit(truncated)
        data, meta = bridge.figure_png(status["run_id"], ["x_m1"], limit=limit)
        assert meta["all_reached"] is False
        assert meta["warned_in_figure"] is True
        assert meta["reached_t"] < truncated["params"]["t_max"]
        # The warning is drawn, not merely reported: the shaded region and the
        # note put ink on the page that a full run does not have.
        assert _ink(data)["red"] > 500  # the limit is still drawn
    finally:
        bridge.SESSION.close(status["run_id"])


def test_a_complete_run_is_not_warned_about(finished_run):
    """The tooth for the test above: a warning on every figure warns about nothing."""
    limit = bridge.deterministic_limit(FIGURE_SPEC)
    _, meta = bridge.figure_png(finished_run, ["x_m1"], limit=limit)
    assert meta["all_reached"] is True
    assert meta["warned_in_figure"] is False
