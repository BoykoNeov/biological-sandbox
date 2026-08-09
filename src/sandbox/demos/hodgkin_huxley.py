"""Hodgkin-Huxley demo — Phase 2a end to end, with its categories kept apart.

Run it:  ``uv run python -m sandbox.demos.hodgkin_huxley``

Four acts, in the order the project's argument runs:

1. **The exact anchors.** The voltage clamp has a closed-form solution and the
   resting fixed point is found by root-finding rather than integration. Both go
   through :func:`~sandbox.core.validation.validate`. Nothing below is worth
   reading if these are red.
2. **The literature-anchored numbers** — rheobase, the f-I curve, spike shape,
   refractory period, the subcritical-Hopf bistable band. These are the famous
   ones, and they live **here rather than in the test suite** on purpose. See the
   note below.
3. **Channel noise.** Finite channel counts against the deterministic limit, in
   both regimes: the subthreshold one where the scaling law holds, and the
   spiking one where it emphatically does not.
4. **The law.** ``D(N) ~ N^{-1/2}`` from
   :func:`~sandbox.core.convergence.convergence_report`, plotted log-log.

**Why act 2 is not in the suite.** Phase 2 splits checkable claims into three
categories: *A* exact analytic (the clamp, the fixed point), *B* an asymptotic
law with a confidence interval (the ``N^{-1/2}`` slope), and *C*
literature-anchored — a number from the published record that no closed form
produces. Reproducing a category-C number is **evidence, not proof**, and the
project's rule is that it never enters ``analytic_predictions`` and never becomes
an assertion's bound. "Rheobase is in [6.2, 6.5]" would be asserting the
0.5-resolution of a planning scan, brittle to any change of horizon or spike
threshold. What the suite *does* assert about act 2 is structural only: that
spike count is monotone non-decreasing in ``I``, that some current gives zero
spikes and some gives more, and that a spike overshoots 0 mV and then
after-hyperpolarises.

**This demo runs reduced configs.** The convergence sweep uses 4 sizes and 6
replicates instead of the suite's 5 and 16, so it finishes in seconds rather than
half a minute; it was seed-checked (PASS at seeds 0, 1, 2) so it does not print a
red check on an unlucky draw, but its error bars are honestly wider and it says
so. The authoritative run is ``tests/test_hh_stochastic.py``.

Figures are written to the current directory (``*.png`` is gitignored). All
printed output is ASCII-only: this project is developed on a Windows console
(cp1252) where a stray Unicode glyph raises.
"""

from __future__ import annotations

import numpy as np

import sandbox.models  # noqa: F401  (registers models)
from sandbox.core.convergence import convergence_report
from sandbox.core.ode import integrate_rk4
from sandbox.core.protocol import Experiment
from sandbox.core.recorder import Trajectory, run_replicate
from sandbox.core.validation import validate
from sandbox.models.hh_stochastic import MODEL as STOCHASTIC_MODEL
from sandbox.models.hh_stochastic import HHStochasticParams
from sandbox.models.hh_voltage_clamp import HHVoltageClampParams
from sandbox.models.hodgkin_huxley import (
    MODEL as DETERMINISTIC_MODEL,
)
from sandbox.models.hodgkin_huxley import (
    STATE_KEYS,
    HHParams,
    fixed_point_eigenvalues,
    hh_rhs,
    n_hh_steps,
    resting_state,
)

# The subthreshold convergence config, reduced from tests/test_hh_stochastic.py
# (5 sizes x 16 replicates). The sweep still starts at 16000: below that the
# membrane spikes spontaneously and D(N) obeys no scaling law at all.
_DEMO_N = [16_000.0, 64_000.0, 256_000.0, 1_024_000.0]
_DEMO_REPLICATES = 6
_DEMO_T_MAX = 40.0
_DEMO_DT = 0.025
_DEMO_STEPS = round(_DEMO_T_MAX / _DEMO_DT)
_DEMO_GRID = np.arange(0, _DEMO_STEPS + 1, 8, dtype=float) * _DEMO_DT

_FI_CURRENTS = [0.0, 2.0, 4.0, 6.0, 6.5, 8.0, 10.0, 14.0, 20.0, 30.0]
# Repetitive firing is measured AFTER an onset transient is discarded, and that is
# not a detail: a sub-rheobase step still fires an onset spike or two before the
# cell falls silent. Counting over the whole window, this demo first reported
# "rheobase in [2, 4]" -- contradicting both the textbook 6.2-6.5 and the planning
# scan, which dropped 100 ms for exactly this reason.
_SPIKE_WINDOW = 100.0
_TRANSIENT = 50.0

# Channel noise is shown just BELOW rheobase, where a threshold system is most
# sensitive to it, not deep in the firing regime where the drive swamps it.
_NOISY_CURRENT = 5.0
_NOISE_WINDOW = 80.0


def _rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def _run_deterministic(params: HHParams) -> tuple[np.ndarray, np.ndarray]:
    traj = run_replicate(
        DETERMINISTIC_MODEL, params, np.random.default_rng(0), max_steps=n_hh_steps(params) + 10
    )
    times, series = traj.as_arrays()
    return times, np.stack([series[k] for k in STATE_KEYS], axis=1)


def _upward_crossings(v: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Indices where ``V`` crosses ``threshold`` going up — one per action potential."""
    return np.flatnonzero((v[:-1] < threshold) & (v[1:] >= threshold))


def _sustained_spikes(times: np.ndarray, v: np.ndarray, after: float = _TRANSIENT) -> int:
    """Spikes in the steady portion only, discarding the onset transient.

    A step of current fires an onset spike even well below the current needed for
    *repetitive* firing, so counting from ``t = 0`` measures excitability rather
    than rheobase and reads 3x too low.
    """
    crossings = _upward_crossings(v)
    return int(np.sum(times[crossings] >= after))


# ---------------------------------------------------------------------------
# Act 1 -- the exact anchors
# ---------------------------------------------------------------------------


def _validate_anchors() -> bool:
    _rule("1. Category A: the exact analytic anchors")

    clamp = validate(
        Experiment(
            model="hh_voltage_clamp",
            params={"v_hold": -65.0, "v_clamp": 0.0, "t_max": 10.0, "dt": 0.01},
            replicates=2,
            observables=("m", "h", "n"),
            seed=0,
            max_steps=2_000,
        ),
        lambda d: HHVoltageClampParams(**d),
        z=4.0,
        sem_floor=1e-7,
    )
    print("voltage clamp -65 -> 0 mV: gating decouples into dx/dt = (x_inf - x)/tau,")
    print("solved exactly by x(t) = x_inf + (x0 - x_inf) exp(-t/tau).")
    print(clamp)

    rest = validate(
        Experiment(
            model="hodgkin_huxley",
            params={"i_ext": 0.0, "t_max": 50.0, "dt": 0.01, "v0": -64.996379331},
            replicates=2,
            observables=STATE_KEYS,
            seed=0,
            max_steps=6_000,
        ),
        lambda d: HHParams(**d),
        z=4.0,
        sem_floor=1e-7,
    )
    print("\nresting fixed point: root-found algebraically, then reached by integrating")
    print("-- two different code paths that have to agree.")
    print(rest)

    print("\nNeither of these can catch a wrong g_Na or m^2 in place of m^3: both paths")
    print("would move consistently and the root would still be a root. That is caught by")
    print("a separately hand-transcribed textbook RHS in tests/test_hodgkin_huxley.py.")
    return clamp.passed and rest.passed


# ---------------------------------------------------------------------------
# Act 2 -- category C, labelled as such
# ---------------------------------------------------------------------------


def _report_literature_anchored() -> dict[str, object]:
    _rule("2. Category C: LITERATURE-ANCHORED -- evidence, not proof")
    print("None of the numbers below is asserted anywhere in the test suite, and none")
    print("appears in analytic_predictions. They are reproductions of published")
    print("measurements, reported so they can be compared by eye.\n")

    window = _SPIKE_WINDOW - _TRANSIENT
    counts, onset_counts = [], []
    for i_ext in _FI_CURRENTS:
        times, y = _run_deterministic(HHParams(i_ext=i_ext, v0=-65.0, t_max=_SPIKE_WINDOW, dt=0.01))
        counts.append(_sustained_spikes(times, y[:, 0]))
        onset_counts.append(len(_upward_crossings(y[:, 0])))
    rates = [1000.0 * c / window for c in counts]

    print(f"f-I curve, counted over t in [{_TRANSIENT:g}, {_SPIKE_WINDOW:g}] ms:")
    print(f"  {'I (uA/cm^2)':>12} {'sustained':>10} {'rate (Hz)':>10} {'incl. onset':>12}")
    for i_ext, count, rate, onset in zip(_FI_CURRENTS, counts, rates, onset_counts, strict=True):
        print(f"  {i_ext:>12.1f} {count:>10d} {rate:>10.1f} {onset:>12d}")
    print("  The last column is why the transient is dropped: sub-rheobase currents")
    print("  still fire an onset spike, and counting those put rheobase at [2, 4].")

    quiet = [i for i, c in zip(_FI_CURRENTS, counts, strict=True) if c == 0]
    firing = [i for i, c in zip(_FI_CURRENTS, counts, strict=True) if c > 0]
    print(f"\nrheobase is bracketed by this scan: [{max(quiet):g}, {min(firing):g}] uA/cm^2")
    print("  (textbook ~6.2-6.5; the bracket is only as fine as the currents scanned,")
    print("   which is exactly why it is not an assertion's bound)")

    # Spike shape, from the first action potential at a clearly suprathreshold current.
    times, y = _run_deterministic(HHParams(i_ext=20.0, v0=-65.0, t_max=30.0, dt=0.01))
    v = y[:, 0]
    crossings = _upward_crossings(v)
    first, second = crossings[0], crossings[1]
    peak = int(first + np.argmax(v[first : first + 500]))
    trough = int(peak + np.argmin(v[peak : peak + 1500]))
    half = 0.5 * (v[peak] + v[first])
    above = np.flatnonzero(v[first : trough + 1] >= half)
    print("\nspike shape at I = 20 uA/cm^2:")
    print(f"  peak                {v[peak]:+8.2f} mV at t = {times[peak]:.2f} ms")
    print(f"  after-hyperpolarise {v[trough]:+8.2f} mV at t = {times[trough]:.2f} ms")
    print(f"  half-width          {(above[-1] - above[0]) * 0.01:8.2f} ms")
    print(f"  inter-spike interval{(times[second] - times[first]):8.2f} ms")
    print("  (published squid-axon spikes overshoot to ~+40 mV and undershoot to ~-75)")

    print("\nsubcritical Hopf -- the bistable band (category C, and the interesting one):")
    print(f"  {'I':>6} {'max Re eig':>12} {'fixed point':>13} {'spikes':>7}")
    band = []
    for i_ext in (0.0, 6.0, 6.5, 7.0, 8.0, 10.0, 12.0):
        params = HHParams(i_ext=i_ext, v0=-65.0, t_max=_SPIKE_WINDOW, dt=0.01)
        worst = float(np.max(fixed_point_eigenvalues(params).real))
        times_i, y_i = _run_deterministic(params)
        n_spikes = _sustained_spikes(times_i, y_i[:, 0])
        stable = "stable" if worst < 0 else "UNSTABLE"
        band.append((i_ext, worst, stable, n_spikes))
        print(f"  {i_ext:>6.1f} {worst:>+12.5f} {stable:>13} {n_spikes:>7d}")
    print("  The fixed point stays stable while the cell already fires: a stable rest")
    print("  state and a stable limit cycle coexist. This is why analytic_predictions")
    print("  raises past the Hopf rather than returning a fixed point nothing settles to.")

    return {"currents": _FI_CURRENTS, "counts": counts, "rates": rates, "band": band}


# ---------------------------------------------------------------------------
# Act 3 -- channel noise in both regimes
# ---------------------------------------------------------------------------


def _stochastic_replicates(params: HHStochasticParams, replicates: int) -> list[Trajectory]:
    from sandbox.core.rng import spawn_rngs

    return [
        run_replicate(
            STOCHASTIC_MODEL,
            params,
            rng,
            max_steps=round(params.t_max / params.dt) + 10,
        )
        for rng in spawn_rngs(np.random.SeedSequence(0), replicates)
    ]


def _channel_noise() -> dict[str, object]:
    _rule("3. Channel noise: the same finite N, in two regimes")
    v_star = float(resting_state(HHParams(i_ext=0.0, t_max=1.0, dt=_DEMO_DT))[0])

    print("Subthreshold (I = 0, started at the fixed point). V fluctuates about rest and")
    print("the fluctuation scale falls as 1/sqrt(N) -- this is where the law is checked.")
    sub = {}
    for n_channels in (16_000.0, 1_024_000.0):
        params = HHStochasticParams(
            i_ext=0.0, t_max=_DEMO_T_MAX, dt=_DEMO_DT, v0=v_star, n_channels=n_channels
        )
        trajectories = _stochastic_replicates(params, 6)
        excursions = [float(np.abs(np.asarray(t.series["V"]) - v_star).max()) for t in trajectories]
        sub[n_channels] = trajectories
        print(
            f"  N = {n_channels:>10,.0f}:  max |V - V*| over 6 replicates "
            f"= {max(excursions):.3f} mV"
        )

    print(f"\nNear threshold (I = {_NOISY_CURRENT:g}, sub-rheobase). Here low N changes the")
    print("spike COUNT -- spontaneous action potentials the deterministic cell never fires.")
    print("A 100 mV discrepancy from a threshold crossing obeys no N^{-1/2} law, which is")
    print("why the slope check lives in the regime above and this one is only a picture.")
    print("(Tried first at I = 20, deep in the firing regime, where even N = 2000 gave")
    print(" [5, 5, 4, 5] against the limit's 5: the drive swamps the noise and the")
    print(" phenomenon is invisible. It lives at threshold, not in the middle.)")
    spiking = {}
    for n_channels in (1_000.0, 20_000.0, 5_000_000.0):
        params = HHStochasticParams(
            i_ext=_NOISY_CURRENT, t_max=_NOISE_WINDOW, dt=0.01, v0=-65.0, n_channels=n_channels
        )
        trajectories = _stochastic_replicates(params, 3)
        counts = [len(_upward_crossings(np.asarray(t.series["V"]))) for t in trajectories]
        spiking[n_channels] = trajectories
        print(f"  N = {n_channels:>10,.0f}:  spikes per replicate = {counts}")

    ode_params = HHParams(i_ext=_NOISY_CURRENT, v0=-65.0, t_max=_NOISE_WINDOW, dt=0.01)
    t_ode, y_ode = integrate_rk4(
        hh_rhs(ode_params),
        DETERMINISTIC_MODEL.initial_concentrations(ode_params),
        _NOISE_WINDOW,
        0.01,
    )
    print(f"  deterministic limit (N -> infinity): {len(_upward_crossings(y_ode[:, 0]))} spikes")
    return {"subthreshold": sub, "spiking": spiking, "ode": (t_ode, y_ode), "v_star": v_star}


# ---------------------------------------------------------------------------
# Act 4 -- the scaling law
# ---------------------------------------------------------------------------


def _convergence(v_star: float):
    _rule("4. Category B: D(N) ~ N^{-1/2}, the Kurtz / van-Kampen law")
    report = convergence_report(
        "hh_stochastic",
        {"i_ext": 0.0, "dt": _DEMO_DT, "v0": v_star},
        lambda d: HHStochasticParams(**d),
        omegas=_DEMO_N,
        t_max=_DEMO_T_MAX,
        dt=0.005,
        replicates=_DEMO_REPLICATES,
        grid=_DEMO_GRID,
        require_exact_grid=True,
        observable_keys=STATE_KEYS,
        compare_keys=("V",),
        omega_key="n_channels",
        seed=0,
        z=3.0,
        n_bootstrap=200,
        max_steps=_DEMO_STEPS + 10,
    )
    print(report)
    print("\nOnly V enters the discrepancy: a channel-state model has occupancy counts,")
    print("whose limits are m^3 h and n^4 -- there is no m, h or n to compare, and")
    print("blending millivolts with dimensionless gates in one L1 average would be")
    print("unprincipled anyway.")
    print("\nREDUCED CONFIG: 4 sizes x 6 replicates against the suite's 5 x 16, so the")
    print("slope error bar here is wider than the authoritative one. Seed-checked at")
    print("0, 1, 2. tests/test_hh_stochastic.py carries the real check and its teeth.")
    return report


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _figures(category_c: dict, noise: dict, report) -> None:
    _rule("Figures")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib not installed; skipping figures (uv sync --extra viz)")
        return

    from sandbox.viz.backends.matplotlib_backend import plot_convergence, plot_replicates

    # 1. deterministic spike trains across the f-I range
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for ax, i_ext in zip(axes, (4.0, 8.0, 20.0), strict=True):
        times, y = _run_deterministic(HHParams(i_ext=i_ext, v0=-65.0, t_max=_SPIKE_WINDOW, dt=0.01))
        ax.plot(times, y[:, 0], color="C0", linewidth=1.0)
        ax.set_ylabel("V (mV)")
        ax.set_title(f"I = {i_ext:g} uA/cm^2", fontsize=9)
    axes[-1].set_xlabel("t (ms)")
    fig.suptitle("Hodgkin-Huxley: sub-rheobase, bistable band, repetitive firing")
    fig.tight_layout()
    fig.savefig("hh_spike_train.png", dpi=120)
    plt.close(fig)
    print("wrote hh_spike_train.png")

    # 2. the f-I curve, labelled
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(category_c["currents"], category_c["rates"], "o-", color="C0")
    ax.set_xlabel("injected current I (uA/cm^2)")
    ax.set_ylabel("firing rate (Hz)")
    ax.set_title("f-I curve -- CATEGORY C (literature-anchored, not asserted)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("hh_fi_curve.png", dpi=120)
    plt.close(fig)
    print("wrote hh_fi_curve.png")

    # 3. channel noise: subthreshold (where the law lives) and spiking (where it does not)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, (n_channels, trajectories) in zip(
        axes[0], sorted(noise["subthreshold"].items()), strict=True
    ):
        plot_replicates(
            trajectories,
            "V",
            deterministic=([0.0, _DEMO_T_MAX], [noise["v_star"]] * 2),
            title=f"subthreshold, N = {n_channels:,.0f}",
            ax=ax,
            alpha=0.6,
        )
        ax.set_ylim(noise["v_star"] - 4.0, noise["v_star"] + 4.0)
        ax.set_xlabel("t (ms)")
        ax.set_ylabel("V (mV)")
    t_ode, y_ode = noise["ode"]
    spiking = sorted(noise["spiking"].items())
    for ax, (n_channels, trajectories) in zip(axes[1], spiking[:2], strict=True):
        plot_replicates(
            trajectories,
            "V",
            deterministic=(t_ode, y_ode[:, 0]),
            title=f"near threshold, N = {n_channels:,.0f} -- spike COUNT differs",
            ax=ax,
            alpha=0.6,
        )
        ax.set_xlabel("t (ms)")
        ax.set_ylabel("V (mV)")
    fig.suptitle("Channel noise: the scaling law holds above, and does not below")
    fig.tight_layout()
    fig.savefig("hh_channel_noise.png", dpi=120)
    plt.close(fig)
    print("wrote hh_channel_noise.png")

    # 4. the N-scaling figure
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_convergence(
        report.omegas,
        report.discrepancy,
        sem=report.discrepancy_sem,
        fit_mask=report.fit_mask,
        slope=report.slope,
        slope_se=report.slope_se,
        title="Channel-noise convergence: D(N) vs the -1/2 law (reduced config)",
        ax=ax,
    )
    ax.set_xlabel("channel count N")
    ax.set_ylabel("discrepancy D(N) in mV")
    fig.tight_layout()
    fig.savefig("hh_convergence.png", dpi=120)
    plt.close(fig)
    print("wrote hh_convergence.png")


def main() -> int:
    anchors_ok = _validate_anchors()
    category_c = _report_literature_anchored()
    noise = _channel_noise()
    report = _convergence(noise["v_star"])
    _figures(category_c, noise, report)

    _rule("Summary")
    print(f"category A (exact closed forms)      : {'PASS' if anchors_ok else 'FAIL'}")
    print(f"category B (D(N) ~ N^-1/2 slope)     : {'PASS' if report.passed else 'FAIL'}")
    print("category C (rheobase, f-I, spike shape): reported above, asserted nowhere")
    return 0 if (anchors_ok and report.passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
