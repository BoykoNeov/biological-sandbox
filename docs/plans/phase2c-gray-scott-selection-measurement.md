# Can Gray-Scott carry the wavelength-selection claim? — a measurement

`docs/plans/phase2c-plan.md` left one deferral with a *trigger condition* rather
than a cost blocker:

> **Promoting the spectral helpers to `core/`.** They live in the model until a
> second model wants them; Gray-Scott's genuine Turing sliver is the candidate, and
> that is a later decision with a real caller behind it.

This is that decision, taken the way this project takes them: measure first, decide
after. **The answer is no, and the reason is structural rather than statistical** —
Gray-Scott's Turing bifurcation at its own validated Turing point is **subcritical**,
so the state that emerges is a large-amplitude localized structure with a broad
spectrum, and linear stability analysis sets no wavelength for it to have predicted.

The finding is one layer inside the one Phase 2 already made. Phase 2 found that at
*Pearson's* parameters there is no real non-trivial homogeneous state, so those
patterns are excitable structures rather than Turing patterns. This says that even at
the parameters where the homogeneous state **does** exist, **is** stable without
diffusion, and **does** have an unstable band at `q > 0` — the three conditions
`test_the_chosen_point_is_a_genuine_turing_point` asserts — the emergent pattern is
still not one linear theory predicts.

**So a genuine Turing instability is necessary but not sufficient. Supercriticality is
the load-bearing precondition, and Phase 2c only ever checked it where it held.**

Everything below was measured with probes under
`M:\claud_projects\temp\gs-selection-slice\` (probe 1-7b), at the module's own
defaults `F = 0.074, k = 0.062, Du = 2e-5, Dv = 1e-5`.

---

## 0. The instrument, scored against a known answer first

Two calibrations, because a probe that has not been scored against a known answer is
not evidence.

**The 1-D right-hand side used throughout is the repo's arithmetic.** Gray-Scott's
shipped model is 2-D only, and nothing validated was modified before the measurement
said whether modifying it was worth it — so the probes write the RHS from the same
`core.laplacian.laplacian` and `core.ode.rk4_step` the model uses, then score it
against the model's own validated `dispersion()` on a seeded mode:

| | value |
|---|---|
| predicted (`dispersion`, `n = 64`, `j = 7`) | `1.4978696675e-02` |
| measured (this probe's 1-D integration) | `1.4978700783e-02` |
| relative error | **`2.7e-07`** |

**The Turing onset is bisected, not read off.** With `Du` fixed, the diffusion ratio
`d = Du/Dv` at which the continuum band closes is

    Dv_c = 1.421687e-05      d_c = 1.406779

so the module's shipped `Dv = 1e-5` sits at **`d/d_c = 1.4217`** — genuinely above
onset, which is the precondition for asking the question at all.

---

## 1. The band is wide, and the operators agree — the first surprise

Phase 2c's worry going in was the opposite one: a razor-thin parameter sliver might
leave too few unstable modes for "the fastest one won" to say anything. It does not.
At `L = 1` the band holds **10-14 modes**, and the continuum band is
`q in [16.05, 76.36]` — a factor of **4.75** in `q`, against Schnakenberg's 17 modes
at a ratio nearer 1.

The problem is the opposite. The two operators pick the **same** winner on every grid
that resolves the pattern:

| `n` (at `L = 1`) | `h` | # unstable | band | stencil `j*` | continuum `j*` | gap | cells/wavelength |
|---|---|---|---|---|---|---|---|
| 24 | 0.04167 | 10 | 3-12 | 9 | 7 | **-2** | 2.67 |
| 32 | 0.03125 | 14 | 3-16 | 8 | 7 | -1 | 4.00 |
| 40 | 0.02500 | 14 | 3-16 | 7 | 7 | 0 | 5.71 |
| 48 | 0.02083 | 12 | 3-14 | 7 | 7 | 0 | 6.86 |
| 64 | 0.01562 | 11 | 3-13 | 7 | 7 | 0 | 9.14 |
| 128 | 0.00781 | 10 | 3-12 | 7 | 7 | 0 | 18.29 |

The lever Phase 2c used — box length — recovers a mode gap, because at fixed spacing
the gap in *modes* is `(q_sten* - q_cont*) L / 2 pi` and grows linearly with `L`:

| spacing | cells/wavelength | gap per unit `L` | `L` for a 2-mode gap |
|---|---|---|---|
| `1/24` | 2.73 | 1.816 | 1.10 |
| `1/32` | 4.16 | 0.708 | 2.82 |
| `1/40` | 5.42 | 0.407 | 4.92 |
| `1/64` | 8.99 | 0.144 | 13.89 |

so e.g. `n = 256, L = 8` gives 4.20 cells per wavelength, 108 unstable modes and a
**5-mode** operator gap. On paper that is a better discrimination than Phase 2c
shipped.

**On paper.** The rate difference between the two winners is only **1.06%** there
(3.39% at the coarsest usable spacing) — the peak is very flat, because the band is
very wide. Whether that flatness kills the claim is not answerable from the table.

---

## 2. The screen: calibrated, confident, and wrong one model over

The decisive quantity is not the operator gap in modes but

    (operator gap) / (per-replicate scatter of the emergent mode)

since the discrimination margin is that ratio times `sqrt(R)`. Under linear growth
from white noise, mode `j` has amplitude `a_j exp(lambda_j T)` with `a_j` iid
Rayleigh, so the selected mode is `argmax_j (log a_j + lambda_j T)` — computable from
the dispersion curve alone, at no cost. Phase 2c **measured** the scatter for
Schnakenberg, so there is a known answer to score it against.

**Calibration arm (Schnakenberg, 2c's shipped config `n = 112`):**

| | screen | measured in 2c |
|---|---|---|
| mean emergent mode | 26.129 | 26.4375 |
| per-replicate sd | 0.728 | 0.629 (from SE 0.1573 at R = 16) |
| margin vs continuum operator at `R = 32` | ~16.3 SE | 9.08 -> 15.95 SE (R = 8 -> 48) |

It also reproduces the collapse 2c measured at finer grids: `gap/sd` falls
`2.88 -> 1.22 -> 0.30` at `n = 112 / 160 / 256`. **The instrument works.**

**Pointed at Gray-Scott it predicts a workable claim:**

| grid | cells/wl | gap | screen sd | gap/sd | predicted margin at `R = 32` |
|---|---|---|---|---|---|
| `n = 192, L = 8` | 2.74 | 14 | 7.58 | 1.85 | 10.45 SE |
| `n = 256, L = 8` | 4.20 | 5 | 3.62 | 1.38 | 7.81 SE |
| `n = 512, L = 16` | 4.16 | 11 | 6.55 | 1.68 | 9.49 SE |
| `n = 512, L = 8` | 8.98 | 1 | 2.87 | 0.35 | 1.97 SE |

Weaker than Schnakenberg's, but well clear of `z = 4`, and collapsing at fine
resolution in exactly the same shape. Every part of that reads like a result.

**Integration says otherwise** (16 replicates, white-noise start, 20 e-folds):

| grid | stencil `j*` | continuum `j*` | measured mean | sd | margin |
|---|---|---|---|---|---|
| `n = 256, L = 8` | 61 | 56 | 56.500 | 6.743 | **-2.37 SE** |
| `n = 512, L = 16` | 123 | 112 | 114.500 | 8.687 | **-2.76 SE** |
| `n = 512, L = 8` | 57 | 56 | 52.000 | 4.066 | -0.98 SE |
| `n = 192, L = 8` | 70 | 56 | 72.062 | 15.190 | +3.69 SE |

A negative margin means the mean sits **closer to the hypothesis the claim exists to
reject**. The screen predicted `+7.8 SE` where integration returns `-2.4`.

**This is the phase's transferable instrument finding.** A screening estimator scored
against a known answer, agreeing with it to 16%, reproducing its resolution
dependence — and then confidently wrong on the next model, because its linearity
assumption fails silently on a subcritical branch. The pattern is this project's
recurring one arriving somewhere new: the check was not loose, it was **inapplicable**,
and nothing in its own output said so.

---

## 3. Before believing that: is it the instrument?

Two things in the integration looked wrong in the way an artefact looks wrong, and
half of Phase 2c's measurement document was already lost to a step-size artefact.

**The horizon changed nothing at all** — bit-identical mean, sd and selected-mode sets
at 12, 30 and 45 e-folds. **And the peak power fraction was 0.05-0.14**, against
Schnakenberg's 0.53-0.90.

Checked, at `n = 256, L = 8`:

* **Step size.** Over a **16x** range (`dt/CFL = 0.360` down to `0.022`, 152 to 2430
  steps) the selected modes are **bit-identical**: `[47, 48, 48, 51, 53, 54, 65, 69]`,
  mean 54.38, peak fraction 0.094. Not a step-size artefact.
* **Stationarity.** Relative change against the final field: `1.4e-01` at `0.25 T`,
  `9.5e-05` at `0.5 T`, `6.9e-06` at `0.75 T`. The field has stopped moving.
* **Far past the horizon.** At 20, 60, 200 and 600 e-folds the modes are identical
  (`[47, 48, 53, 69]`), peak fraction 0.091, amplitude 0.62879 — a genuine steady
  state, not a slow drift.

So the instrument is clean and the broad spectrum is the model's answer. And the
spectrum is the finding — one seed at 200 e-folds:

| mode | 69 | 58 | 65 | 64 | 47 | 39 | 54 | 61 | 60 | 56 | 76 | 73 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| power fraction | .085 | .081 | .078 | .052 | .036 | .035 | .032 | .032 | .029 | .029 | .029 | .028 |

Twelve modes between 2.8% and 8.5%. **There is no wavelength here to predict.** The
"dominant mode" is an arbitrary pick among comparable competitors, which is exactly
why its scatter is 6.7 modes where Schnakenberg's is 0.63.

---

## 4. The structural reason: the bifurcation is subcritical

Phase 2c's precondition for a selection claim was that the bifurcation be
**supercritical** — amplitude continuous from zero, monotone, no jump. Measured
against distance from onset (`n = 256, L = 8`, 40 e-folds, 4 seeds), with
Schnakenberg run through its own shipped model as the reference:

| `d/d_c` | Gray-Scott amplitude | peak fraction | modes (4 seeds) | | Schnakenberg amplitude | peak fraction | modes |
|---|---|---|---|---|---|---|---|
| 1.02 | **0.58038** | 0.078 | 12, 15, 16, 17 | | **0.37890** | **1.000** | 25, 25, 25, 25 |
| 1.05 | 0.59074 | 0.107 | 23, 26, 27, 28 | | 0.57657 | 0.999 | 25, 25, 25, 25 |
| 1.10 | 0.60247 | 0.121 | 33, 34, 37, 38 | | 0.77639 | 0.973 | 25, 25, 25, 25 |
| 1.25 | 0.61665 | 0.124 | 39, 39, 48, 53 | | 1.11742 | 0.924 | 23, 24, 25, 25 |
| 1.50 | 0.63659 | 0.079 | 48, 53, 60, 69 | | 1.45877 | 0.718 | 22, 23, 23, 23 |
| 2.00 | 0.65468 | 0.051 | 82, 103, 120, 121 | | 1.83699 | 0.655 | 21, 21, 22, 22 |

Schnakenberg is textbook supercritical: the amplitude rises continuously from a small
value, the pattern is a **single mode** (fraction 1.000 at onset), and four seeds
agree unanimously.

Gray-Scott's amplitude is **already 0.580 at `d/d_c = 1.02`**, where the linear growth
rate is `9.8e-04`, and only reaches 0.655 by `d/d_c = 2`. It does not go to zero at
onset. Squeezed harder (2 seeds):

| `d/d_c` | `lambda*` | amplitudes | peak fraction | modes | linear fastest mode |
|---|---|---|---|---|---|
| 1.005 | `2.5094e-04` | 0.56732, 0.56732 | 0.077 | 16, 17 | 44 |
| 1.001 | `5.0010e-05` | 0.56615, 0.56615 | 0.059 | 6, 10 | 43 |

Two things there. The amplitude is **bit-identical across seeds** while the mode is
not — the emergent object is a fixed-shape, fixed-amplitude localized structure, and
only *how many* of them appear varies. And at a linear growth rate of `5e-05` the
linear theory's fastest mode is 43 while what appears is 6 to 10. **The emergent state
is not a perturbation of the homogeneous one and is unrelated to the linear
prediction.**

### The hard signature: bistability, measured

"The amplitude did not look small" is an eyeball. The decisive test is hysteresis —
below onset the homogeneous state is linearly stable, so if a *formed* pattern also
survives there, two stable states coexist and the branch is subcritical.

A pattern formed at `d/d_c = 1.50` (amplitude 0.65172), then continued below onset,
with the stencil's own maximum growth rate quoted so "below onset" is not taken on
trust from a continuum bisection:

| `d/d_c` | stencil max rate | from the PATTERN | mode | from NOISE | verdict |
|---|---|---|---|---|---|
| 0.999 | `-5.149e-05` | 0.5840925 | 58 | 0.5655596 | finite noise crosses the threshold |
| 0.990 | `-5.140e-04` | 0.5835758 | 47 | — | pattern survives |
| 0.950 | `-2.730e-03` | 0.5477525 | 12 | **exactly 0** | **BISTABLE** |
| 0.900 | `-6.038e-03` | exactly 0 | — | exactly 0 | homogeneous only |
| 0.800 - 0.500 | | exactly 0 | — | exactly 0 | homogeneous only |

At `d/d_c = 0.95` the homogeneous state is linearly stable and a white-noise start
decays to it *exactly*, while a formed pattern persists at amplitude 0.548. That is
bistability at one parameter set, and it is the definition of a subcritical branch.

The `d/d_c = 0.999` row is worth its own line: the stencil band is genuinely stable
there (`-5.1e-05`) and yet a `1e-3` **finite** perturbation still ignites the pattern.
A finite-amplitude instability below linear onset is the same phenomenon seen from the
other side.

---

## 5. Verdict, and what it costs

**Gray-Scott cannot carry the wavelength-selection claim, at its own genuine Turing
point, for a reason no amount of replicates or grid tuning can fix.** The deferral is
closed as a measured dead end rather than as a cost decision.

Consequences, all of them "nothing to build":

* **The spectral helpers stay in `models/schnakenberg.py`.** The second caller that
  would have justified promoting them to `core/` does not exist. Phase 2c's own rule
  applies unchanged: they move when a real caller wants them.
* **Nothing in `models/gray_scott.py` needs a new refusal.** Its `initial` accepts only
  `"mode"` and `"pearson"`, so the model **cannot be asked** the selection question —
  there is no code path to guard, and a later reader should not go looking for a
  missing one. A test now pins that, so adding a `"noise"` initial condition with
  selection in mind fails loudly and lands here.
* **No `ndim` parameter was added to Gray-Scott.** The 1-D path exists only in these
  probes. Adding it to a validated 2-D model to support a test of a negative result is
  work the result does not justify.

### What this does not establish

* Only `(F, k) = (0.074, 0.062)` was measured. The genuine Turing sliver is
  `F in [0.049, 0.117]`, `k in [0.054, 0.062]`, and **subcriticality was not swept
  across it**. The claim here is about the module's validated point, which is the point
  any caller would have used; whether some corner of the sliver is supercritical is
  open and was not cheap enough to settle alongside.
* The bistable window is bracketed only between `d/d_c = 0.95` (bistable) and `0.90`
  (homogeneous only). Its lower edge was not bisected — nothing here needs it.
* All of it is 1-D. In 2-D the emergent structures differ, but the subcritical branch
  is a property of the reaction kinetics and the bistability demonstration does not
  depend on dimension.

---

## 6. What travels

1. **A genuine Turing instability is necessary but not sufficient for a wavelength
   claim.** Real non-trivial state, diffusionlessly stable, unstable band at `q > 0` —
   all three hold here, and the emergent pattern still has no predicted wavelength.
   The missing fourth condition is **supercriticality**, and it is checkable: amplitude
   continuous from zero as onset is approached, and no pattern surviving below onset.
2. **A calibrated instrument can be confidently wrong one model over.** The linear
   screen matched a known answer to 16%, reproduced its resolution dependence, and then
   predicted `+7.8 SE` where the truth was `-2.4 SE`. What failed was not its precision
   but its *applicability*, and its output looked identical either way. The fix is not
   a tighter screen — it is checking the precondition (here, supercriticality) before
   trusting a linear argument at all.
3. **A negative result is the deliverable when it is the answer.** Phase 2c's deferral
   named Gray-Scott as the candidate for a second caller. It measured out; recording
   that costs one document and saves a phase.
