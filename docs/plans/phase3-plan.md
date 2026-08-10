# Phase 3 — Food webs, Daisyworld & adaptive dynamics (plan)

**Status: planning.** Phase 2 is complete (`phase2-plan.md`). This is the phase
HANDOFF §6 calls *"cross gradually into the speculative"* — and, like Phase 2, it
opens with a promised validation that **had to be reframed to stay honest**. Read
"The HANDOFF deviation" below before writing any May-criterion code; it is the
most important section here.

Every number in this document was measured before it was planned
(`M:\claud_projects\temp\phase3-slice\`), including the ones that killed a
proposed check.

## Goal

HANDOFF §6 lays out the arc from verifiable to exploratory in four stops. Phase 3
takes the **three that still have a checkable prediction**, and stops exactly
where the predictions do:

- **Generalized Lotka-Volterra + May/Allesina-Tang** — food webs. Exact interior
  equilibria on small hand-built systems; the random community matrix as pure
  linear algebra.
- **Daisyworld** — climate coupling. Its regulation claim turns out to be
  available in **closed form**, not as a qualitative "the curve looks flat".
- **Adaptive dynamics** — the crossing point. Evolutionary branching as a
  *predicted* outcome with a **sign change** at `sa = sK`.
- **Stochastic gLV** — demographic noise, carrying the project's organizing
  thread (`D(Omega) ~ Omega^{-1/2}`) into the phase. Priced at 15.6 s.

Nothing in this phase enters the `models/ecosystem/` quarantine. HANDOFF §6 calls
all three of these "still verifiable"/"still checkable", and the measurements
below bear that out, so they belong in `models/` beside the Phase 0-2 core. **That
Phase 3 adds nothing to the quarantine is the correct outcome, not an oversight.**

---

## The HANDOFF deviation (read before writing May-criterion code)

HANDOFF §6 promises:

> *"Then May's result (random community matrices: stability decreases with
> complexity) becomes a real, checkable prediction about your own simulated
> webs."*

**As literally written, that check cannot be run.** May's criterion is a statement
about a random matrix *assumed to be* some equilibrium's Jacobian. It is not a
statement about a gLV system whose equilibrium was computed from a random `A` and
`r`. Two measurements separate them:

**1. The random-gLV ensemble is empty at the `S` where May's asymptotic criterion
means anything.** Fraction of draws with a feasible interior equilibrium
(`x* = -A^{-1} r` all-positive), `A = -I + B`, `B` off-diagonals nonzero w.p. 0.5,
400 draws per cell:

| `S` | `sigma=0.1` | `sigma=0.25` | `sigma=0.5` |
|---|---|---|---|
| 5 | 1.000 | 0.970 | 0.517 |
| 10 | 1.000 | 0.598 | 0.035 |
| 20 | 0.958 | 0.028 | 0.000 |
| 40 | 0.405 | 0.000 | 0.000 |
| 80 | 0.003 | 0.000 | 0.000 |

**2. Conditioning on feasibility moves the spectrum.** At `S = 20`, `sigma = 0.25`
(8183 draws to collect 200 feasible ones): `max Re eig(A)` averages `-0.3345`
(sd 0.1239) while `max Re eig(diag(x*) A)` averages `-0.1170` (sd 0.0916), with
`x*` components spanning a factor of 67. The community matrix of a *feasible*
random gLV is simply not distributed like a May matrix.

**So Phase 3 splits the promise in three, and labels all three:**

1. **Category A — the random-matrix law itself, on the community matrix
   directly.** Pure linear algebra, no ODE, enormous statistics, and a tolerance
   that tightens with evidence. See "What is sharply checkable" below.
2. **Category A — gLV equilibria and stability on small hand-built systems**,
   where `x*` is exact and known to be feasible by construction. This is Phase 3's
   `birth_death`.
3. **Not claimed — that a feasibility-conditioned random gLV obeys the
   criterion.** Measurement 2 says it does not, and asserting it would be exactly
   the Gray-Scott error of Phase 2: comparing a measurement to a prediction that
   makes no claim about it.

### What is sharply checkable — and what was measured and rejected

The obvious anchor is the **spectral radius**: `max|lam| -> sigma sqrt(SC)`. It was
measured and **rejected**. `E[max|lam|] / R` sits at `1.035, 1.044, 1.041, 1.038,
1.031, 1.022` for `S = 25 ... 800` — a 2-3.5% excess that barely moves across two
decades, log-log slope of the excess only **`-0.1446`**. A hard-edge extreme with
log corrections is not an anchor you can put a tightening tolerance on.

**The circular law itself is.** Eigenvalues are uniform on the disc, so the
fraction inside radius `rho R` is exactly `rho^2` — `S` eigenvalues per draw times
many draws gives huge statistics and a *binomial* SE. At `S = 400`, 25 draws
(10 000 eigenvalues):

| `rho` | predicted `rho^2` | measured | binomial SE | z |
|---|---|---|---|---|
| 0.2 | 0.0400 | 0.040500 | 1.96e-3 | 0.26 |
| 0.4 | 0.1600 | 0.162900 | 3.67e-3 | 0.79 |
| 0.5 | 0.2500 | 0.251500 | 4.33e-3 | 0.35 |
| 0.6 | 0.3600 | 0.360700 | 4.80e-3 | 0.15 |
| 0.8 | 0.6400 | 0.642800 | 4.80e-3 | 0.58 |
| 0.9 | 0.8100 | 0.815400 | 3.92e-3 | 1.38 |

**And the elliptic law is better still.** Allesina & Tang's generalization: when
the off-diagonal *pairs* `(A_ij, A_ji)` are correlated with correlation `rho`, the
spectrum fills an **ellipse** with semi-axes `R(1 + rho)` and `R(1 - rho)`, so
stability needs `sigma sqrt(SC) (1 + rho) < d`. This is the ecologically
meaningful statement — predator-prey structure (`rho < 0`) is *more* stable than
random, competition/mutualism (`rho > 0`) less — and it is strictly sharper than
May's `rho = 0` special case at identical cost. Measured at `S = 400`, mapping the
ellipse to the unit disc at `t = 0.5`:

| `rho` | ellipse frac | pred | z | `E[max Re]` | pred `R(1+rho)` |
|---|---|---|---|---|---|
| -0.8 | 0.255000 | 0.25 | 1.15 | 4.2180 | 4.0 |
| -0.4 | 0.250600 | 0.25 | 0.14 | 11.9850 | 12.0 |
| 0.0 | 0.253100 | 0.25 | 0.72 | 19.8535 | 20.0 |
| +0.4 | 0.251000 | 0.25 | 0.23 | 27.6930 | 28.0 |
| +0.8 | 0.260000 | 0.25 | 2.31 | 35.4159 | 36.0 |

**The trap in this check, and it is the Phase-2 floor lesson in a new guise.** The
circular law is exact only as `S -> infinity`; at finite `S` there is a **bias**,
and piling on more eigenvalues shrinks the SE *below* it, so a correct
implementation starts failing. Measured at `rho = 0.5` with the eigenvalue count
held at 40 000 (SE fixed at 2.17e-3):

| `S` | bias | z |
|---|---|---|
| 50 | 1.227e-2 | 5.67 |
| 100 | 5.875e-3 | 2.71 |
| 200 | 3.025e-3 | 1.40 |
| 400 | 1.425e-3 | 0.66 |
| 800 | 1.000e-3 | 0.46 |

log-log slope **`-0.9279`**, i.e. **bias ~ 0.6 / S** (predicts 1.2e-2, 6.0e-3,
3.0e-3, 1.5e-3 — matching every row). So the test does not pick `S` by taste: it
**derives** `S` from `0.6/S << SE(n)`. At `S = 400` and 20 000 eigenvalues the bias
and the SE are the same size, which is the boundary, not a safe operating point.

**The stability transition itself was measured and is not an anchor.** `P(stable)`
vs `u = sigma sqrt(SC)/d` sharpens with `S` (width 0.322, 0.236, 0.141, 0.070 for
`S = 25...200`), but the measurement is grid-sensitive — at `S = 400` a 0.05-wide
`u` grid straddling a `0 -> 0.68 -> 0` jump returned a meaningless 0.229 — and
costs 45 s at `S = 400` alone. It is reported in the demo, not asserted.

---

## Generalized Lotka-Volterra

```
dx_i/dt = x_i (r_i + sum_j A_ij x_j)
```

### Measured before planning

- **Interior equilibrium** `x* = -A^{-1} r`, community matrix `M = diag(x*) A`.
  Two-species competition (`r = 1`, `A_ii = -1`, `A_ij = -a`) has the closed form
  `x* = 1/(1+a)` and `eig(M) = x*(-1 -+ a)`; at `a = 0.5`, `|rhs(x*)| = 0.0`
  exactly and `eig(M) = {-1, -1/3}` to all digits. A three-species asymmetric case
  gives `x* = (0.7009569378, 0.4354066986, 0.8421052632)`, `|rhs| = 1.87e-16`,
  `eig(M) = {-1.0367556167, -0.5911131304, -0.3506001525}`.
- **Non-stiff.** Slowest/fastest timescales differ by only 3x (`tau = 2.85` vs
  `0.96`). Explicit RK4 from `core/ode.py` is ample; no new integrator.
- **Relaxation rate = leading eigenvalue** — the `lambda(q)` analogue. Perturbing
  along the leading eigenvector and fitting `log|x - x*|` gives `-0.3504937983`,
  `-0.3505895944`, `-0.3505990967` at `eps = 1e-2, 1e-3, 1e-4` against the
  eigenvalue `-0.3506001525`: relative error `3.03e-4, 3.01e-5, 3.01e-6`. **Error
  exactly linear in `eps`** — this is a *linearization*, so the honest instrument
  is Richardson in the amplitude, precisely as Gray-Scott needed.
- **Classic LV predator-prey carries two exact invariants**, and they are a better
  category-A anchor than any equilibrium check because they hold *far from* the
  fixed point:
  - the conserved quantity `V = delta x - gamma ln x + beta y - alpha ln y`, drift
    `3.97e-13` over `t = 400` at `dt = 0.01` (small amplitude), `8.42e-9` at
    amplitude 4;
  - the **time-average identity** `<x> = x*` and `<y> = y*` over a full cycle, for
    **any** amplitude. Measured `<x> = 3.9999934909` and `3.9998998989` against
    `x* = 4` at amplitudes 1.2 and 4.0. The residual is dominated by the
    *cycle-endpoint detection*, not the integrator — the honest test interpolates
    the crossings.
  - small-oscillation period `2 pi / sqrt(alpha gamma) = 9.472258250995`, measured
    `9.49121951` at amplitude 1.2 (the period *grows* with amplitude: 9.805 at
    2.0, 11.271 at 4.0 — so this is a **limit**, checked by extrapolation, not a
    tolerance at one amplitude).

### The order-4 window is a constraint, not a footnote

RK4 order 4 on the gLV RHS is measurable only inside a window bounded on **both**
sides. At `t_max = 20` the trajectory has all but reached `x*`, the error has
collapsed to roundoff, and the ratios read `65.89, 5.76, 12.65, 15.09` — noise. At
`t_max = 5` on the transient, `dt = 0.5 -> 0.03125` gives `19.34, 17.02, 16.54,
16.28`, converging on 16 **from above** (`dt = 0.5` is outside the asymptotic
regime in the other direction).

**Prescribed configuration: `t_max = 5`, `dt in [0.125, 0.03125]`.** Written down
here because the natural-looking `t_max = 20` reads 65.89.

---

## Stochastic gLV (the spine)

Reaction set for `A_ij <= 0`:

```
birth        X_i -> 2 X_i         rate r_i x_i
self-limit   X_i + X_i -> X_i     rate |A_ii| x_i^2
competition  X_i + X_j -> X_i     rate |A_ij| x_i x_j   (i != j)
```

**Priced from the ODE first** (the Phase-1 discipline): total propensity at `x*` is
`2.6667 * Omega` events per time unit, so `53.3 * Omega` events per replicate at
`t_max = 20`. Measured: **`slope = -0.4984 +/- 0.0488`** over
`Omega in [100, 1600]` with 8 replicates, `1.32 M` events in **15.6 s** at
**11.8 us/event**. That is an eighth of the 122 s repressilator floor, so it costs
**zero wall clock** at `-n 6`.

### The bimolecular propensity — a decision taken with its cost known

This is the first model in the project needing a bimolecular reaction, which is
the item Phase 1 explicitly deferred. The engine's `a_j = Omega f_j(n/Omega)` is
exact for zeroth- and first-order reactions but uses `n^2/Omega` where the
microscopically exact self-limitation propensity is `n(n-1)/Omega`.

**Decision: keep the macroscopic propensities.** The excess death rate is
`|A_ii| n_i / Omega`, i.e. an effective `r_i' = r_i - |A_ii|/Omega`, so the bias is
`O(Omega^{-1})` — **subdominant** to the `O(Omega^{-1/2})` signal it would have to
corrupt, and therefore incapable of flooring the slope (unlike the `N`-independent
floors of Phase 2, which could).

**The argument was measured, and it is confirmed in magnitude but NOT in scaling.**
For this `A` and `r = 1` the effective-`r` shift predicts a *specific* number,
`x*' = x*(1 - 1/Omega)`, hence

```
<x>_exact - <x>_macro  =  x* / Omega
```

Measured with long single trajectories (`T = 3000`, 10 replicates per arm — the
time-average converges as `1/sqrt(T)`, so length beats replicate count here):

| `Omega` | measured diff | SE | predicted | z | measured/predicted |
|---|---|---|---|---|---|
| 25 | 2.998e-2 | **2.10e-1** | 2.667e-2 | 0.02 | 1.124 |
| 50 | 1.513e-2 | **3.29e-2** | 1.333e-2 | 0.05 | 1.135 |
| 100 | 5.043e-3 | 2.51e-3 | 6.667e-3 | -0.65 | 0.756 |

**Only the `Omega = 100` row is a measurement.** At `Omega = 25` and `50` the SE
*exceeds the signal* — and the reason is itself worth recording: **the bias cannot
be measured where it is largest.** As `Omega` falls, the SSA's own fluctuations
grow faster than the bias does (`z ~ sqrt(x* T / Omega)`), and below `Omega ~ 50`
the process makes near-extinction excursions — `<x> = 0.427` against `x* = 0.667`
at `Omega = 25`. The fitted slope `-1.2858` is therefore a fit through two noise
points and one real one, and **is not evidence of the `1/Omega` law**.

So the standing position, stated plainly: the `O(Omega^{-1})` subdominance rests on
the effective-`r` derivation plus a **single-point** consistency check (`z = -0.65`
at `Omega = 100`). It is not fully measured. Closing it needs either a
variance-reduced estimator (the two arms share the common `O(1/Omega)` nonlinearity
bias, which is why differencing them is right and comparing either to `x*` alone is
not) or roughly `T = 5400` at `Omega = 100`, ~340 s of SSA. **Carried as an open
item of the phase, not as a settled argument.**

---

## Daisyworld

Watson & Lovelock 1983, standard constants (`S = 917 W/m^2`, `q = 2.06e9 K^4`,
`gamma = 0.3`, albedos `0.75 / 0.25 / 0.5`, `T_opt = 295.5 K`, `beta` coefficient
`0.003265`).

### The interior equilibrium is closed-form — no root-find

The slice set out to root-find `T*(L)` and found it did not have to. At an interior
equilibrium both daisies satisfy `x beta(T_i) = gamma`, so `beta(T_w) = beta(T_b)`;
`beta` is a downward parabola symmetric about `T_opt`, so `T_w` and `T_b` are its
two roots, `T_opt -+ delta`. The local-temperature law then gives

```
T_b^4 - T_w^4 = q (A_w - A_b)
```

which is **one equation in `delta` alone** — a depressed cubic
`delta^3 + T_opt^2 delta - q(A_w - A_b)/(8 T_opt) = 0`, solvable by Cardano. Hence:

| quantity | value | depends on `L`? |
|---|---|---|
| `delta` | 4.988282516541 K | **no** |
| `T_w*` | 290.511717483459 K | **no** |
| `T_b*` | 300.488282516541 K | **no** |
| `beta(T_w*) = beta(T_b*)` | 0.918757127552 | **no** |
| bare fraction `x*` | 0.326528079079 | **no** |

`L` only sets how the fixed daisy cover `1 - x* = 0.673471920921` splits between
white and black, via a *linear* energy balance:
`A* = (k - T_w*^4 - q A_w)/(k - q)` with `k = S L / sigma`, then
`a_w = (A* - (1-x*) A_b - x* A_g)/(A_w - A_b)`.

Verified: `|rhs(y*)| <= 3.68e-16` at every `L` in the band, and the simulated
`T_w, T_b` reproduce `290.51171748 / 300.48828252` to all printed digits at
`L = 0.8, 0.9, 1.0, 1.1, 1.2, 1.3`.

### The regulation claim, sharp

This is what the closed form buys. The daisy temperatures are **pinned**, so

```
dT_w/dL = dT_b/dL = 0     exactly
```

— measured as `+0.000e+00`, not "small". And the planetary temperature does not
merely flatten, it **overcompensates**: `dT_e/dL` is *negative* across the band
while the bare planet's is strongly positive.

| `L` | `T_e*` daisy | `T_e` bare | `dT_e/dL` daisy | `dT_e/dL` bare | ratio |
|---|---|---|---|---|---|
| 0.9 | 296.199676 | 292.074452 | -13.565077 | 81.131792 | -0.1672 |
| 1.0 | 294.991245 | 299.869947 | -10.765397 | 74.967487 | -0.1436 |
| 1.1 | 294.020610 | 307.100918 | -8.751609 | 69.795663 | -0.1254 |
| 1.2 | 293.223807 | 313.854424 | -7.254677 | 65.386338 | -0.1110 |

**The regulating band is exact too**: the interior state is feasible for
`L in [0.7387224182, 1.3594723713]` (`a_w = 0` and `a_b = 0` respectively), width
0.620750. Outside it, boundary equilibria take over.

### Stability, and the category-C part

The interior fixed point is **stable across the whole band**, eigenvalues real and
negative (`{-0.0986, -0.7031}` at `L = 0.8` through `{-0.9213, -0.0420}` at
`L = 1.3`), and the relaxation-rate check behaves exactly like gLV's: measured
`-0.2298985946 / -0.2299881652 / -0.2299972042` at `eps = 1e-2/1e-3/1e-4` against
`-0.2299982102`, relative error `4.33e-4, 4.37e-5, 4.37e-6` — linear in `eps`
again, so Richardson in the amplitude again.

**Hysteresis and dieback are category C.** Ramping `L` up and down produces genuine
bistability — at `L = 1.4` the up-ramp holds `a_w = 0.687` while the down-ramp is
already dead; at `L = 1.0` the down-ramp settles on a *white-only* boundary state
`(0.339, 0)` rather than the interior `(0.400, 0.273)`. Reported and labelled,
never asserted against a bound.

### Two measured traps

- **`beta` is clipped at zero, so the RHS is only C^0** and RK4's order-4 claim
  needs smoothness. Measured: on the `L = 1.0` transient from `(0.01, 0.01)` the
  clip **never bites** — `beta_raw` stays above 0.73 throughout, and clipped and
  smooth `beta` give **bit-identical** results at every `dt`. Real in principle,
  not triggered here; it *would* bite on a dieback trajectory.
- **An order test run to `t_max = 200` measures nothing.** It read
  `1.28e-12` at `dt = 0.4, 0.2, 0.1, 0.05` — **ratio 1.00 at every step size** —
  because the trajectory had reached its attractor and the "error" was the
  distance to the fixed point, which is `dt`-independent. At `t_max = 20` the same
  code gives `15.3, 15.6, 15.8`. This is the Phase-2 mutant lesson exactly:
  **only a transient can see it.**

---

## Adaptive dynamics — the crossing point

Doebeli & Dieckmann's competition model: `K(x) = K0 exp(-x^2/(2 sK^2))`,
`a(x,y) = exp(-(x-y)^2/(2 sa^2))`, invasion fitness of a rare mutant `y` in
resident `x` at `n* = K(x)`:

```
s_x(y) = r (1 - a(y,x) K(x) / K(y))
```

### The closed forms, and the sign error the finite difference caught

Writing `s = r(1 - e^g)` with `g(y) = -(y-x)^2/(2 sa^2) + (y^2 - x^2)/(2 sK^2)`:

| quantity | closed form | at `x* = 0` |
|---|---|---|
| selection gradient `D(x) = ds/dy\|_{y=x}` | `-r x / sK^2` | 0 |
| `d2s/dy2` | `r (1/sa^2 - 1/sK^2)` | sign decides branching |
| `d2s/dxdy` | `-r / sa^2` | |
| convergence stability `dD/dx` | `-r / sK^2` | `< 0` **always** |

So `x* = 0` is **always convergence-stable**, and is a **branching point iff
`sa < sK`**. Verified against finite differences to `7.6e-8` or better at
`sa = 0.4, 0.7, 0.9, 0.99, 1.01, 1.2, 1.6`, and `dD/dx = -1.000000000000` exactly.

**The first draft of both second derivatives had the wrong sign** — closed form and
finite difference agreed in magnitude and disagreed in sign at every probe. A test
comparing `|closed|` to `|fd|` would have passed. **Compare signed values.**

### What is validated, and what is not

- **Category A — the derivatives above**, exact to 1e-8.
- **Category A (structural) — branch vs no-branch across `sa = sK`.** A
  multi-morph gLV on a trait grid, seeded at the singular point, splits for
  `sa/sK = 0.70, 0.95` and does not for `1.05, 1.50`. **This is the sign change,
  and it is the claim.**
- **Category B — branching time diverges as `1/(splitting rate)`.** The splitting
  rate at the singular point *is* `d2s/dy2 = r(1/sa^2 - 1/sK^2)`, which goes to
  zero as `sa -> sK`, so `t_branch` should diverge like its reciprocal. Measured
  across a **16x range** in the rate:

  | `sa` | rate | `t_branch` | `t_branch * rate` |
  |---|---|---|---|
  | 0.60 | 1.777778 | 3 200.5 | 5689.778 |
  | 0.70 | 1.040816 | 5 400.5 | 5620.929 |
  | 0.80 | 0.562500 | 10 000.5 | 5625.281 |
  | 0.90 | 0.234568 | 24 000.5 | 5629.747 |
  | 0.95 | 0.108033 | 52 600.5 | 5682.602 |

  The product is constant to **1.2%** and the log-log slope is **`-1.0003`**. This
  is the phase's cleanest asymptotic law and it costs 0.3-4.6 s per point. Note
  the *prefactor* is **not** universal — it is `log(threshold / seed amplitude)`,
  so it moves with the detection threshold; **the exponent is the claim, the
  constant is not.**
- **Not claimed — the number and positions of the post-branching morphs.**
  Measured grid-dependent: the two morphs sit at `+-0.100`, `+-0.050`, `+-0.025`
  for `ngrid = 81, 161, 321` — always exactly one grid spacing either side of the
  seed, at *every* resolution and every horizon up to `t = 200 000`. That is the
  measured fact. The *explanation* — that a Gaussian competition kernel is
  positive definite and the Gaussian x Gaussian pair is structurally degenerate
  (Gyllenberg & Meszena) — is **literature-anchored, category C**, and is stated
  in that order.

**The first branching diagnostic was green for the wrong reason.** Counting local
maxima on the trait grid reported "2 peaks = branching" for every `sa < sK` — but
the peaks were the seed's two immediate grid neighbours, i.e. a discretization
artifact, and it reported success at `t = 4000` where the population had not
branched at all. Replacing it with a **gap criterion** (clusters separated by
genuinely dead bins) changed the answer: `sa = 0.7` needs `t = 40 000` and
`sa = 0.95` needs `t = 200 000`.

### The canonical equation — and why it must NOT be a single-`sm` tolerance

The canonical equation `dx/dt = (1/2) mu sm^2 K(x) D(x)` is the deterministic limit
of the stochastic trait-substitution sequence as the mutation step `sm -> 0` — a
genuine second convergence track for the phase.

**The five-point sweep as first written was a single-point check in disguise.**
Holding `mu sm^2 t_max` fixed so every run travels the same distance *also pins the
canonical prediction at `1.849492` for every `sm`* — so the "sweep" only tested
whether five differently-noisy estimators land on one number, and the fitted slope
`0.7180` was a fit through noise (errors `4.79e-2, 2.23e-2, 1.02e-2, 1.16e-2,
5.50e-3` against SEs `9.6e-3 ... 3.0e-3`).

The obvious repair — demote it to a category-A tolerance at one small `sm` — was
tried, and **measurement rejected it**. At `sm = 0.0125` with 1200 replicates the
stochastic mean is `1.853567 +/- 0.001725` against a canonical `1.849492`: `z =
2.36`, a pass at `z = 4`. But four further independent seed groups give `1.851486,
1.852668, 1.856422, 1.853136` — **every one of them above the prediction**, mean
offset `~ +0.004`. That is not noise; it is the real `O(sm)` correction the
canonical equation is a limit of. A tolerance derived from the replicate SE
therefore **tightens onto a systematic bias**, and the check would start failing a
*correct* implementation as replicates grow. Precisely the failure mode
`validate()`'s tightening tolerance is supposed to protect against, arriving from
the other direction.

**So the honest form is category B after all: assert that the discrepancy vanishes
linearly in `sm`.** At fixed canonical distance the errors above go `4.79e-2 ->
2.23e-2 -> 1.02e-2` for `sm = 0.2 -> 0.1 -> 0.05` (ratios `2.15`, `2.19`) and
`~4.1e-3` at `sm = 0.0125`, consistent with `O(sm)` once the low-replicate points
are given adequate statistics. This is **Richardson in the amplitude for the third
time in this project** — HH's transient, Gray-Scott's linearization, and now a
mutation-step limit. The instrument generalizes; reach for it whenever the claim is
a limit rather than an identity.

**The teeth are verified, and they bite hard** (1200 replicates, `z = 4`
tolerance = 0.006902):

| variant | canonical `x` | `z` | |
|---|---|---|---|
| correct | 1.849492 | 2.36 | pass |
| drop the `1/2` | 1.662326 | **110.84** | fails |
| omit `K(x)` | 1.213061 | **371.21** | fails |
| `sm` for `sm^2` | 0.000000 | **1074.26** | fails |

---

## Performance and suite budget

Nothing in this phase is a plausible bottleneck, and no dependency is added:

- gLV / Daisyworld / adaptive dynamics ODEs are 2-3 dimensional. Non-stiff.
- The random-matrix track is `eigvals` on `S x S` matrices — 40 000 eigenvalues at
  `S = 400` costs 8.95 s, at `S = 800` 34.2 s. **`S` is chosen by the bias/SE
  argument, and the cost is a reason to prefer `S = 400` where that argument
  allows it.**
- Stochastic gLV: 15.6 s, measured.
- The branching simulations are the one item to watch: `t = 200 000` at
  `ngrid = 161` costs ~16 s per parameter point, so the branch/no-branch sign
  change across four `sa` values is ~65 s. Under the repressilator floor, but
  **it must be re-timed, not assumed** — and `sa = 0.95` was the point that needed
  the long horizon, so trimming the horizon trims the sign change.

**The baseline must be re-measured on a quiet machine before any of these are
compared to it.** The one run taken this session (`310 passed in 365.51s`) is
**invalid** — a slice script was running concurrently against the same cores. The
recorded figure is 130 s at `-n 6`.

---

## Build order

Ordered so each step is validated before the next depends on it, and so the three
models stay independent.

**3a — generalized Lotka-Volterra**

1. `models/glv.py` — the gLV RHS on `core/ode.py`; `analytic_predictions` = the
   closed-form interior equilibrium of a hand-built system. Test first, confirm
   red.
2. Relaxation rate = leading eigenvalue of `diag(x*) A`, with the Richardson-in-
   amplitude instrument (error measured linear in `eps`).
3. `models/lotka_volterra.py` — the predator-prey invariants: conserved `V`, the
   **time-average identity** `<x> = x*` at any amplitude, and the
   small-oscillation period as an extrapolated limit.
4. RK4 order 4 at the **prescribed window** (`t_max = 5`, `dt in [0.125,
   0.03125]`).

**3b — the random community matrix**

5. `core/random_matrix.py` (or `models/may_web.py`) — the May/elliptic ensemble.
   Circular-law and elliptic-law fraction checks with **binomial** SE, and `S`
   **derived** from `0.6/S << SE(n)`, not chosen.
6. Teeth: a wrong `R`, a wrong `rho` sign, a Hermitian-ized draw. Verified across
   3-4 seeds.
7. The demo reports the `P(stable)` transition and its sharpening — reported, not
   asserted — and states plainly that feasibility-conditioned gLV is **not**
   claimed to obey it, with the feasibility table as the evidence.

**3c — stochastic gLV**

8. `models/glv_stochastic.py` on the existing Gillespie engine.
9. **Close the `O(1/Omega)` bias measurement** — magnitude is confirmed at
   `Omega = 100` only, and the scaling is *not* resolved (see above). Needs a
   variance-reduced estimator or ~340 s of SSA. Until it closes, the subdominance
   claim is labelled a derivation with one supporting point, in the code comment
   as well as here.
10. `D(Omega) ~ Omega^{-1/2}` via `core/convergence.py`, with teeth verified
    across seeds.

**3d — Daisyworld**

11. `models/daisyworld.py` — closed-form `delta` by Cardano; `analytic_predictions`
    = `T_w*`, `T_b*`, `x*`, `a_w*(L)`, and **`dT_w/dL = 0`**. It must **raise**
    outside the regulating band `L in [0.73872, 1.35947]`, where no interior state
    exists — the Phase-2 refusing-to-answer discipline.
12. Order-4 on a **transient** (`t_max = 20`, never 200), and the `beta`-clip
    smoothness note.
13. Category C: hysteresis, dieback, the bare-planet comparison — demo only.

**3e — adaptive dynamics**

14. `models/adaptive_dynamics.py` — invasion fitness, the selection gradient and
    both second derivatives in closed form, checked **signed** against finite
    differences.
15. The branch/no-branch sign change across `sa = sK`, using the **gap** criterion
    and the measured horizons (`sa = 0.95` needs `t = 200 000`); then the
    `t_branch * rate = const` law (slope `-1.0003`), asserting **the exponent, not
    the prefactor**.
16. The canonical equation as an **`O(sm)` vanishing discrepancy**, not a
    single-`sm` tolerance — a tolerance at one `sm` was measured to tighten onto a
    real `+0.004` bias. Teeth already verified (`z = 110.84 / 371.21 / 1074.26`).
17. Label the post-branching morph structure exploratory, with the grid-dependence
    measurement as the reason.

**All**

18. Re-time the suite cleanly; update `CLAUDE.md`, memory, docs; commit and push.

## Explicitly deferred

- **The `models/ecosystem/` quarantine stays empty.** The full biosphere is
  HANDOFF §6's last stop and has no checkable prediction; nothing in Phase 3
  required crossing that line.
- **A non-positive-definite competition kernel** (e.g. `exp(-|x-y|^p/(p sa^p))`
  with `p = 4`) to escape the Gaussian degeneracy. Even if it makes morph
  positions grid-convergent, that is a *numerical-robustness* statement with no
  closed form behind it — it is not a validated claim, and promoting it would be
  the error this project exists to avoid. Measured note only.
- **An `sm` sweep in which the canonical *prediction itself* moves** (sweeping `sm`
  at fixed `t_max` rather than at fixed `mu sm^2 t_max`). The `O(sm)` form above
  is a sound claim at fixed canonical distance; a moving-target design would test
  a strictly stronger one, at more replicates than the phase can afford.
- **The `P(stable)` transition width as a category-B exponent** — measured
  (`-0.2735` over `S = 25..400`, contaminated by a grid-resolution artifact at
  `S = 400`), rejected as too delicate and too expensive for what it buys.
- **Exact combinatorial (falling-factorial) propensities in the Gillespie engine**
  — carried over from Phase 1 again, now with a measured justification rather than
  an assumption.
- **Structured food webs** (trophic levels, niche/cascade models) — the random
  ensemble first.
