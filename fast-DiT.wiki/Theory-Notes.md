# Theory Notes

Working analysis for the paper's method section. Drafted 2026-07-30. Conventions as in [Method](Method.md): `t=0` noise, `t=1` data, `x_t = (1−t)x₀ + t·x₁`, learned velocity `v_θ ≈ E[x₁ − x₀ | x_t]`, Euler step `dt = 1/N`, CFG velocity `v_w = v_∅ + w(v_y − v_∅)`.

---

## 1. The velocity–denoiser identity (the workhorse)

For the straight-line interpolant with `x₀ ~ N(0, I)`, taking posterior expectations of `x_t = (1−t)x₀ + t·x₁` and `v = x₁ − x₀` conditioned on `x_t = x` gives

```
v(x, t) = ( x̂₁(x, t) − x ) / (1 − t),        x̂₁(x, t) := E[x₁ | x_t = x]
x̂₀(x, t) = x − t·v(x, t)
```

So the unconditional velocity field is an affine re-scaling of the **posterior-mean denoiser** `x̂₁`. Every statement about `v_∅` translates into a statement about the denoiser, which is where manifold intuition lives.

## 2. What the corrector's fixed point actually is

The corrector iterates `G(x) = x + s·dt·(v_∅(x, t') − v̄)` with `t' = t + dt`, `s =` `--proj-step-scale`, and anchor `v̄`. Its fixed point is the **level set**

```
M(v̄, t') := { x : v_∅(x, t') = v̄ }.
```

Via §1, `v_∅(x,t') = v̄  ⇔  x̂₁(x, t') − x = (1−t')·v̄`: the iterate is moved until the model's posterior belief about the endpoint, relative to the current position, is restored to what the anchor encodes. This is a **belief-consistency constraint**, not a metric projection — there is no distance being minimized and no orthogonality condition. The honest name is *self-consistency corrector*; "manifold projection" is justified only through the first-order geometry below.

## 3. Why it behaves like a projection: first-order geometry

Let `J := ∂v_∅/∂x = (∂x̂₁ − I)/(1−t')`. For a well-trained denoiser near the data manifold, `∂x̂₁` acts approximately as the **orthogonal projector onto the manifold tangent space**: eigenvalue ≈ 1 along tangent directions (the denoiser preserves on-manifold perturbations), ≈ 0 along normal directions (it flattens off-manifold noise). Hence, approximately,

```
J ≈ 0                on tangent directions
J ≈ −1/(1−t')·I      on normal directions.
```

One Picard update moves the iterate by `s·dt·(v_∅(x,t') − v̄) ≈ s·dt·J·δ` for displacement `δ` from the consistent point, i.e.

- **tangent components of δ are untouched** (G is the identity there, to first order);
- **normal components are shrunk** by the factor `1 − s·dt/(1−t')` per iteration.

This is the precise sense in which the corrector "projects": to first order it acts only on the manifold-normal part of the guidance-induced displacement. It also yields the convergence rate for free:

```
Picard contraction factor (normal directions) ≈ | 1 − s·dt/(1−t') |
```

- Stable iff `0 < s·dt/(1−t') < 2`.
- **Slow early, strong late**: at `t' = 0.3`, `dt = 1/50`, `s = 0.5` the factor is ≈ 0.986 per iteration (nearly no progress); at `t' = 0.98` it is ≈ 0.5. Near `t' → 1` with `s·dt/(1−t') > 2` it can *overshoot and diverge* — with N=50, s=0.5 this threshold is crossed at `t' > 1 − dt/4 = 0.995`, i.e. only past the final step, but larger `s` or smaller N moves it into the integration range: **`s·dt < 2(1−t'_max)` is a stability constraint worth stating** (relevant for the E8 step-size sweep — s=1.0 halves the margin).
- This gives the K-sweep (E6) a prediction: Picard needs `K−1 ≫ 2` iterations to make visible progress at mid-trajectory contraction ≈ 0.99, which is exactly the regime where a secant/Anderson extrapolation should dominate — the Anderson comparison has a theoretical reason to win, not just an empirical one.

## 4. What the corrector removes — and the stale-anchor bias

Expand the post-predictor residual around `(z, t)` (predictor `x⁰ = z + v_w·dt`):

```
v_∅(x⁰, t') − v̄_stale ≈ [ J·(v_w − v_∅)·dt ]  +  [ (∂_t v_∅ + J·v_∅)·dt ]  + O(dt²)
        (v̄_stale = v_∅(z,t))     guidance excursion      natural drift along the flow
```

The corrector drives this whole residual to zero. The first term is the **off-manifold component of the guidance kick** — `J·(v_w − v_∅) = w·J·(v_y − v_∅)`, which by §3 is (to first order) purely normal — killing it is the intended behaviour. The second term is the **legitimate time-evolution of the unconditional field along its own flow**; killing it too is an unintended *over-correction of size O(dt) per step, O(1)·(gate width) accumulated*, systematically dragging the trajectory toward where the unconditional field looked one step earlier. This is the precise cost of the free (0-NFE) stale anchor.

**Fresh anchor, done right.** Recomputing the anchor at the guided post-step point `x⁰` is *degenerate*: `G(x⁰) = x⁰` identically and the corrector becomes a 1-NFE no-op. The non-degenerate +1-NFE variant anchors at the **unconditional continuation** `x_∅' = z + v_∅(z,t)·dt`:

```
v̄_fresh = v_∅(x_∅', t')      ⇒ fixed point:  v_∅(x*, t') = v_∅(x_∅', t')
```

Both sides now sit at the same time `t'` — the natural-drift term cancels in the residual and only the guidance-excursion term survives. E10 (stale vs fresh) is therefore a *theory-backed* ablation: the measurable difference is exactly the accumulated drift bias.

## 5. Relation to the CFG-MP operator (arXiv:2601.21892)

Their per-iteration operator (2 NFE) is `G(x) = x − ½Δt·v_∅(x) + ½Δt·v_y(x − ½Δt·v_∅(x))`, fixed point `v_y(x − ½Δt·v_∅(x)) = v_∅(x)`: a **cross-branch gap-closing** condition (conditional velocity at a half-back-step point must match the unconditional velocity here). Ours (1 NFE/iter) never evaluates the conditional branch in the corrector: it is a **single-branch self-consistency** condition anchored across the step. Consequences worth stating in the paper:

1. **Cost**: 1 vs 2 NFE per iteration at the same K.
2. **Different fixed points**: theirs couples the two branches (the corrected point depends on `y` through `v_y`), ours is conditioning-independent given the anchor — the corrector itself cannot re-inject guidance, so attribute accuracy can only be paid for by the predictor. This is a falsifiable prediction: **our corrector should preserve accuracy at least as well as theirs at matched FID improvement** (their corrector can drag the iterate along conditional directions; ours cannot).
3. **Both are heuristic**: neither optimizes a stated objective with guarantees; both are damped fixed-point maps on velocity-consistency residuals.

## 6. Anderson step = per-sample secant extrapolation

With residual `f(x) = G(x) − x`, the implemented update is type-II Anderson, memory 1, β = 1: `x⁺ = g₂ − α(g₂ − g₁)`, `α = ⟨f₂, Δf⟩/‖Δf‖²` — the least-squares solution of "combine the last two iterates to null the linearized residual", i.e. the secant/Aitken-Δ² method per sample. Two structural notes:

- In the near-linear regime of §3 (residual dominated by one contraction factor `ρ` per sample), the secant step is *exact*: it lands on the fixed point of the linearization in one shot regardless of how close `ρ` is to 1. This is why Anderson at 2 NFE can beat Picard at K−1 = 2 (contraction 0.986² ≈ 0.97 of the residual remaining vs ≈ 0 for exact linear extrapolation).
- `α` is unbounded as `Δf → 0` — which happens both at convergence *and* when `ρ ≈ 1` makes successive residuals nearly equal. The `--alpha-clamp` mitigation trades a small bias for removing this per-sample instability; report if used (Design-Decisions D5).

## 7. Why gate to the middle — partial theory, honest limits

Three time-resolved quantities determine where the corrector can matter:

1. **Size of the disease**: the guidance excursion `w·J·(v_y − v_∅)·dt` — requires the branches to disagree (they don't at `t≈0`, where both fields see mostly noise and `v_y ≈ v_∅`) and `J` to be structured (no manifold to be normal to at `t≈0`).
2. **Strength of the medicine**: per-iteration contraction `s·dt/(1−t')` grows with `t` (§3) — the corrector is nearly powerless very early.
3. **Need**: by late `t`, `x̂₁` is nearly committed, branch disagreement collapses onto texture detail, and vanilla CFG's residual damage is small.

(1) and (2) both argue the corrector is wasted at small `t`; (3) argues it is unnecessary at large `t`. That brackets the useful window into the middle — consistent with the guidance-interval finding (Kynkäänniemi et al.) — but the location `[0.3, 0.7]` and the size of the effect are empirical claims for E5/E9, not theorems. The paper should present the gate as *empirically located, theoretically rationalized*.

## 8. What is NOT claimed (state in the paper's limitations)

- No guarantee the fixed point exists, is unique, or is reached (K is tiny; Anderson is un-safeguarded unless clamped).
- No orthogonality/metric-projection property beyond the first-order argument of §3.
- **The corrector does not preserve the conditional distribution.** It deterministically shifts probability mass toward unconditional self-consistency; there is no reweighting (contrast Feynman–Kac correctors) and no claimed marginal-preservation property (contrast Rectified-CFG++'s bounded-tube result). It is a quality-control heuristic evaluated by its (NFE, FID, accuracy) Pareto behaviour, full stop.
- The tangent/normal spectral picture of `∂x̂₁` (§3) is an idealization; for a real network it holds only approximately and degrades off-manifold — which is, circularly, where the corrector operates. Worth one honest sentence.

## 9. Candidate theory contributions for the paper (ranked)

1. §3 contraction analysis + the `s·dt < 2(1−t_max)` stability condition (new, simple, checkable against E8).
2. §4 decomposition of the residual into guidance excursion + natural drift, and the stale-anchor bias as exactly the second term (new; motivates E10 and the fresh-anchor variant; the degenerate-anchor observation is a nice remark).
3. §5 single-branch vs cross-branch fixed-point comparison with the accuracy-preservation prediction (positions against CFG-MP with a testable claim).
4. §7 three-factor gating rationale (positioning; ties to E5/E9).
