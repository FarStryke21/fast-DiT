# Method

Everything here is read directly off `sample_generator.py` (the canonical implementation used for all metrics) and `sample-cfg-mp.py` (the grid-visualisation twin). Where the two differ, `sample_generator.py` wins.

---

## 1. Generative model: rectified flow / flow matching

Training (`train.py:251-272`) is **not** DDPM. The `diffusion/` package is imported and `create_diffusion()` is called but its output is never used — the loop is pure conditional flow matching:

```
x_1 ~ data                          # clean 64×64 RGB image in [-1, 1]
x_0 ~ N(0, I)                       # same shape
t   ~ U(0, 1)                       # per-sample, shape (B,1,1,1)
x_t = (1 - t)·x_0 + t·x_1           # straight-line probability path
v_target = x_1 - x_0                # constant along the path
loss = MSE( v_θ(x_t, t, y)[:, :3], v_target )
```

So `v_θ` learns the **velocity field of the straight-line (rectified) interpolant**. Note the time direction: **`t=0` is noise and `t=1` is data.** Every gating window, every log statement, and every "Early/Middle/Late" label in the ablations follows this convention.

### Sampling = forward Euler on the ODE

`dx/dt = v_θ(x, t, y)`, integrated from `t=0` to `t=1` with `N` uniform steps (`sample_generator.py:57-65`):

```
dt = 1/N
for i in 0..N-1:
    t = i/N
    x_{t+dt} = x_t + v(x_t, t, ·) · dt
```

The loop evaluates `t` at `0, 1/N, ..., (N-1)/N`, so the last step integrates from `(N-1)/N` to `1`. Output is unnormalised with `(z + 1)/2` and clamped to `[0,1]` — **no VAE decode**, the ODE runs in pixel space.

---

## 2. Vanilla CFG

Standard, applied to velocities rather than epsilon (`sample_generator.py:68-79`):

```
v_cfg = v_∅ + w · (v_y − v_∅)
```

Implementation detail that matters: conditional and unconditional passes are **batched into one forward** by duplicating `z` and `t` and concatenating `[y_cond, y_uncond]`, with

```python
drop_ids = cat([zeros(B, bool), ones(B, bool)])
v = model(z2, t2, y2, force_drop_ids=drop_ids)
```

`force_drop_ids=1` makes `MultiLabelEmbedder` swap the label vector for the learned `null_token` **before** the MLP. This is why the "unconditional" branch is *not* simply `y=0` — it is a trained null embedding. (See [Gotchas](Gotchas-and-Known-Issues.md#1-the-uncond-baseline-does-not-use-the-null-token) for the one place where this is violated.)

Cost: **2 NFE per Euler step.**

### Why CFG needs fixing

Large `w` extrapolates along `(v_y − v_∅)`, which is not a direction the model was ever trained to travel. The iterate drifts into a region of state space where the learned velocity field is unreliable — the classic high-guidance failure mode: saturated colours, over-sharpened texture, collapsed diversity, rising FID even as attribute accuracy improves. The project's hypothesis is that this drift is correctable *post hoc*, per step, without retraining.

---

## 3. The manifold-projection corrector

After the CFG Euler step produces the **predictor** iterate

```
x⁰ = z_t + v_cfg · dt
```

the corrector performs extra **unconditional-only** model evaluations at the *next* time `t' = t + dt` and applies the fixed-point map (`sample_generator.py:90-94`)

```
G(x) = x + ( v_θ(x, t', ∅) − v̄_∅ ) · dt · 0.5
```

where `v̄_∅ = v_θ(z_t, t, ∅)` is the unconditional velocity **already computed during the CFG pass at the previous time and reused** (the variable `v_uncond_out`; it is *not* recomputed).

### What the fixed point means

`G(x*) = x*` ⟺ `v_θ(x*, t', ∅) = v̄_∅`.

The corrector therefore searches for a point at time `t'` at which the **unconditional** velocity field agrees with the unconditional velocity the model assigned before the guided jump. The reading is: the unconditional field is the model's own description of where the data lies; the guided predictor step moved somewhere the unconditional field no longer says the same thing; the corrector slides the iterate back until the unconditional field is self-consistent again. That is the "projection back onto the manifold".

Two honest caveats to keep in mind when writing this up:

1. This is a **heuristic proxy for a projection**, not a projection in any metric sense. There is no orthogonality condition, no distance being minimised, and no guarantee a fixed point exists or is unique.
2. `v̄_∅` is a *stale* anchor — evaluated at `(z_t, t)`, not at `(·, t')`. As `dt → 0` the mismatch vanishes, but at 50 steps it is a real O(dt) bias. It also means the corrector's target moves every outer step.

Cost is the corrector's whole design problem: each `G` evaluation is **+1 NFE**, on top of the 2 NFE the CFG predictor already spent.

### Variant A — `cfg_mp_std`: plain fixed-point (Picard) iteration

```python
for k in range(1, proj_K):          # K-1 iterations
    v_proj = model(x, t_next, y_uncond, force_drop_ids=ones)[:, :3]
    x = x + (v_proj - v_uncond_out) * dt * 0.5
```

Straight Banach iteration `x ← G(x)`. With the default `proj_K=3` this is **2 iterations = +2 NFE per step**. Convergence rate is linear in the contraction factor of `G`, which is roughly `1 − (dt/2)·∂v_∅/∂x` — i.e. very slow when `dt` is small, which is exactly the regime you want to be in.

### Variant B — `cfg_mp_anderson`: Anderson acceleration, depth 1

```python
v1 = model(x,   t', ∅);   g_1 = x   + (v1 − v̄_∅)·dt/2;   f_1 = g_1 − x
v2 = model(g_1, t', ∅);   g_2 = g_1 + (v2 − v̄_∅)·dt/2;   f_2 = g_2 − g_1

Δf = f_2 − f_1
α  = ⟨f_2, Δf⟩ / (⟨Δf, Δf⟩ + 1e-8)      # per-sample scalar, flattened over C·H·W
x  = g_2 − α · (g_2 − g_1)
```

This is **type-II Anderson mixing with memory depth m=1 and mixing parameter β=1**, equivalently a secant / Aitken-Δ² extrapolation on the residual `f(x) = G(x) − x`. Standard Anderson-1 gives `x_{k+1} = G(x_k) − γ·(G(x_k) − G(x_{k-1}))` with `γ = ⟨f_k, Δf⟩/⟨Δf,Δf⟩`; substituting `x_k = g_1`, `G(x_k) = g_2`, `G(x_{k-1}) = g_1` reproduces the code exactly. The `α` is computed **per sample in the batch** (`.view(B,1,1,1)`), so each image gets its own least-squares extrapolation coefficient.

Cost: **exactly +2 NFE per step**, always.

> **Critical framing point.** With the default `proj_K=3`, `cfg_mp_std` also costs +2 NFE. The two corrector variants at their default settings are therefore **NFE-matched**, and the Anderson claim is not "cheaper" but "better use of the same two evaluations" — the extrapolation step is free, it only reuses `g_1, g_2` already in memory. The efficiency claim over `cfg_mp_std` only appears if you argue Anderson reaches a comparable residual at K−1=2 that Picard would need K−1 ≫ 2 to reach.

### Variant C — `cfg_mp_anderson_gated`: time-gated Anderson

```python
is_in_gate = tmin <= t_val <= tmax
if method == "cfg_mp_anderson_gated" and is_in_gate:
    ... run the Anderson corrector ...
```

The corrector fires only for Euler steps whose `t` falls in `[tmin, tmax]`. Motivation: manifold drift from CFG is not uniform in time. Very early (near-noise) the iterate is dominated by Gaussian noise and "the manifold" is barely defined; very late (near-image) the velocity field is nearly constant and the corrector has little to fix. The **middle band is where the sample commits to its semantic content** — the project's working name for this is the *speciation phase*, and `[0.3, 0.7]` is the hypothesised sweet spot, hard-coded as `optimal_tmin, optimal_tmax` in `ablations.py:41`.

Note that `cfg_mp_std` and `cfg_mp_anderson` **ignore** the gate entirely (`sample_generator.py:85`) — they always correct. Setting the gate to `[0.0, 1.0]` on the gated method is therefore an exact re-run of `cfg_mp_anderson` and serves as a built-in consistency check.

---

## 4. NFE accounting

Per Euler step:

| Stage | uncond | cfg | cfg_mp_std (K) | cfg_mp_anderson | gated (in window / outside) |
|---|---|---|---|---|---|
| Predictor | 1 | 2 | 2 | 2 | 2 / 2 |
| Corrector | 0 | 0 | K−1 | 2 | 2 / 0 |
| **Total** | **1** | **2** | **K+1** | **4** | **4 / 2** |

Totals at the standard `--num-steps 50`, computed from `t_val = i/50, i ∈ [0,49]`:

| Method | In-gate steps | NFE / sample |
|---|---|---|
| `uncond` | — | 50 |
| `cfg` | — | 100 |
| `cfg_mp_std` (K=3) | 50 | 200 |
| `cfg_mp_anderson` | 50 | 200 |
| gated **Full** `[0.0, 1.0]` | 50 | 200 |
| gated **Early** `[0.0, 0.3]` | i=0..15 → 16 | 132 |
| gated **Middle** `[0.3, 0.7]` | i=15..35 → 21 | 142 |
| gated **Late** `[0.7, 1.0]` | i=35..49 → 15 | 130 |

`sample_generator.py:129-147` accumulates `nfe_total` as *batch-element* evaluations and writes `avg_nfe_per_sample` to `generation_stats.json`, so these are the numbers the ablation table pulls in. (`ablations.py:82` uses a hard-coded fallback of `140` if the file is missing — the true Middle value is **142**; the fallback is a placeholder, not a measurement.)

The headline efficiency argument is the pair **(gated-Middle 142 NFE) vs (full Anderson 200 NFE)**: a 29% NFE reduction, against whatever FID/accuracy delta the sweep measures.

---

## 5. Design properties worth stating in a write-up

- **Training-free.** No fine-tuning, no auxiliary network, no gradients (`torch.set_grad_enabled(False)` for all sampling).
- **Model-agnostic.** The corrector only needs an unconditional forward pass; nothing is DiT-specific.
- **Orthogonal to CFG scale.** `w` and the corrector are swept independently (`ablations.py` Phase 1 vs Phase 2).
- **Deterministic and paired.** Seed 50 everywhere, and the target attribute vectors are chosen by `torch.randperm(...)[:N]` under that seed, so **every method sees identical noise and identical conditioning vectors** — the comparison is paired, not just distributional.
