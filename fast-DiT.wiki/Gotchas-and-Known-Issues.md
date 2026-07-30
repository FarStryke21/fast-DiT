# Gotchas and Known Issues

Verified by reading the code at HEAD (`4d43a3e`). Ordered by how much they could affect a conclusion.

> **Status update (2026-07-30, branch `worktree-publication-restructure`):** items **1, 2, 4, 7** are fixed on that branch, **8** has an opt-in mitigation (`--alpha-clamp`, off by default), **12** is partially addressed (`ablations.py --ckpt` flag; scripts still default to different checkpoints), and the stale `environment.yml` / upstream `README.md` issues in **13** are resolved. Rationale for each fix is in [Design-Decisions](Design-Decisions.md). Items **3, 5, 9, 10, 11** remain by design and must be stated as caveats in any write-up. The descriptions below document the pre-fix state at `4d43a3e`.

---

## 1. The `uncond` baseline does not use the null token

`sample_generator.py:63`

```python
v = model(z, t, y_uncond)          # y_uncond = zeros(B, 40), no force_drop_ids
```

With `force_drop_ids=None` and `model.eval()`, `MultiLabelEmbedder.forward` skips the drop branch entirely and embeds the **zero attribute vector**. Every other code path obtains the unconditional branch via `force_drop_ids=ones`, which substitutes the learned `null_token`.

So the `uncond` row in `evaluation.sh` measures *"conditioned on the all-negative attribute vector"* (no shadow, not attractive, not male, no beard, not young, …) — an out-of-distribution and internally contradictory condition — rather than the model's true unconditional distribution. Its FID is therefore not the clean "no guidance" reference it is presented as.

**Fix**: `model(z, t, y_uncond, force_drop_ids=torch.ones(B, dtype=torch.bool, device=device))`.

**Impact**: affects only the `uncond` baseline row. All CFG/MP comparisons are unaffected, since they all use `force_drop_ids` consistently.

---

## 2. `ablations.py` Phase 2 writes to the wrong directory and crashes

`ablations.py:97-121`. Phase 2 computes

```python
out_dir  = f"samples_cfg_mp_anderson_gated_{name}_w{standard_w}_steps{STEPS}"
fake_dir = os.path.join(out_dir, "fake")
```

but the generation command it builds **omits `--out-dir`**. `sample_generator.py` then falls back to its default `samples_{method}_w{cfg}_steps{steps}` — the *same* directory for every gate window, which is also the directory Phase 1's `w=4.0` run wrote to. Consequences, in order:

1. Each gate run overwrites the previous one and clobbers Phase 1's `w=4.0` results.
2. `evaluate_metrics.py` is then pointed at `{out_dir}/fake`, which **does not exist**.
3. `calculate_fid_given_paths` raises, `run_command` sees a non-zero return code and calls `exit(1)`.

So Phase 2 does not merely produce bad rows — it **terminates the script**, and the CSV is never written at all. Phase 1's already-completed runs are lost with it.

`time_ablation.sh` does the same sweep correctly (passes `--out-dir` to both the generator and the evaluator, unique dir per gate) and was added in the same commit that introduced the `--out-dir` flags. **Use `time_ablation.sh` for the gating ablation; use `ablations.py` only for Phase 1, or patch it.**

**Fix**: add `"--out-dir", out_dir` to `gen_cmd` and `"--out-dir", out_dir` to `eval_cmd` in Phase 2 (and, for cleanliness, Phase 1 too).

---

## 3. The corrector's anchor velocity is stale

`G(x) = x + (v_θ(x, t+dt, ∅) − v̄_∅)·dt/2` uses `v̄_∅ = v_θ(z_t, t, ∅)`, computed at the **previous** time and the **pre-step** latent, reused from the CFG batch.

At the fixed point this asks `v_θ(x*, t+dt, ∅) = v_θ(z_t, t, ∅)` — the two sides are evaluated at different times. The discrepancy is O(dt) and vanishes as steps → ∞, but at 50 steps it is a systematic bias: the corrector's target is not "the unconditional manifold at `t+dt`" but "wherever the unconditional field still looks like it did one step ago". This is defensible as a cheap choice (it costs zero extra NFE) but it should be stated explicitly rather than implied to be a true projection.

A more principled variant would recompute the unconditional velocity at `t+dt` from an uncorrected reference point, at +1 NFE per step.

---

## 4. Gradient checkpointing is always on, including at inference

`models.py:281`

```python
for block in self.blocks:
    x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c)
```

Unconditional — no `if self.training`, no flag. Under `torch.no_grad()` during sampling this adds pure overhead for zero memory benefit, and it is applied on every one of the ~200 forward passes per sample for the MP methods. It also emits a `use_reentrant` deprecation warning on modern PyTorch.

**Impact on conclusions: none** (it slows all methods equally, and NFE — not wall-clock — is the reported compute metric). **Impact on any wall-clock claim: large.** If a timing number is ever quoted, this must be disabled first.

---

## 5. `learn_sigma=True` in a flow-matching model

`out_channels = 6`, and every call site immediately does `.chunk(2, dim=1)` and discards the second half. Flow matching has no variance head to learn, so half of `final_layer.linear`'s output weights are trained on nothing and half the unpatchify work is wasted. Harmless to correctness (the discarded channels never influence anything), but it is dead capacity and a natural "we know about this" footnote.

---

## 6. `--num-classes` is a no-op for the model

`DiT.__init__` hard-codes `MultiLabelEmbedder(in_channels=40, ...)` at `models.py:201` and never consults `num_classes`. Passing `--num-classes 20` will build a 40-input embedder while the samplers construct 20-wide `y` tensors, and the MLP will raise a shape error. The flag exists on every CLI but **must stay at 40**.

---

## 7. Corrector step size `0.5·dt` is hard-coded

Appears four times (`sample_generator.py:94,100,105`, and the twin in `sample-cfg-mp.py`). It is a genuine hyperparameter of the method — it sets the contraction factor of `G` and therefore both the convergence rate and the stability of the Anderson extrapolation — but it is not exposed on any CLI and was never swept. Worth naming as a limitation, and it is the cheapest additional ablation available.

---

## 8. Anderson's `α` is unbounded

```python
alpha = (⟨f_2, Δf⟩ / (⟨Δf, Δf⟩ + 1e-8)).view(B, 1, 1, 1)
x = g_2 - alpha * (g_2 - g_1)
```

When the two residuals are nearly parallel and equal in magnitude, `Δf → 0` and `α` blows up (the `1e-8` guard prevents division by zero, not a large quotient). No clamping, no damping, no fallback to the plain Picard update. In a converged region — exactly where you'd hope to be — `Δf` is small by construction, so this is a real instability risk for individual samples, and it is per-sample so one bad `α` corrupts one image rather than the batch.

**Mitigation if artefacts appear**: `alpha.clamp(-1, 2)`, or skip extrapolation when `‖Δf‖ < ε‖f_2‖`.

---

## 9. Gate windows overlap at the endpoints

`args.tmin <= t_val <= args.tmax` is inclusive on both sides, so at N=50 the step `t=0.30` runs in both Early and Middle, and `t=0.70` in both Middle and Late. The four windows are not a partition (16 + 21 + 15 = 52 ≠ 50). Doesn't change any conclusion; does change the NFE arithmetic if someone recomputes it.

---

## 10. Directory-name collisions between sweeps

The default output directory is `samples_{method}_w{cfg}_steps{steps}` — it does **not** encode `tmin/tmax` or `proj_K`. Two gated runs with different windows, or two `cfg_mp_std` runs with different `K`, collide by default. `--out-dir` exists precisely to work around this and must be used for any sweep over those parameters. Also note the generator writes into an existing `fake/` without clearing it, so a smaller run over a larger one leaves stale PNGs behind — which the evaluator will catch only via its `len(images) != len(y_true)` check.

---

## 11. FID at N=1000

`SAMPLES=1000` everywhere. FID's bias is strongly sample-size dependent and 1000 is well below the usual 10k–50k. The numbers are usable as a *paired ranking* across methods (identical seed, identical conditions, identical real reference) but should never be quoted as CelebA FID values. Ideally state the sample size next to every FID figure in the write-up.

---

## 12. Checkpoint inconsistency across scripts

- `evaluation.sh` → `0066000.pt`
- `ablations.py`, `time_ablation.sh` → `0072000.pt`
- `hf_push.py` → `0100000.pt`

Three different training steps across the three result-producing entry points. The five-method comparison and the ablations were therefore run on **different models**. Rows from `evaluation.sh` and rows from `ablations.py` are not directly comparable, and the published Hub checkpoint matches neither. Worth either re-running the method comparison at 72k or stating the checkpoint per table.

---

## 13. Minor / cosmetic

- `MultiLabelEmbedder.forward` computes a random `drop_ids` tensor and then unconditionally overwrites it when `force_drop_ids` is given — a wasted RNG draw per forward, and it *does* advance the global RNG state, so sampling is only reproducible when every method makes the same number of forward calls. Different methods make different numbers of calls, but since each method's initial `z` is drawn before the loop starts, the paired-noise property survives. Worth knowing before adding any other stochastic step to the sampler.
- `ablations.py:82` uses `140` as the fallback `avg_nfe_per_sample` for the Middle gate; the true value is `142`.
- `train.py` still asserts `image_size % 8 == 0` and computes an unused `latent_size` — VAE leftovers.
- `environment.yml` is missing `datasets`, `pytorch_fid`, `huggingface_hub`, `scikit-learn`, `pandas`.
- `README.md` is entirely upstream's and documents a different project.
