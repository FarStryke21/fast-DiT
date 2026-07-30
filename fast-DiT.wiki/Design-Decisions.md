# Design Decisions

Running log of non-obvious choices, with alternatives considered and rationale. Newest first. Add an entry whenever a decision would otherwise live only in a chat transcript or a commit message.

---

## 2026-07-30 — Publication-restructure branch

### D9. Wiki committed into the repo
The `fast-DiT.wiki/` directory was previously untracked. Decision: commit it on this branch so the project documentation (and this decision log) is versioned alongside the code. Alternative — keeping it as a separate GitHub wiki repo — rejected for now: one repo, one history, and the wiki *is* the project's real documentation while `README.md` was upstream's. Note for merging: the untracked local copy in the main checkout must be removed (or moved aside) before merging this branch, or git will refuse to overwrite it.

### D8. Upstream code quarantined into `legacy/`, not deleted
Half the repo is inert three-generations-deep fork residue (IDDPM `diffusion/` package, latent-space samplers, `train_options/`, upstream ImageNet SLURM logs). Decision: `git mv` into `legacy/` with a warning README rather than delete. Rationale: preserves provenance and keeps `git log --follow` working, while making the live surface of the project (12 files) legible at a glance — reviewers and collaborators should not have to learn which half of the repo is real. Deletion was rejected as needlessly destructive; leaving in place was rejected because the upstream README + logs have already caused confusion (they describe a different project's results).

### D7. `uncond` baseline fixed to use the learned null token
`sample_generator.py` conditioned the "unconditional" baseline on the all-negative 40-attribute vector (no `force_drop_ids`), i.e. an out-of-distribution, internally contradictory condition rather than the model's true unconditional distribution. Fixed to `force_drop_ids=ones`. This *changes the baseline's numbers* — any previously measured `uncond` FID is invalid and the row must be re-run. All CFG/MP comparisons are unaffected (they were already consistent). Decision: fix outright rather than add a compatibility flag, because no committed result depends on the old behaviour and the old behaviour is simply wrong.

### D6. Corrector step size exposed as `--proj-step-scale` (default 0.5)
Previously hard-coded as `* dt * 0.5` in four places across two files. It is a genuine hyperparameter (sets the contraction factor of the fixed-point map and the stability of Anderson extrapolation) and was never swept. Default preserves existing behaviour exactly; the flag makes the cheapest missing ablation runnable.

### D5. Anderson α clamp added as opt-in (`--alpha-clamp`, default None = off)
The per-sample Anderson mixing coefficient is unbounded and blows up when the residual difference `Δf → 0` (i.e. exactly near convergence). Decision: add an optional symmetric clamp but keep it **off by default** so existing results remain reproducible; turn it on only if per-sample artefacts are observed, and report it if used. A default-on clamp was rejected because it would silently change the method between the course results and the paper.

### D4. `ablations.py` Phase 1 now runs a vanilla-CFG arm at every scale
The CFG-scale sweep previously only ran the gated corrector — there was no baseline curve, so the central claim ("the corrector bends the FID/accuracy trade-off") was unshowable. Decision: sweep `cfg` and `cfg_mp_anderson_gated` in the same loop, same seed, so the two arms are paired point-for-point. Phase 2 also fixed to pass `--out-dir` (it previously wrote every gate window to the same directory, clobbered Phase 1, then crashed the whole script — Gotchas §2). Checkpoint made a CLI argument (`--ckpt`, default 72k) toward fixing the 66k/72k/100k inconsistency across entry points; the paper should use **one** checkpoint everywhere.

### D3. Inference-time gradient checkpointing disabled (`if self.training`)
Checkpointing ran unconditionally, adding pure overhead on every one of the ~200 no-grad forward passes per MP sample. No effect on NFE accounting or any numeric result; large effect on wall-clock. Required before any wall-clock claim is quoted in the paper.

### D2. `learn_sigma=True` left in place
Half the output head is dead capacity in a flow-matching model, but removing it changes the state-dict shape and breaks every existing checkpoint. Decision: keep, and state it as a known-inefficiency footnote in the paper. Retraining without it is not worth the compute.

### D1. Session workflow (standing instruction from the user, 2026-07-30)
Fable acts as planner/coordinator; implementation is delegated to Opus/Sonnet worker agents. Design decisions are logged here rather than only in chat. Recorded in persistent memory as well.

---

## Earlier (course-project era, reconstructed)

- **Pixel space, no VAE** (commit `4f88c1d`): the VAE was removed and the ODE runs directly in 64×64 RGB. Simplifies the manifold story (one space, no decoder confound) at the cost of resolution.
- **Corrector anchored to the *stale* unconditional velocity** `v̄_∅ = v(z_t, t, ∅)`, reused from the CFG batch at zero extra NFE, rather than recomputed at `t+dt`. O(dt) bias, acknowledged in [Method](Method.md#3-the-manifold-projection-corrector); a recomputed-anchor variant (+1 NFE/step) is a candidate ablation for the paper.
- **Gate `[0.3, 0.7]` chosen as the hypothesised "speciation" window** where samples commit semantic content; Early/Late windows exist as falsification arms, Full as a consistency check against ungated Anderson.
