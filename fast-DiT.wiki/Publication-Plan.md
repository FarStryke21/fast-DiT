# Publication Plan

Drafted 2026-07-30. Goal: turn the course project into a publishable paper. Primary target **TMLR**, with a NeurIPS 2026 workshop submission as a parallel, non-archival visibility shot.

---

## 1. Thesis of the paper

> High-scale CFG drags the sampling trajectory off the model's own data manifold. A **training-free, per-step fixed-point corrector** — which restores self-consistency of the *unconditional* velocity field across each guided Euler step — recovers the fidelity lost to guidance without sacrificing conditioning accuracy. Anderson acceleration makes the corrector converge in two evaluations, and **time-gating** concentrates those evaluations in the mid-trajectory window where guidance actually does damage, so the whole mechanism costs ~40% extra NFE instead of 2×.

The paper's unit of claim is a point in **(NFE, FID, attribute-accuracy)** space; every comparison must be paired (same seed, same conditions) and NFE-annotated.

## 2. Contribution structure

1. **The corrector**: unconditional-velocity self-consistency as a cheap, training-free, model-agnostic proxy for manifold projection in flow-matching CFG (honestly framed as a heuristic proxy — no metric projection is claimed; the stale-anchor O(dt) bias is stated, and a recomputed-anchor variant is ablated).
2. **Anderson acceleration** of the corrector's fixed-point iteration (depth-1, per-sample): same NFE as Picard at K=3, better residual — the claim is *better use of matched compute*, not a speedup, unless the K-sweep shows Picard needs K−1 ≫ 2 to match it.
3. **Time-gating**: the corrector matters only in the mid-trajectory "speciation" window; gating to [0.3, 0.7] keeps most of the benefit at 142/200 of the NFE. Early/Late windows are falsification arms.

## 3. Related work and positioning

*(To be finalized — a dedicated related-work scan is in progress covering CFG++, limited-interval guidance (Kynkäänniemi et al. 2024), MPGD, MCG, autoguidance (Karras et al. 2024), Anderson/fixed-point acceleration in diffusion sampling, and predictor-corrector samplers. This section will record: closest works, precise differences, which claims survive as novel, and which baselines reviewers will demand.)*

## 4. Experiments

### 4.1 Must-have (claims don't stand without these)

| # | Experiment | Config | Why |
|---|---|---|---|
| E1 | **Vanilla-CFG scale sweep** | `cfg`, w ∈ {2,4,6,8} | The baseline trade-off curve. Now automated in `ablations.py` Phase 1 (paired arm added on this branch). |
| E2 | Corrector scale sweep | `cfg_mp_anderson_gated` Middle, same w's | The other arm of the headline figure: does the corrector bend the curve? |
| E3 | **NFE-matched control** | vanilla CFG at 100 steps (200 NFE) vs full Anderson at 50 steps (200 NFE); also CFG at ~71 steps (~142 NFE) vs gated-Middle (142 NFE) | First reviewer question: "is this better than just taking more Euler steps?" Nothing currently answers it. |
| E4 | Method comparison table | 5 methods × (NFE, FID, exact %, element %), w=4.0, **one checkpoint everywhere** | Re-run of `evaluation.sh` at the paper checkpoint (previous runs mixed 66k/72k). |
| E5 | Gating sweep | Early/Middle/Late/Full at w=4.0 (`time_ablation.sh`) | The efficiency claim + falsification arms. Verify Full ≡ ungated Anderson exactly (determinism check). |
| E6 | Picard-vs-Anderson K sweep | `cfg_mp_std` K ∈ {2,3,5,8} vs Anderson | The only way to substantiate the acceleration claim. |
| E7 | Re-run `uncond` baseline | with the null-token fix | The old uncond row measured the wrong thing (Gotchas §1); its FID is invalid. |

### 4.2 Should-have (robustness; reviewers will probe)

| # | Experiment | Why |
|---|---|---|
| E8 | Corrector step-size sweep, `--proj-step-scale` ∈ {0.25, 0.5, 1.0} | Previously hard-coded, now exposed. Un-swept hyperparameters in the headline method are a standard rejection reason. |
| E9 | Narrower gates ([0.4,0.6], [0.35,0.65]) | Where does the benefit break? Pushes NFE below 142. |
| E10 | Stale vs recomputed anchor (+1 NFE/step variant) | Turns the honesty caveat into an ablation. |
| E11 | Seed robustness: 3 seeds for the headline configs | FID at N=1000 is noisy; error bars or at least seed-spread needed for any Δ < a few FID points. |
| E12 | FID at N=5000–10000 for the 3–4 headline rows | Blunts the "N=1000 FID is biased" objection on the rows that matter. |

### 4.3 Scale-up (decides venue ceiling)

| # | Experiment | Why |
|---|---|---|
| E13 | Same corrector on a **public pretrained model** (e.g. SiT/DiT-XL-2 ImageNet-256, or SD-class latent flow model) — corrector is training-free and model-agnostic, so this is sampling-only compute | This single experiment moves the paper from "workshop/TMLR" toward "conference plausible". If it works on one pretrained model, the model-agnostic claim becomes real evidence rather than an assertion. |

### 4.4 Standing methodology rules

- One checkpoint for every table (pick 72k or retrain-to-100k; state it everywhere).
- Every FID quoted with its N. Every method row quoted with measured NFE (from `generation_stats.json`, never the fallback constant).
- Paired sampling everywhere: seed 50, identical noise, identical condition vectors across methods.
- Wall-clock numbers only now that inference-time gradient checkpointing is disabled (Design-Decisions D3); still prefer NFE as the compute metric.

## 5. Venue targets (verified 2026-07-30)

| Rank | Venue | Deadline | Fit |
|---|---|---|---|
| 1 | **TMLR** | Rolling; submit when ready (~late Sept–Oct) | Best fit: acceptance = "claims supported by rigorous evidence", explicitly not a novelty/scale bar. ~2-month decisions. Compatible with a non-archival workshop copy. |
| 2 | **NeurIPS 2026 workshop** | **Aug 29, 2026** (suggested framework deadline; notify Sept 29) | Parallel short-paper submission of current+E1–E7 results. Best confirmed fit so far is GDDL (geometric/OT framing, 2–4pp and 5–9pp tracks, non-archival, dual-submission-friendly). **Re-check the official NeurIPS workshop list mid-August** — a dedicated diffusion/probabilistic-inference workshop may still appear with the same deadline. |
| 3 | AISTATS 2027 | ~Oct 8, 2026 (estimated, CFP not posted) | Fallback conference target; more scale-tolerant reviewers. |
| — | ICLR 2027 main | Sept 16, 2026 (verified) | Only if E13 (pretrained-model result) lands convincingly within ~6 weeks; otherwise skip. |

## 6. Timeline (~9 weeks to a TMLR submission)

- **Weeks 1–2 (to ~Aug 13)**: land the restructure branch; re-verify environment on the GPU machine; recover any surviving ablation CSVs from the cluster before re-running; run E1–E2 + E4 + E7 at one checkpoint.
- **Weeks 3–4 (to ~Aug 27)**: E3, E5, E6. Freeze the 4-page workshop version; **submit to a NeurIPS workshop by Aug 29**.
- **Weeks 5–7**: E8–E12; start E13 if a pretrained backbone is chosen.
- **Weeks 8–9**: writing, figures (headline: FID-vs-accuracy curves with NFE annotations, both arms), TMLR submission.

## 7. Known risks

1. **Scooping / overlap** — the corrector-per-step + guidance-interval space is active (2024–2026); the related-work scan (§3) determines how much repositioning is needed. Biggest known threats: CFG++ (manifold-constrained CFG) and limited-interval guidance (gates *guidance* by time where we gate the *corrector*).
2. **The result itself** — no quantitative results survive in the repo; the central curve has never actually been measured with its baseline arm. The paper only exists if E1/E2 show separation.
3. **FID noise at small N** — mitigated by E11/E12; avoid claiming deltas within noise.
4. **Single-dataset, single-model evidence** — CelebA-64 + DiT-B/2 alone caps the venue at workshop/TMLR; E13 is the lever.
5. **Baseline demands** — expect reviewers to ask for CFG++, limited-interval guidance, and possibly autoguidance as baselines, not just vanilla CFG. Budget for at least the first two (both are cheap, training-free changes to the sampler).
