# Publication Plan

Drafted 2026-07-30. Goal: turn the course project into a publishable paper. Primary target **TMLR**, with a NeurIPS 2026 workshop submission as a parallel, non-archival visibility shot.

---

## 1. Thesis of the paper (reframed 2026-07-30: this is an *improvement-over-CFG-MP+* paper)

The user's directive: CFG-MP/CFG-MP+ (arXiv:2601.21892, ICML 2026) is known, published prior work — the goal is to **beat it**, not to coexist with it.

> Manifold-projection correctors fix high-scale CFG's fidelity loss, but the published corrector (CFG-MP+) re-evaluates both guidance branches at every step of the trajectory. We show the same — or better — correction is available at a fraction of the cost: (i) a **single-branch corrector** that needs only unconditional evaluations (1 NFE/iteration vs 2, and their two forwards are sequential while ours is one forward per iteration, so the wall-clock gap is larger than the NFE gap); (ii) **time-gating**, which we show is a universal plug-in that also improves *their* corrector — correction only matters in the mid-trajectory window; (iii) a **principled anchor** and a contraction/stability analysis that the prior work lacks, yielding testable predictions (accuracy preservation of single-branch correction; the `s·dt < 2(1−t_max)` stability bound). Concrete headline: CFG-MP+ ≈ 300 NFE/sample at 50 steps; gated single-branch ≈ 142; gated CFG-MP+ (our plug-in applied to their method) ≈ 184.

The paper's unit of claim is a point in **(NFE, FID, attribute-accuracy)** space; every comparison must be paired (same seed, same conditions) and NFE-annotated. Secondary axis: wall-clock, now that inference checkpointing is fixed (their sequential double-forward loses there too).

## 2. Contribution structure (revised 2026-07-30; improvement-over-CFG-MP+ framing per the user)

1. **Single-branch projection**: the corrector needs only the *unconditional* velocity field — a self-consistency condition costing **1 NFE/iteration** vs CFG-MP's 2 (and 1 sequential forward vs their 2). Theory-backed prediction: because our corrector cannot move the iterate along conditional directions, it preserves conditioning accuracy at least as well ([Theory-Notes §5](Theory-Notes.md)). Verified head-to-head in B3/E13.
2. **Time-gating as a universal plug-in for projection correctors** (guidance always on, corrector only for t ∈ [0.3, 0.7]): no prior work gates a corrector by time, and we show it improves *both* our corrector (200→142 NFE) *and* CFG-MP+ itself (300→184 NFE). The insight "the middle matters" is Kynkäänniemi et al.'s; the transfer to correctors, the universality evidence, and the three-factor rationale ([Theory-Notes §7](Theory-Notes.md)) are ours. Early/Late windows are falsification arms.
3. **Theory the prior work lacks**: contraction analysis with the `s·dt < 2(1−t_max)` stability bound; the residual decomposition identifying the stale-anchor bias as exactly the natural-drift term, motivating the principled (unconditional-continuation) fresh anchor; the linear-regime argument for why secant/Anderson dominates Picard precisely in the mid-trajectory contraction regime.
4. **The empirical study**: attribute-conditioned (40 binary attributes), pixel-space CelebA-64 with paired-seed (NFE, FID, accuracy) protocol — plus the E13 head-to-head on CFG-MP+'s own benchmark family.

**Explicitly NOT contributions** (established prior art, cite in the motivation): CFG's off-manifold diagnosis and training-free per-step projection as a thesis (CFG++, Rectified-CFG++, CFG-MP); Anderson acceleration of a CFG-corrector fixed point (CFG-MP+ does type-II, m=1 — exactly ours); mid-trajectory restriction as an insight (Kynkäänniemi et al.).

**Naming**: the shorthand "CFG-MP" **collides with the ICML 2026 paper's own name** and must change. Candidates: **UCC** (Unconditional-Consistency Corrector), **GUCC** (gated UCC), **SCoG** (Self-Consistency after Guidance). Decision pending — update README/wiki/scripts when chosen.

## 3. Related work and positioning (scan completed 2026-07-30)

### Closest prior work — must be addressed head-on

| Work | Mechanism | Relation |
|---|---|---|
| **CFG-MP / CFG-MP+** (arXiv:2601.21892, Jan 2026; ICML 2026 poster) | Per-step fixed-point corrector after the CFG Euler step: half-step back along unconditional velocity, half-step forward along conditional velocity at the intermediate point; K≈2 iterations; **CFG-MP+ adds type-II Anderson (m=1, β=1)**. DiT-XL/2-256, Flux, SD3.5. No time-gating. | **The direct competitor.** Our map is unconditional-only with a stale anchor (1 vs 2 NFE/iter, different fixed point: self-consistency of v_∅ across the step, not descent on a cond–uncond gap). Reviewers will read us as "a cheaper uncond-only ablation of CFG-MP+ with time-gating" unless we differentiate mathematically and compare empirically. If our experiments predate Jan 2026, say "concurrent/independent" honestly — but still cite and compare. |
| **Rectified-CFG++** (arXiv:2510.07631, NeurIPS 2025) | Predictor–corrector *guidance*: half-step along conditional velocity, then CFG-mix at the midpoint with a time-decaying weight. ~1.5× CFG NFE, no iteration/fixed point. Proves bounded-tube manifold proximity. | Owns the framing "predictor–corrector fixes CFG's off-manifold drift in rectified flows" (Oct 2025). Our intro cannot claim that framing as new. |
| **CFG++** (Chung et al., arXiv:2406.08070) | One-line DDIM change: renoise with the *unconditional* prediction instead of the CFG-extrapolated one. Zero extra NFE. | Owns the off-manifold diagnosis (2024) and the "let the unconditional model carry the iterate back" principle. Its velocity-space analogue is a **zero-cost baseline we must run before writing** — if it matches us, the story changes. |
| **Guidance interval** (Kynkäänniemi et al., arXiv:2404.07724, NeurIPS 2024) | Guidance weight w only inside a mid-trajectory noise interval, 1 outside; *reduces* NFE; ImageNet-512 FID 1.81→1.40. | Owns "the middle of the trajectory is where guidance matters". They gate guidance; we keep guidance on and gate the corrector. Mandatory baseline on our setup. |

### Lineage and orthogonal work (cite, not competitors)

- **MPGD** (ICLR 2024) / **MCG** (NeurIPS 2022): training-free manifold-preserving *external-loss* guidance — the intellectual lineage of on-manifold sampling corrections, not CFG.
- **Autoguidance** (Karras et al., NeurIPS 2024): fixes the guidance *signal* (bad-model extrapolation) instead of the iterate; orthogonal philosophy, requires a second network; arguable to decline as a baseline at CelebA-64 scale.
- **Anderson/fixed-point machinery in sampling**: ParaTAA (ICML 2024), ParaDiGMS (NeurIPS 2023), DEQ samplers — established technology; Anderson can only ever be presented as a borrowed efficiency tool.
- **Predictor–corrector**: Song et al. 2021 Langevin correctors; stochastic-corrector-for-flow-ODE work (arXiv:2410.02217); Feynman–Kac correctors (ICML 2025). An NFE-matched Langevin corrector is a plausible reviewer request.
- **Cheap CFG variants to acknowledge**: CFG-Zero*, weight schedulers (Wang et al., TMLR 2024 — increasing w(t) beats constant at zero cost), adaptive/feedback guidance, ICG/TSG.

### Positioning sentence (draft)

> Building on the observation that classifier-free guidance drives flow-matching trajectories off the data manifold (Chung et al., 2024; Saini et al., 2025) and that its dynamics matter chiefly at intermediate noise levels (Kynkäänniemi et al., 2024), we propose a minimal per-step corrector that enforces self-consistency of the *unconditional* velocity field across each guided Euler step — requiring only unconditional evaluations, one per fixed-point iteration, unlike prior projection correctors that re-evaluate both branches (CFG-MP+, 2026) — and show that time-gating this corrector to the mid-trajectory interval retains essentially all of its FID recovery at a fraction of the compute.

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

### 4.1b Competitor baselines (added after the §3 scan; the first two are go/no-go gates)

| # | Baseline | Cost | Why |
|---|---|---|---|
| B1 | **CFG++ velocity-space analogue** (use v_∅ in place of v_cfg where CFG++ swaps the renoising prediction) | ~0 extra NFE, small sampler patch | **Run before writing anything.** If a zero-cost sampler change matches the corrector, the paper's story must change. |
| B2 | **Guidance interval** (Kynkäänniemi): w active only in a mid-trajectory window, 1 outside; sweep (interval × w) | *Reduces* NFE, small sampler patch | The other kill-shot: if gating guidance alone beats always-on-guidance + gated corrector on the FID/accuracy trade-off at lower NFE, the method needs a different justification. |
| B3 | **CFG-MP / CFG-MP+ operator** (uncond back-step + cond forward-step, K=2, with/without Anderson) | 2 NFE/iteration, moderate patch | The direct competitor (ICML 2026). Same seed, same NFE budget — the paper's differentiation claim (1 vs 2 NFE/iter) lives or dies here. |
| B4 | Increasing-w scheduler (Wang et al., TMLR 2024) | one line, 0 extra NFE | Cheap, published, reviewers know it. |
| B5 | Reduced constant w (does w=2–3 vanilla just match corrected w=4+?) | free — falls out of E1 | Sanity anchor for the whole trade-off story. |

### 4.2 Should-have (robustness; reviewers will probe)

| # | Experiment | Why |
|---|---|---|
| E8 | Corrector step-size sweep, `--proj-step-scale` ∈ {0.25, 0.5, 1.0} | Previously hard-coded, now exposed. Un-swept hyperparameters in the headline method are a standard rejection reason. |
| E9 | Narrower gates ([0.4,0.6], [0.35,0.65]) | Where does the benefit break? Pushes NFE below 142. |
| E10 | Stale vs recomputed anchor (+1 NFE/step variant) | Turns the honesty caveat into an ablation. |
| E11 | Seed robustness: 3 seeds for the headline configs | FID at N=1000 is noisy; error bars or at least seed-spread needed for any Δ < a few FID points. |
| E12 | FID at N=5000–10000 for the 3–4 headline rows | Blunts the "N=1000 FID is biased" objection on the rows that matter. |

### 4.3 Scale-up (upgraded to near-must under the improvement framing)

| # | Experiment | Why |
|---|---|---|
| E13 | Head-to-head vs CFG-MP+ on a **public pretrained model from their evaluation family** (DiT-XL/2 ImageNet-256 is their main setting; SiT variants also fine). Sampling-only compute — feasible on the 5090 (their K=2 corrector triples per-sample cost; budget FID at 10k samples). | An improvement claim over a published ICML method is only credible if demonstrated **on that method's own benchmark**, not only on our CelebA-64 testbed. This experiment decides whether the paper can target a main conference (ICLR 2027 / AISTATS 2027) instead of workshop/TMLR. |

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

- **Compute reality (2026-07-30)**: course-era numbers are lost; development happens on a MacBook (no meaningful compute), experiments run over SSH on an RTX 5090 server. Therefore: all sweeps must be scripted, resumable, and manifest-driven *before* the server session, so GPU time is pure execution.
- **Weeks 1–2 (to ~Aug 13)**: land the restructure branch; implement B1–B4 as sampler methods + the one-shot experiment runner (in progress); pick the method's new name; then on the 5090: E1–E2 + E4 + E7 at one checkpoint **and the two go/no-go baselines B1–B2** — the outcome decides whether the framing holds before any writing starts.
- **Weeks 3–4 (to ~Aug 27)**: E3, E5, E6, B3 (CFG-MP+ comparison). Freeze the 4-page workshop version; **submit to a NeurIPS workshop by Aug 29**.
- **Weeks 5–7**: E8–E12; start E13 if a pretrained backbone is chosen.
- **Weeks 8–9**: writing, figures (headline: FID-vs-accuracy curves with NFE annotations, both arms), TMLR submission.

## 7. Known risks

1. **The head-to-head must actually be won.** Framed as an improvement over CFG-MP+ (per the user's directive — they are known, published prior work, cited as such; no concurrent-work framing), the paper stands or falls on B3/E13: matched-quality-at-lower-NFE or better-quality-at-matched-NFE against their operator, on our testbed *and* on their benchmark family. If the single-branch corrector loses at matched NFE, the fallback contribution is "time-gating as a universal plug-in for projection correctors" (shown on their method) + the theory — still a paper, but a smaller one. Rename our method before anything is public ("CFG-MP" is their name).
2. **The result itself** — no quantitative results survive in the repo; the central curve has never actually been measured with its baseline arm. The paper only exists if E1/E2 show separation.
3. **FID noise at small N** — mitigated by E11/E12; avoid claiming deltas within noise.
4. **Single-dataset, single-model evidence** — CelebA-64 + DiT-B/2 alone caps the venue at workshop/TMLR; E13 is the lever.
5. **Baseline demands** — expect reviewers to ask for CFG++, limited-interval guidance, and possibly autoguidance as baselines, not just vanilla CFG. Budget for at least the first two (both are cheap, training-free changes to the sampler).
