# Results

## Status: no quantitative results are committed to this repository

This is a factual statement about the repo, not a claim about whether the experiments were run. `.gitignore` contains `*.json`, `*.pt`, `results`, `data`, and `samples_*/`, which excludes every file that carries a number:

- `samples_*/evaluation_results.json` — FID, exact-match accuracy, element-wise accuracy
- `samples_*/generation_stats.json` — NFE
- `ablation_summary_*.csv` — not gitignored, but none was ever committed

Confirmed by searching the full git history for added `.json` / `.csv` / `.out` files: the only ones that ever existed are `.vscode/settings.json` and the upstream `performance/*.out` SLURM logs.

**Do not source numbers from `performance/A100/*.out` or `performance/2A100/*.out`.** Those are dated May 2023, arrived in the repo's `Initial commit`, and are upstream fast-DiT's ImageNet *training-throughput* benchmarks (0.52–1.33 steps/sec, loss ≈ 0.17 on the DDPM objective). They have nothing to do with CFG, manifold projection, or CelebA.

If the ablation CSVs still exist on the machine or cluster where the sweeps ran, they are the only surviving source — recover them before regenerating, since a re-run costs the full sweep.

---

## Derived-from-code numbers (safe to cite)

These follow from the implementation, not from measurement, and are exact.

**NFE per sample at `--num-steps 50`:**

| Method | Configuration | NFE |
|---|---|---|
| `uncond` | — | 50 |
| `cfg` | — | 100 |
| `cfg_mp_std` | K=3 | 200 |
| `cfg_mp_anderson` | — | 200 |
| `cfg_mp_anderson_gated` | Full `[0.0, 1.0]` | 200 |
| `cfg_mp_anderson_gated` | **Middle `[0.3, 0.7]`** | **142** |
| `cfg_mp_anderson_gated` | Early `[0.0, 0.3]` | 132 |
| `cfg_mp_anderson_gated` | Late `[0.7, 1.0]` | 130 |

The efficiency headline the design is built around: **gated-Middle costs 142 NFE vs 200 for full Anderson — a 29% reduction** — and the open question the sweep was meant to answer is how much of Full's FID/accuracy benefit survives that cut.

---

## Qualitative evidence that *is* in the repo

Nine grids in `images/` plus three at the repo root. The most informative set is the fixed-attribute triptych at `Male ∧ Chubby ∧ Blond_Hair`, seed 50, 4×4 grids:

| File | Setting |
|---|---|
| `vanilla_cfg_Male_Chubby_Blond_Hair_3.png` | vanilla CFG, w≈3 |
| `vanilla_cfg_Male_Chubby_Blond_Hair_8.png` | vanilla CFG, w≈8 |
| `cfg_mp_anderson_Male_Chubby_Blond_Hair.png` | CFG + Anderson corrector |
| `images/cfg_mp_standard_Male_Chubby_Blond_Hair.png` | CFG + Picard corrector |

(The `_3` / `_8` suffixes are manual renames — the scripts don't encode the scale in the filename. They read as CFG scales, and the visual evidence supports it.)

**What the grids show, described plainly:** the two vanilla grids are at matched seed and produce the same 16 identities in the same layout. At the low scale, skin tone, lighting, and backgrounds are varied and photographic. At the high scale the same 16 faces come back **markedly more saturated and contrast-heavy**, with pushed skin tones, harder edges, and backgrounds collapsing toward flat blocks of colour — the textbook high-guidance off-manifold signature the method is designed to counter. The Anderson-corrected grid sits closer to the low-scale grid in colour and background naturalism while keeping the conditioned attributes legible.

This is a visual impression from the committed PNGs, not a measurement. It is consistent with the hypothesis but is not evidence for it — FID and attribute accuracy are what decide the question, and neither is in the repo.

---

## What a complete results section needs

1. **Method comparison table** — 5 methods × (NFE, FID, exact-match %, element-wise %) at w=4.0, one checkpoint. From `evaluation.sh` (§4 of the [Runbook](Runbook.md)).
2. **CFG-scale curve** — FID vs element-wise accuracy traced over `w ∈ {2,4,6,8}`, plotted for vanilla CFG *and* for the corrector. **The vanilla arm is currently missing**: `ablations.py` Phase 1 only sweeps `cfg_mp_anderson_gated`. Without a vanilla `w` sweep there is no curve to compare against and the central claim ("the corrector bends the trade-off") cannot be shown. **This is the single most important gap.**
3. **Gating table** — Early / Middle / Late / Full × (NFE, FID, accuracy) at w=4.0. From `time_ablation.sh`. Verify Full ≡ `cfg_mp_anderson`.
4. **Checkpoint consistency** — the comparison used 66k, the ablations 72k, the Hub upload 100k. Re-run §1 at 72k or label the checkpoint on every table ([Gotchas §12](Gotchas-and-Known-Issues.md#12-checkpoint-inconsistency-across-scripts)).
5. **Caveats to state**: FID at N=1000 (biased, ranking-only); the `uncond` baseline uses the zero-attribute vector rather than the null token ([Gotchas §1](Gotchas-and-Known-Issues.md#1-the-uncond-baseline-does-not-use-the-null-token)); `cfg_mp_std` at K=3 is NFE-identical to Anderson, so that comparison is quality-at-matched-compute, not a speedup.

## Cheapest additional experiments, by value

| Experiment | Cost | Why |
|---|---|---|
| Vanilla-CFG `w` sweep at {2,4,6,8} | 4 runs × 100 NFE | Closes gap #2 — without it there is no baseline curve |
| `cfg_mp_std` with K ∈ {2,3,5} | 3 runs | The only way to substantiate the Anderson claim: does Picard need K−1 ≫ 2 to match Anderson's 2 evals? |
| Corrector step size `0.5·dt` ∈ {0.25, 0.5, 1.0} | 3 runs + a CLI flag | Currently an unswept, unexposed hyperparameter ([Gotchas §7](Gotchas-and-Known-Issues.md#7-corrector-step-size-05dt-is-hard-coded)) |
| Narrower gates (`[0.4,0.6]`, `[0.35,0.65]`) | 2 runs | Pushes the NFE saving below 142 and finds where the benefit actually breaks |
| NFE-matched control: vanilla CFG at 100 steps (200 NFE) vs `cfg_mp_anderson` at 50 steps (200 NFE) | 1 run | **The strongest control available.** Answers "is the corrector better than just taking twice as many Euler steps?" — a reviewer will ask this first, and nothing in the current sweep addresses it. |
