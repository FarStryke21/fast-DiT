# CFG with Manifold Projection — Project Wiki

CMU 10-799 project. A fork of [`chuanyangjin/fast-DiT`](https://github.com/chuanyangjin/fast-DiT) (itself a fork of Meta's DiT) that has been substantially rewritten into a **flow-matching, pixel-space, multi-attribute-conditional DiT on CelebA-64**, used as a testbed for one research question:

> Classifier-Free Guidance (CFG) at high guidance scale pushes samples *off* the data manifold, trading fidelity (FID) for conditioning strength (attribute accuracy). Can a cheap, training-free **corrector step** that projects the guided iterate back toward the unconditional model's own manifold recover fidelity without giving up controllability — and can **Anderson acceleration** plus **time-gating** make that corrector cheap enough in NFE to be worth it?

Fork remote: `https://github.com/FarStryke21/fast-DiT.git`, branch `main`, HEAD `4d43a3e` ("Cleanup and Final Push").

---

## Pages

| Page | Contents |
|---|---|
| [Method](Method.md) | The math: flow matching, CFG, the manifold-projection corrector, Anderson acceleration, time gating |
| [Architecture-and-Training](Architecture-and-Training.md) | DiT-B/2 modifications, `MultiLabelEmbedder`, flow-matching training loop, dataset pipeline |
| [Sampling-Scripts](Sampling-Scripts.md) | The five samplers, every CLI flag, NFE accounting |
| [Evaluation-and-Ablations](Evaluation-and-Ablations.md) | FID + attribute-accuracy protocol, the two ablation axes, exact sweep configurations |
| [Codebase-Map](Codebase-Map.md) | File-by-file: what is live, what is inherited dead code from upstream |
| [Runbook](Runbook.md) | End-to-end reproduction commands, artifacts, HF assets |
| [Gotchas-and-Known-Issues](Gotchas-and-Known-Issues.md) | Verified bugs, silent inconsistencies, and unexposed hyperparameters |
| [Results](Results.md) | What was measured, what is (and is not) committed to the repo |
| [Design-Decisions](Design-Decisions.md) | Running log of non-obvious choices, alternatives considered, rationale |
| [Publication-Plan](Publication-Plan.md) | Positioning vs related work, required experiments, venue targets and dates |
| [Theory-Notes](Theory-Notes.md) | Fixed-point analysis: contraction rates, stale-anchor bias, CFG-MP comparison, gating rationale |

---

## The one-paragraph summary

The model is a **DiT-B/2 trained with rectified-flow / flow matching directly in 64×64 RGB pixel space** (the VAE was deliberately removed), conditioned on CelebA's **40 binary attributes** through a small MLP embedder with a learned `null_token` for CFG dropout. Sampling is a plain **forward-Euler ODE integration from t=0 (noise) to t=1 (data)**. On top of vanilla CFG, each Euler step is followed by an optional **corrector**: one or more extra *unconditional* model evaluations at the post-step latent, used to null out the residual between the unconditional velocity at the new point and the unconditional velocity at the old point. Three corrector variants exist — fixed-point iteration (`cfg_mp_std`), **Anderson-accelerated** fixed-point iteration (`cfg_mp_anderson`), and **time-gated Anderson** (`cfg_mp_anderson_gated`), which only pays the corrector cost inside a window `[tmin, tmax]`. Everything is evaluated against an unconditional baseline and a vanilla-CFG baseline on 1000 samples with **FID** (vs. real CelebA-64) and **attribute accuracy** from a fine-tuned ResNet-18 CelebA classifier, with **NFE logged per run** so quality is always read against compute.

## Key coordinates (memorize these)

- **Time convention**: `t=0` is pure noise, `t=1` is data. `x_t = (1-t)·x_0 + t·x_1`, target velocity `v = x_1 - x_0`. **This is inverted relative to standard DDPM notation** — "Early" gating means near-noise, "Late" means near-image.
- **Backbone**: `DiT-B/2`, depth 12, hidden 768, 12 heads, patch 2, `input_size=64`, `in_channels=3` → **1024 tokens**.
- **`learn_sigma=True` is still on**, so the model emits 6 channels and every call does `.chunk(2, dim=1)` and throws the second half away.
- **Conditioning**: 40-dim float multi-hot → `MultiLabelEmbedder` MLP; unconditional = a learned `null_token`, selected via `force_drop_ids`.
- **Default sweep point**: 1000 samples, batch 100, 50 Euler steps, cfg-scale 4.0, seed 50.
- **NFE at 50 steps**: uncond 50 · vanilla CFG 100 · `cfg_mp_std` (K=3) 200 · `cfg_mp_anderson` 200 · gated-Middle 142.
- **Checkpoints referenced in scripts**: `results/000-DiT-B-2/checkpoints/0066000.pt` (`evaluation.sh`), `0072000.pt` (`ablations.py`, `time_ablation.sh`), `0100000.pt` (`hf_push.py`).
- **HF assets**: dataset `electronickale/cmu-10799-celeba64-subset`, classifier `FarStryke21/celeba-resnet18-classifier`, model `FarStryke21/cmu-10799-dit-b2`.
