# Codebase Map

The repo is a fork three generations deep (Meta DiT → `chuanyangjin/fast-DiT` → this project). Roughly half the files are inherited and inert. This page separates them.

## Live — written or substantially rewritten for this project

| File | Role |
|---|---|
| `models.py` | DiT backbone. **Project change**: `MultiLabelEmbedder` (lines 97-124) + wiring at line 201. Everything else is upstream. |
| `train.py` | Flow-matching training loop (lines 251-272 are the rewrite), Accelerate/bf16, EMA, resume. |
| `dataset.py` | CelebA loader with full-split RAM tensor caching, HF-hub / Arrow / local-dir modes, attribute extraction. Written for this project (commit `4f88c1d`). |
| `download_dataset.py` | Dumps the FID reference set (`real_images/`) and the conditioning pool (`attributes.pt`). |
| **`sample_generator.py`** | **The core artifact.** All five sampling methods + NFE accounting. Everything quantitative comes from here. |
| `sample-cfg-mp.py` | Qualitative grids with the MP corrector, attribute names on the CLI. |
| `sample-vanilla-cfg.py` | Qualitative grids, vanilla CFG. |
| `evaluate_metrics.py` | FID + attribute accuracy via the HF ResNet-18 classifier. |
| `evaluation.sh` | Five-method comparison at w=4.0, ckpt 66k. |
| `ablations.py` | CFG-scale sweep (vanilla + gated arms, paired) + gate sweep, aggregates to CSV. Phase 2 `--out-dir` bug fixed and vanilla baseline arm added on the restructure branch. |
| `time_ablation.sh` | Gate sweep done correctly, with `--out-dir`. |
| `hf_push.py` | Uploads `0100000.pt` to `FarStryke21/cmu-10799-dit-b2`. |

## Inherited and **inert** — do not treat as project code

> **As of the `worktree-publication-restructure` branch (2026-07-30), everything in this table lives under `legacy/`** (upstream `README.md` → `legacy/README_upstream.md`; the root `README.md` is now a new project README). Paths below are the pre-move locations at `4d43a3e`. See [Design-Decisions D8](Design-Decisions.md).

| File / dir | Status |
|---|---|
| `diffusion/` (`gaussian_diffusion.py`, `respace.py`, `timestep_sampler.py`, `diffusion_utils.py`) | Full IDDPM implementation. `create_diffusion()` is *called* in `train.py:168` but its return value is **never used** — training is flow matching. Dead. |
| `sample.py`, `sample_ddp.py` | Upstream DDPM samplers. Build a **latent-space** model (`input_size = image_size//8`, default `in_channels=4`) and decode with `AutoencoderKL`. Incompatible with the pixel-space checkpoints in this project. |
| `extract_features.py` | VAE feature pre-extraction to `.npy`. Obsolete since the VAE was removed; `dataset.py` replaced it. |
| `download.py` | Fetches Meta's pretrained ImageNet DiT-XL/2 weights. |
| `train_options/` (`train_original.py`, `train_baseline.py`, `train_features.py`, `train_amp.py`, `train_tf32_disabled.py`, `models_original.py`) | Upstream fast-DiT's *training-speed* ablations (AMP vs bf16, TF32 on/off, pre-extracted features). Unrelated to this project's research question. |
| `performance/A100/`, `performance/2A100/` | SLURM logs from **May 2023**, present in the repo's `Initial commit`. Upstream's ImageNet throughput benchmarks (`Train Steps/Sec` 0.52–1.33, loss ~0.17). **Not this project's training runs** — do not cite them as results. |
| `run_DiT.ipynb` | Upstream Colab demo for pretrained DiT-XL/2. |
| `README.md` | Still the **upstream fast-DiT README** — describes ImageNet, the VAE pipeline, `sample.py`, and `train_options/`. It documents almost none of this project. Treat this wiki as the real documentation. |
| `visuals/sample_grid_{0,1}.png` | Upstream ImageNet DiT samples used by that README. |
| `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `LICENSE.txt` (CC-BY-NC 4.0), `environment.yml` | Upstream boilerplate. `environment.yml` is stale — it lacks `datasets`, `pytorch_fid`, `huggingface_hub`, `scikit-learn`, `pandas`. |

## Artifacts

| Path | Contents |
|---|---|
| `images/` | Nine committed qualitative grids (see [Sampling-Scripts](Sampling-Scripts.md#committed-qualitative-artifacts)) |
| repo root `*.png` | Final-push triptych: `vanilla_cfg_Male_Chubby_Blond_Hair_3.png`, `_8.png`, `cfg_mp_anderson_Male_Chubby_Blond_Hair.png` |
| `chkp66/` | Empty but for a `.gitignore` containing `*.png` — a scratch dir for checkpoint-66k sample grids |

## What `.gitignore` excludes (and why there are no numbers in the repo)

```
results        # all training checkpoints
features
data           # dataset cache, real_images/, attributes.pt
*.json         # generation_stats.json AND evaluation_results.json
*.pt           # checkpoints AND conditions.pt
samples_*/     # every generated sample directory
```

`*.json` catching `evaluation_results.json` is the consequential one: **no measured FID or accuracy value has ever been committed to this repository.** `ablation_summary_*.csv` is not ignored, but none was committed either. See [Results](Results.md).

## Commit narrative

```
cfa344e  Flow Matching and CelebA update      ← pivot away from upstream ImageNet/DDPM
0878d58  Extract features updated
9e38093  training setup
675ab37  Prepping for Implementations
53caf18  MP Implementation                    ← manifold projection lands
25612f3  Mp
4f88c1d  Remvoe VAE encoding                  ← dataset.py written, pixel space
f97983a  Clean Up
82dc7b4  some experiments                     ← sample_generator.py created
0621ce4  Results                              ← evaluate_metrics.py, evaluation.sh, download_dataset.py;
                                                sample-cfg-mp-time.py deleted (gating folded in)
d1ca0f2  Added Ablations                      ← ablations.py
a0f67d2  Final Submit                         ← --out-dir flags + time_ablation.sh (fixes gate sweep)
4d43a3e  Cleanup and Final Push               ← final qualitative grids, hf_push points at 0100000.pt
```
