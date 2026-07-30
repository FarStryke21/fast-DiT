# legacy/ — inherited upstream code

This directory holds code and assets inherited from the upstream forks (Meta DiT → `chuanyangjin/fast-DiT`), kept only for provenance and attribution.

**Nothing in this directory is used by the project.** It is not imported, not executed, and not part of any training, sampling, or evaluation path. The live code lives at the repository root.

In particular, **do not cite `performance/` as project results** — those are upstream's May-2023 ImageNet training-throughput SLURM logs on a DDPM objective, unrelated to this project's CelebA-64 flow-matching experiments. The same applies to `visuals/` (upstream ImageNet sample grids) and `README_upstream.md`.

`diffusion/` is the IDDPM Gaussian-diffusion package; this project trains with flow matching and never calls it. `sample.py`, `sample_ddp.py`, `extract_features.py`, and `download.py` assume a latent-space VAE pipeline incompatible with this project's pixel-space checkpoints.
