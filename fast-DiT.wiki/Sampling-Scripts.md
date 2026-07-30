# Sampling Scripts

Three project-written samplers, plus two inherited ones that no longer work with this model.

| Script | Purpose | Output |
|---|---|---|
| `sample_generator.py` | **Canonical.** Bulk generation of N samples under any of 5 methods, with NFE logging. Everything measured comes from here. | `<out_dir>/fake/*.png`, `conditions.pt`, `generation_stats.json` |
| `sample-cfg-mp.py` | Qualitative grid for one attribute combination, with the MP corrector | `cfg_mp_{method}_{attrs}.png` |
| `sample-vanilla-cfg.py` | Qualitative grid, vanilla CFG only | `vanilla_cfg_{attrs}.png` |
| `sample.py`, `sample_ddp.py` (now `legacy/`) | **Inherited from upstream, broken here** — they build a latent-space model (`input_size = image_size//8`, `in_channels=4`), use `create_diffusion` DDPM sampling and `AutoencoderKL`. Do not use. | — |

---

## `sample_generator.py`

```bash
python sample_generator.py \
  --method {uncond,cfg,cfg_mp_std,cfg_mp_anderson,cfg_mp_anderson_gated} \
  --ckpt results/000-DiT-B-2/checkpoints/0072000.pt \
  --num-samples 1000 --batch-size 100 \
  --cfg-scale 4.0 --num-steps 50 \
  [--proj-K 3] [--tmin 0.3 --tmax 0.7] [--out-dir DIR] \
  [--attr-path ./data/local_celeba/attributes.pt] \
  [--model DiT-B/2 --image-size 64 --num-classes 40 --seed 50]
```

| Flag | Default | Notes |
|---|---|---|
| `--method` | *required* | See method table below |
| `--ckpt` | *required* | Loads `state_dict["ema"]` if present |
| `--attr-path` | `./data/local_celeba/attributes.pt` | Pool of real 40-dim attribute vectors, produced by `download_dataset.py` |
| `--num-samples` | 1000 | |
| `--batch-size` | 100 | Effective forward batch is 2× this for CFG methods |
| `--cfg-scale` | 4.0 | `w` |
| `--num-steps` | 50 | Euler steps |
| `--proj-K` | 3 | `cfg_mp_std` only → `K−1 = 2` corrector iterations |
| `--tmin` / `--tmax` | 0.3 / 0.7 | `cfg_mp_anderson_gated` only; **inclusive on both ends** |
| `--out-dir` | `samples_{method}_w{cfg}_steps{steps}` | Added late (commit `a0f67d2`) so parallel gate sweeps don't collide |
| `--seed` | 50 | Seeds both the noise and the `randperm` that picks conditions |

### Conditioning selection

```python
all_real_y = torch.load(args.attr_path)
indices = torch.randperm(len(all_real_y))[:args.num_samples]
target_conditions = all_real_y[indices]
```

Conditions are **drawn from the real attribute marginal**, not sampled independently per attribute — so the 40-bit targets are realistic co-occurring combinations, not e.g. `Bald ∧ Wearing_Lipstick ∧ Male=0`. Because the seed is fixed, every method in a sweep gets the **same conditions in the same order and the same initial noise**, making all comparisons paired. `target_conditions` is written to `<out_dir>/conditions.pt` and is the ground truth for attribute accuracy.

For `--method uncond`, `y_cond` is overwritten with zeros and no `conditions.pt` is meaningful (the evaluator handles the missing-file case by skipping accuracy).

### The five methods

| `--method` | Predictor | Corrector | NFE/step |
|---|---|---|---|
| `uncond` | `v(z, t, y=0)`, single pass | none | 1 |
| `cfg` | batched CFG | none | 2 |
| `cfg_mp_std` | batched CFG | Picard, `K−1` iterations, **always** | `K+1` |
| `cfg_mp_anderson` | batched CFG | Anderson-1, 2 evals, **always** | 4 |
| `cfg_mp_anderson_gated` | batched CFG | Anderson-1, 2 evals, **only if `tmin ≤ t ≤ tmax`** | 4 in-gate / 2 out |

The gate variable `is_in_gate` is computed for every method but only consulted by the gated one (`sample_generator.py:83-85`).

### Outputs

- `<out_dir>/fake/{00000..}.png` — one PNG per sample, `torch.clamp((z+1)/2, 0, 1)`, index-aligned with `conditions.pt`.
- `<out_dir>/conditions.pt` — `(N, 40)` float tensor.
- `<out_dir>/generation_stats.json` — `{method, num_samples, num_steps, cfg_scale, total_nfe_batch, avg_nfe_per_sample}`, plus `proj_K` for MP methods and `tmin/tmax` for gated. **This is the compute half of every quality/compute claim.**

All three are gitignored (`*.pt`, `*.json`, `samples_*/`).

---

## `sample-cfg-mp.py` — qualitative grids

```bash
python sample-cfg-mp.py --ckpt CKPT \
  --attributes Male Chubby Blond_Hair \
  --mp-method {standard,anderson} \
  --n 16 --cfg-scale 4.0 --num-steps 50 \
  [--proj-K 3] [--tmin 0.0 --tmax 1.0] [--seed 50]
```

Same predictor/corrector code as `sample_generator.py`, but conditions are **constructed by name** rather than sampled: a zero 40-vector with `1.0` written at each requested attribute index via `ATTR_MAP`. Unknown names print a warning and are ignored. Saves an `√n × √n` grid to `cfg_mp_{mp_method}_{attrs joined by _}.png`.

Two differences from the canonical script to be aware of: `--mp-method` here corresponds to `cfg_mp_std` / `cfg_mp_anderson`, and the gate defaults to `[0.0, 1.0]` (always on) rather than `[0.3, 0.7]`. Default `--num-steps` is 50.

`sample-vanilla-cfg.py` is the same script with the corrector removed and `--num-steps` defaulting to **100**. Don't compare its grids against MP grids without matching the step count.

An earlier standalone `sample-cfg-mp-time.py` (gate flags named `--t-proj-start` / `--t-proj-end`) existed and was deleted in commit `0621ce4` once gating was folded into `sample_generator.py`.

### Committed qualitative artifacts

`images/` holds the grids kept for the write-up, all at attribute set `Male, Chubby, Blond_Hair` for the CFG-scale comparison:

```
images/vanilla_cfg_Male_Chubby_Blond_Hair.png        images/cfg_mp_standard_Male_Chubby_Blond_Hair.png
images/vanilla_cfg_Male_Chubby_Blond_Hair_3.png      images/cfg_mp_anderson_Male_Chubby_Blond_Hair.png
images/vanilla_cfg_Smiling_Male_Bald_Narrow_Eyes.png
images/vanilla_cfg_Smiling_Male_Brown_Hair.png
images/vanilla_cfg_unconditional{,_24,_57}.png
```

Repo root additionally carries `vanilla_cfg_Male_Chubby_Blond_Hair_3.png`, `..._8.png`, and `cfg_mp_anderson_Male_Chubby_Blond_Hair.png` from the final push — the trailing `_3` / `_8` are **CFG scales**, not seeds, giving a w=3 vs w=8 vs MP-corrected visual triptych. `legacy/visuals/sample_grid_{0,1}.png` are upstream ImageNet DiT samples used by the original README, unrelated to this work.
