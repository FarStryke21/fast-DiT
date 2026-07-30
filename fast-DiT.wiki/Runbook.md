# Runbook

## 0. Environment

`environment.yml` is upstream's and incomplete. The actual dependency set:

```
torch, torchvision, timm, accelerate, diffusers        # model + training
datasets, huggingface_hub                              # data + classifier/model hub
pytorch_fid, scikit-learn, pandas, tqdm, pillow, numpy # evaluation + aggregation
```

## 1. Data preparation

```bash
python download_dataset.py
```

Streams `electronickale/cmu-10799-celeba64-subset` (HF cache → `./data/hf_cache`), then with `shuffle=False, augment=False` writes:

```
./data/local_celeba/real_images/{000000..}.png     # FID reference set
./data/local_celeba/attributes.pt                  # (N, 40) conditioning pool, index-aligned
```

Run this **once**; both artifacts must come from the same pass or the index alignment breaks.

## 2. Training

```bash
accelerate launch train.py \
  --model DiT-B/2 --image-size 64 --num-classes 40 \
  --feature-path ./data/hf_cache \
  --global-batch-size 256 \
  --epochs 3000 --ckpt-every 6000 --log-every 100 \
  --results-dir results
```

- `--feature-path` is the HF cache root, not a feature directory (the name is a leftover).
- bf16 via `Accelerator(mixed_precision="bf16")`; nothing to configure.
- Checkpoints → `results/000-DiT-B-2/checkpoints/{step:07d}.pt`, containing `{model, ema, opt, args}`.
- Resume: `--resume-from results/000-DiT-B-2/checkpoints/0066000.pt` (step and epoch are re-derived from the filename; logs append).
- The first epoch is slow to start — `dataset.py` decodes and caches the whole split into RAM before step 0 and prints the GB used.

Checkpoints referenced by downstream scripts: **66k** (method comparison), **72k** (ablations), **100k** (Hub upload).

## 3. Qualitative grids

```bash
CKPT=results/000-DiT-B-2/checkpoints/0072000.pt

# vanilla CFG, w=3 vs w=8 (note: default --num-steps here is 100)
python sample-vanilla-cfg.py --ckpt $CKPT --attributes Male Chubby Blond_Hair --cfg-scale 3.0 --n 16
python sample-vanilla-cfg.py --ckpt $CKPT --attributes Male Chubby Blond_Hair --cfg-scale 8.0 --n 16

# with the corrector (default --num-steps 50 — match it explicitly when comparing)
python sample-cfg-mp.py --ckpt $CKPT --attributes Male Chubby Blond_Hair \
  --mp-method anderson --cfg-scale 4.0 --num-steps 50 --n 16
python sample-cfg-mp.py --ckpt $CKPT --attributes Male Chubby Blond_Hair \
  --mp-method standard --proj-K 3 --cfg-scale 4.0 --num-steps 50 --n 16
```

Filenames are auto-derived (`vanilla_cfg_{attrs}.png`, `cfg_mp_{method}_{attrs}.png`) and **collide across CFG scales** — rename immediately (the committed `_3` / `_8` suffixes were added by hand).

## 4. Running the full experiment matrix (current entry point)

```bash
python run_experiments.py --ckpt results/000-DiT-B-2/checkpoints/0072000.pt --dry-run   # inspect the plan
python run_experiments.py --ckpt results/000-DiT-B-2/checkpoints/0072000.pt             # default tiers: must,baseline (37 runs)
python run_experiments.py --ckpt <ckpt> --tiers must,baseline,should                    # everything (50 runs)
python run_experiments.py --ckpt <ckpt> --only cfg_mp_icml                              # substring-filtered subset
```

Resumable (re-invoking skips completed runs), failure-isolated (one bad run doesn't kill the sweep), and aggregates everything into `results_summary.csv` / `results_summary.json` — both committable. §§4a–6 below document the underlying per-run commands and the retired course-era scripts.

## 4a. Five-method comparison (manual form)

```bash
CKPT=results/000-DiT-B-2/checkpoints/0072000.pt   # prefer 72k for consistency with the ablations
REAL=./data/local_celeba/real_images
CLF=FarStryke21/celeba-resnet18-classifier
S=1000; B=100; N=50; W=4.0

python sample_generator.py --method uncond               --ckpt $CKPT --num-samples $S --batch-size $B --num-steps $N
python sample_generator.py --method cfg                  --ckpt $CKPT --num-samples $S --batch-size $B --num-steps $N --cfg-scale $W
python sample_generator.py --method cfg_mp_std           --ckpt $CKPT --num-samples $S --batch-size $B --num-steps $N --cfg-scale $W --proj-K 3
python sample_generator.py --method cfg_mp_anderson      --ckpt $CKPT --num-samples $S --batch-size $B --num-steps $N --cfg-scale $W
python sample_generator.py --method cfg_mp_anderson_gated --ckpt $CKPT --num-samples $S --batch-size $B --num-steps $N --cfg-scale $W --tmin 0.3 --tmax 0.7

for M in uncond cfg cfg_mp_std cfg_mp_anderson cfg_mp_anderson_gated; do
  python evaluate_metrics.py --fake-dir samples_${M}_w${W}_steps${N}/fake --real-dir $REAL --classifier $CLF
done
```

(`legacy/evaluation.sh` did this but has the generation block commented out and points at the 66k checkpoint — retired; use `run_experiments.py`. Note also that `uncond` writes to `samples_uncond_w4.0_steps50` — the `w` in the name is the CLI default, not a guidance scale that was applied.)

Expected NFE per sample: **50 / 100 / 200 / 200 / 142**.

## 5. Ablations A (CFG scale) and B (time gating) — retired scripts

Both sweeps are rows in the `run_experiments.py` manifest (tags E1/E2 and E5). The retired course-era entry points are `legacy/ablations.py` (Phase 2 `--out-dir` bug was fixed on this branch before retirement) and `legacy/time_ablation.sh`.

Sanity check that survives the migration: the **Full-gate** `[0,1]` row must match the `cfg_mp_anderson` row exactly (200 NFE, same FID, same accuracy) — the runner includes both.

## 7. Publish the checkpoint

```bash
python hf_push.py     # uploads results/000-DiT-B-2/checkpoints/0100000.pt
                      # → FarStryke21/cmu-10799-dit-b2 as final-checkpoint.pt
```

Requires `huggingface-cli login`. Edit `local_file_path` in the file to change which step gets published.

---

## Artifacts and where they live

| Artifact | Path | Gitignored? |
|---|---|---|
| Checkpoints | `results/000-DiT-B-2/checkpoints/*.pt` | yes (`results`, `*.pt`) |
| Real images (FID ref) | `data/local_celeba/real_images/` | yes (`data`) |
| Conditioning pool | `data/local_celeba/attributes.pt` | yes |
| Generated samples | `samples_*/fake/*.png` | yes (`samples_*/`) |
| Per-run conditions | `samples_*/conditions.pt` | yes (`*.pt`) |
| NFE log | `samples_*/generation_stats.json` | yes (`*.json`) |
| Metrics | `samples_*/evaluation_results.json` | yes (`*.json`) |
| Per-run outputs (runner) | `experiment_runs/<run_id>/` | yes (`experiment_runs/`) |
| **Aggregated results** | `results_summary.csv`, `results_summary.json` | **no — commit these.** (`.gitignore` has an explicit `!results_summary.json` exception to the blanket `*.json` rule) |
| Qualitative grids | `images/` | no |

Per-run JSONs remain gitignored; the aggregated summary is the survivable artifact — **commit `results_summary.*` after every server session** so the numbers can never be lost again (course-era numbers died to the old blanket ignore).

## External assets

| | |
|---|---|
| Dataset | `electronickale/cmu-10799-celeba64-subset` (HF, 64×64 CelebA subset with 40 attributes) |
| Classifier | `FarStryke21/celeba-resnet18-classifier` (ResNet-18, 40-way multi-label, loaded via `PyTorchModelHubMixin`) |
| Model | `FarStryke21/cmu-10799-dit-b2` (`final-checkpoint.pt`, from step 100k) |
| Fork | `https://github.com/FarStryke21/fast-DiT.git` |
