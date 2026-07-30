# Evaluation and Ablations

## 1. Metrics (`evaluate_metrics.py`)

Two axes, deliberately opposed — the whole thesis is about the trade-off between them.

### Fidelity: FID

```python
calculate_fid_given_paths([real_dir, fake_dir], batch_size=50, device=..., dims=2048)
```

`pytorch_fid`, standard InceptionV3 pool3 (2048-d). Real set is `./data/local_celeba/real_images` produced by `download_dataset.py`. Device selection falls back `cuda → mps → cpu`, and **forces `cpu` for FID when the device is `mps`** (`evaluate_metrics.py:35`) because the Inception path is unreliable on Metal.

⚠️ **FID at N=1000 is heavily biased upward and high-variance.** Absolute values here are not comparable to any published CelebA FID; only the *relative ordering across methods* is meaningful, and only because the noise and conditions are seed-matched across runs. Any claimed difference smaller than a few FID points should be treated as noise.

### Controllability: attribute accuracy

A ResNet-18 fine-tuned on CelebA, pulled from the Hub via `PyTorchModelHubMixin`:

```python
class CelebAResNet(nn.Module, PyTorchModelHubMixin):
    self.resnet = resnet18(weights=IMAGENET1K_DEFAULT); self.resnet.fc = nn.Linear(512, 40)

CelebAResNet.from_pretrained("FarStryke21/celeba-resnet18-classifier")
```

Generated PNGs are read, resized **64 → 224**, ImageNet-normalised, and scored; `preds = sigmoid(logits) > 0.5`. Two numbers come out:

| Metric | Definition | Character |
|---|---|---|
| `exact_match_accuracy` | `sklearn.accuracy_score(y_true, y_pred)` on 40-bit vectors — **all 40 attributes simultaneously correct** | Brutally strict. Expect single-digit or low-double-digit percentages. Dominated by ambiguous attributes (`Attractive`, `Oval_Face`, `Big_Lips`) that the classifier itself is noisy on. |
| `elementwise_accuracy` | `(y_true == y_pred).mean()` over all 40·N bits | The stable, reportable one. But note the **base rate is high** — most CelebA attributes are heavily imbalanced, so always-predict-majority already scores ~80%. Read deltas, never absolutes. |

Ground truth comes from `conditions.pt`, which the evaluator looks for **inside `--fake-dir` and then in its parent** (`evaluate_metrics.py:49-59`); the generator writes it to the parent. If it's missing, accuracy is skipped and only FID is reported — the intended path for the `uncond` baseline.

It asserts `len(images) == len(y_true)` and bails on mismatch, which catches the case where a sweep partially overwrote a directory.

Result JSON is written to `--out-dir` if given, else to the parent of `--fake-dir`:

```json
{"fid_score": ..., "exact_match_accuracy": ..., "elementwise_accuracy": ...}
```

### The third axis: NFE

Read from `generation_stats.json` (see [Method §4](Method.md#4-nfe-accounting)). Every quality claim in this project is meant to be read as a point in **(NFE, FID, accuracy)** space — a corrector that improves FID by burning 2× the compute is only interesting if it beats vanilla CFG *at matched NFE*, which is why the gated variant exists.

---

## 2. Method comparison (`evaluation.sh`)

Fixed at `SAMPLES=1000, BATCH=100, STEPS=50, CFG=4.0`, checkpoint `0066000.pt`. Five runs:

| # | Method | Role |
|---|---|---|
| 1 | `uncond` | Floor for accuracy, reference for FID (no guidance ⇒ best on-manifold behaviour) |
| 2 | `cfg` | The baseline being improved on |
| 3 | `cfg_mp_std --proj-K 3` | Corrector, no acceleration |
| 4 | `cfg_mp_anderson` | Corrector + Anderson (NFE-identical to #3) |
| 5 | `cfg_mp_anderson_gated --tmin 0.3 --tmax 0.7` | The efficiency claim: 142 NFE vs 200 |

In HEAD the generation block of this script is **commented out** — it was left in evaluate-only mode after generation had already been run.

## 3. Ablation A — CFG scale (`ablations.py` Phase 1)

Sweeps `w ∈ {2.0, 4.0, 6.0, 8.0}` holding the gate at the hypothesised optimum `[0.3, 0.7]`, method `cfg_mp_anderson_gated`, checkpoint `0072000.pt`.

The expected shape of the result — and the reason the sweep exists — is that vanilla CFG traces a monotone trade-off (accuracy ↑, FID ↑ as `w` grows), and the claim under test is that the corrector **bends that curve**: same accuracy at lower FID, or the FID blow-up deferred to a higher `w`. A single-`w` comparison cannot show this, which is why the full sweep matters more than any single row.

Collected per run: NFE, FID, exact-match %, element-wise %.

## 4. Ablation B — time gating (`time_ablation.sh`, `ablations.py` Phase 2)

Fixed `w = 4.0`, sweeping the window:

| Name | `[tmin, tmax]` | Phase (t=0 noise → t=1 image) | Steps in gate @ N=50 | NFE |
|---|---|---|---|---|
| Early | `[0.0, 0.3]` | Layout / coarse structure emerging from noise | 16 | 132 |
| **Middle** | `[0.3, 0.7]` | **"Speciation" — identity and attributes commit** | 21 | 142 |
| Late | `[0.7, 1.0]` | Texture, high-frequency detail | 15 | 130 |
| Full | `[0.0, 1.0]` | Everything (≡ ungated `cfg_mp_anderson`) | 50 | 200 |

**Full is a built-in control**: it should reproduce `cfg_mp_anderson` exactly, same seed, same numbers. If it doesn't, something is non-deterministic and the rest of the table is suspect.

The hypothesis is that Middle captures most of Full's benefit at ~71% of the NFE, because that's where CFG's off-manifold excursion actually does damage. Early and Late are the falsification arms: if Late ≈ Middle, the story is about detail cleanup, not semantic commitment; if Early ≈ Full, the corrector is really just fixing an initialisation artefact.

Because the comparison is inclusive on both endpoints (`tmin <= t <= tmax`), **Early and Middle both contain the step at `t=0.30`, and Middle and Late both contain `t=0.70`.** The windows are not a strict partition — a one-step overlap at each boundary. Immaterial to conclusions, worth knowing before someone recomputes the NFE table and gets 52 instead of 50.

### Use `time_ablation.sh`, not `ablations.py` Phase 2

`time_ablation.sh` passes `--out-dir` to both the generator and the evaluator, giving each gate its own `samples_gated_{Name}_w4.0/` directory. **`ablations.py` Phase 2 does not pass `--out-dir` to `sample_generator.py`** and is therefore broken — see [Gotchas §2](Gotchas-and-Known-Issues.md#2-ablationspy-phase-2-writes-to-the-wrong-directory-and-crashes). `time_ablation.sh` was added in the later commit `a0f67d2` alongside the `--out-dir` flags, i.e. it is the fix.

## 5. Output aggregation

`ablations.py` builds a `pandas` DataFrame with columns

```
Ablation | CFG_Scale (w) | Gate_Window | NFE | FID | Exact_Acc (%) | Elem_Acc (%)
```

sorted by `(Ablation, CFG_Scale, Gate_Window)`, printed and saved to `ablation_summary_{YYYYmmdd_HHMMSS}.csv`. Rows are appended **only if both** `evaluation_results.json` and `generation_stats.json` exist — a failed run vanishes from the table silently rather than erroring, so always check the row count against the expected number of configurations.
