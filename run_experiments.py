"""
Manifest-driven, resumable experiment runner for the publication sweep
(see fast-DiT.wiki/Publication-Plan.md §4 for the experiment matrix).

Goal: the GPU session (SSH, single RTX 5090) is pure execution. Every
experiment is a declarative entry in EXPERIMENTS (built by
build_experiments below); this script just walks the list, calls
sample_generator.py then evaluate_metrics.py for each, skips work that
is already done on disk, never lets one failed run kill the sweep, and
aggregates everything it touched into results_summary.csv/.json.

Usage:
    python3 run_experiments.py --ckpt results/000-DiT-B-2/checkpoints/0072000.pt
    python3 run_experiments.py --ckpt CKPT --tiers must --dry-run
    python3 run_experiments.py --ckpt CKPT --only cfg_mp_std

Design notes / judgment calls (see accompanying report for the full list):
  - Identical configs that appear under more than one experiment number in
    the plan (e.g. the w=4 gated-Middle run is E2, E4, and E5's "Middle" row
    at once) are registered ONCE — same run_id, same out_dir — so the
    generation/evaluation only happens once and resumability naturally
    de-duplicates GPU time across overlapping experiment numbers. The
    `tags` field on each entry records every experiment number it satisfies.
  - --samples/--batch/--steps are *global* overrides for entries that don't
    hardcode their own step count. E3's NFE-matched-control rows (100 and
    71 steps) intentionally keep their literal step counts regardless of
    --steps, since those specific values are the point of the experiment.
  - E11's seed-50 rows are never registered here: they are bit-for-bit the
    same config as the corresponding E1/E2/B1/B2 rows (default seed=50),
    so those existing entries already cover them. Only seed 51/52 are new.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

# pandas is only needed for the final aggregation; imported lazily there so
# --dry-run works on machines without the full environment (e.g. the dev MacBook)

# --- Fixed evaluation inputs (match ablations.py / Runbook.md) ---
REAL_DIR = "./data/local_celeba/real_images"
CLASSIFIER = "FarStryke21/celeba-resnet18-classifier"

# --- Defaults (overridable via CLI) ---
DEFAULT_SAMPLES = 1000
DEFAULT_BATCH = 100
DEFAULT_STEPS = 50

RESULTS_ROOT = "experiment_runs"

TIER_ORDER = {"must": 0, "baseline": 1, "should": 2}


def _fmt(x):
    """Render 2.0 as '2' but keep 0.3 as '0.3', for compact run_ids."""
    if isinstance(x, float) and x == int(x):
        return str(int(x))
    return str(x)


def build_experiments(samples, batch, steps):
    """Build the full EXPERIMENTS manifest (list of dicts) from the
    Publication-Plan §4 matrix. `samples`/`batch`/`steps` are the global
    defaults; individual entries may override `steps` explicitly (E3)."""

    registry = {}

    def register(tier, tag, method, w=None, run_steps=None, run_samples=None,
                 run_batch=None, seed=50, gate=None, w_window=None, K=None,
                 step_scale=None, schedule=None, fresh_anchor=False):
        eff_steps = run_steps if run_steps is not None else steps

        parts = [method]
        if w is not None:
            parts.append(f"w{_fmt(w)}")
        if gate is not None:
            parts.append(f"gate{_fmt(gate[0])}-{_fmt(gate[1])}")
        if K is not None:
            parts.append(f"K{K}")
        if step_scale is not None:
            parts.append(f"ss{_fmt(step_scale)}")
        if schedule is not None:
            parts.append(f"sched-{schedule}")
        if fresh_anchor:
            parts.append("freshanchor")
        if w_window is not None:
            parts.append(f"wwin{_fmt(w_window[0])}-{_fmt(w_window[1])}")
        parts.append(f"steps{eff_steps}")
        if seed != 50:
            parts.append(f"seed{seed}")
        run_id = "_".join(parts)

        if run_id in registry:
            # Same config already registered under a different experiment
            # number in the plan (expected — see module docstring). Record
            # the extra tag, don't create a second run.
            if tag not in registry[run_id]["tags"]:
                registry[run_id]["tags"].append(tag)
            return run_id

        registry[run_id] = {
            "run_id": run_id,
            "tier": tier,
            "tags": [tag],
            "method": method,
            "w": w,
            "steps": eff_steps,
            "samples": run_samples if run_samples is not None else samples,
            "batch": run_batch if run_batch is not None else batch,
            "seed": seed,
            "gate": gate,
            "w_window": w_window,
            "K": K,
            "step_scale": step_scale,
            "schedule": schedule,
            "fresh_anchor": fresh_anchor,
            "out_dir": os.path.join(RESULTS_ROOT, run_id),
        }
        return run_id

    EARLY = (0.0, 0.3)
    MIDDLE = (0.3, 0.7)
    LATE = (0.7, 1.0)
    FULL = (0.0, 1.0)

    # --- E1 / E2: w-sweep, vanilla CFG vs gated-Middle corrector ---
    for w in (2.0, 4.0, 6.0, 8.0):
        register("must", "E1", "cfg", w=w)
        register("must", "E2", "cfg_mp_anderson_gated", w=w, gate=MIDDLE)

    # --- E3: NFE-matched control ---
    register("must", "E3", "cfg", w=4.0, run_steps=100)   # 200 NFE
    register("must", "E3", "cfg", w=4.0, run_steps=71)    # ~142 NFE
    register("must", "E3", "cfg_mp_anderson", w=4.0)      # full Anderson, 50 steps, 200 NFE

    # --- E4: five original methods @ w=4.0, 50 steps ---
    register("must", "E4", "cfg", w=4.0)
    register("must", "E4", "uncond")
    register("must", "E4", "cfg_mp_std", w=4.0, K=3)
    register("must", "E4", "cfg_mp_anderson", w=4.0)
    register("must", "E4", "cfg_mp_anderson_gated", w=4.0, gate=MIDDLE)

    # --- E5: gating sweep (+ Full ≡ ungated-Anderson determinism check) ---
    register("must", "E5", "cfg_mp_anderson_gated", w=4.0, gate=EARLY)
    register("must", "E5", "cfg_mp_anderson_gated", w=4.0, gate=MIDDLE)
    register("must", "E5", "cfg_mp_anderson_gated", w=4.0, gate=LATE)
    register("must", "E5", "cfg_mp_anderson_gated", w=4.0, gate=FULL)

    # --- E6: Picard (cfg_mp_std) K sweep vs Anderson ---
    for K in (2, 3, 5, 8):
        register("must", "E6", "cfg_mp_std", w=4.0, K=K)

    # --- E7: uncond baseline, null-token fix ---
    register("must", "E7", "uncond")

    # --- B1: CFG++ velocity-space analogue ---
    for lam in (0.2, 0.4, 0.6, 0.8):
        register("baseline", "B1", "cfg_pp", w=lam)

    # --- B2: guidance interval (Kynkäänniemi et al.) ---
    for w in (2.0, 4.0, 6.0, 8.0):
        register("baseline", "B2", "cfg_interval", w=w, w_window=MIDDLE)

    # --- B3: CFG-MP / CFG-MP+ operator (ICML 2026) — primary head-to-head ---
    # proj_K follows the repo's K-1 convention: proj_K=3 -> 2 corrector iterations,
    # which is the CFG-MP paper's recommended FPI=2 (anderson variants always do 2 evals)
    register("baseline", "B3", "cfg_mp_icml", w=4.0, K=3)
    register("baseline", "B3", "cfg_mp_icml_anderson", w=4.0, K=3)
    register("baseline", "B3", "cfg_mp_icml_anderson_gated", w=4.0, K=3, gate=MIDDLE)
    for w in (2.0, 6.0, 8.0):  # w=4.0 arm already registered just above
        register("baseline", "B3", "cfg_mp_icml_anderson", w=w, K=3)

    # --- B4: increasing-w scheduler (Wang et al., TMLR 2024) ---
    for w in (2.0, 4.0, 6.0, 8.0):
        register("baseline", "B4", "cfg", w=w, schedule="linear")

    # --- E8: corrector step-size sweep (0.5 default is covered by E2) ---
    for ss in (0.25, 1.0):
        register("should", "E8", "cfg_mp_anderson_gated", w=4.0, gate=MIDDLE, step_scale=ss)

    # --- E9: narrower gates ---
    register("should", "E9", "cfg_mp_anderson_gated", w=4.0, gate=(0.4, 0.6))
    register("should", "E9", "cfg_mp_anderson_gated", w=4.0, gate=(0.35, 0.65))

    # --- E10: fresh vs stale anchor ---
    register("should", "E10", "cfg_mp_anderson_gated", w=4.0, gate=MIDDLE, fresh_anchor=True)

    # --- E11: seed robustness, 4 headline configs, seeds 51/52 only ---
    # (seed 50 is already covered by the E1 / E2 / B1 / B2 entries above —
    # deliberately not re-registered here.)
    for seed in (51, 52):
        register("should", "E11", "cfg", w=4.0, seed=seed)
        register("should", "E11", "cfg_mp_anderson_gated", w=4.0, gate=MIDDLE, seed=seed)
        register("should", "E11", "cfg_pp", w=0.6, seed=seed)
        register("should", "E11", "cfg_interval", w=4.0, w_window=MIDDLE, seed=seed)

    return list(registry.values())


def gen_command(entry, ckpt, python_exe):
    cmd = [
        python_exe, "sample_generator.py",
        "--method", entry["method"],
        "--ckpt", ckpt,
        "--num-samples", str(entry["samples"]),
        "--batch-size", str(entry["batch"]),
        "--num-steps", str(entry["steps"]),
        "--seed", str(entry["seed"]),
        "--out-dir", entry["out_dir"],
    ]
    if entry["w"] is not None:
        cmd += ["--cfg-scale", str(entry["w"])]
    if entry["gate"] is not None:
        cmd += ["--tmin", str(entry["gate"][0]), "--tmax", str(entry["gate"][1])]
    if entry["K"] is not None:
        cmd += ["--proj-K", str(entry["K"])]
    if entry["step_scale"] is not None:
        cmd += ["--proj-step-scale", str(entry["step_scale"])]
    if entry["schedule"] is not None:
        cmd += ["--w-schedule", entry["schedule"]]
    if entry["fresh_anchor"]:
        cmd.append("--fresh-anchor")
    if entry["w_window"] is not None:
        cmd += ["--w-tmin", str(entry["w_window"][0]), "--w-tmax", str(entry["w_window"][1])]
    return cmd


def eval_command(entry, python_exe):
    fake_dir = os.path.join(entry["out_dir"], "fake")
    return [
        python_exe, "evaluate_metrics.py",
        "--fake-dir", fake_dir,
        "--real-dir", REAL_DIR,
        "--classifier", CLASSIFIER,
        "--out-dir", entry["out_dir"],
    ]


def generation_done(entry):
    stats_path = os.path.join(entry["out_dir"], "generation_stats.json")
    fake_dir = os.path.join(entry["out_dir"], "fake")
    if not os.path.isfile(stats_path):
        return False
    n_pngs = len(glob.glob(os.path.join(fake_dir, "*.png")))
    return n_pngs == entry["samples"]


def evaluation_done(entry):
    return os.path.isfile(os.path.join(entry["out_dir"], "evaluation_results.json"))


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def gate_window_str(entry):
    if entry["gate"] is not None:
        return f"{entry['gate'][0]}-{entry['gate'][1]}"
    if entry["w_window"] is not None:
        return f"{entry['w_window'][0]}-{entry['w_window'][1]}"
    return None


def build_row(entry, status):
    stats = load_json(os.path.join(entry["out_dir"], "generation_stats.json"))
    ev = load_json(os.path.join(entry["out_dir"], "evaluation_results.json"))

    nfe = stats.get("avg_nfe_per_sample")  # never a hardcoded fallback
    fid = ev.get("fid_score")
    exact = ev.get("exact_match_accuracy")
    elem = ev.get("elementwise_accuracy")

    return {
        "run_id": entry["run_id"],
        "tier": entry["tier"],
        "method": entry["method"],
        "w": entry["w"],
        "steps": entry["steps"],
        "seed": entry["seed"],
        "gate/window": gate_window_str(entry),
        "K": entry["K"],
        "step_scale": entry["step_scale"],
        "schedule": entry["schedule"],
        "fresh_anchor": entry["fresh_anchor"],
        "NFE": nfe,
        "FID": fid,
        "exact_acc_pct": (exact * 100) if exact is not None else None,
        "elem_acc_pct": (elem * 100) if elem is not None else None,
        "status": status,
        "tags": ";".join(entry["tags"]),
    }


def run_one(entry, ckpt, python_exe, dry_run):
    t0 = time.time()
    status = "ok"

    try:
        if generation_done(entry):
            print(f"SKIP generation ({entry['run_id']}): "
                  f"{entry['samples']} PNGs already present")
        else:
            cmd = gen_command(entry, ckpt, python_exe)
            if dry_run:
                print(f"[DRY-RUN gen] {' '.join(cmd)}")
            else:
                # sample_generator.py creates out_dir/fake itself; nothing to
                # pre-create here.
                print(f"[GEN] {entry['run_id']}: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)

        if evaluation_done(entry):
            print(f"SKIP evaluation ({entry['run_id']}): "
                  f"evaluation_results.json already present")
        else:
            cmd = eval_command(entry, python_exe)
            if dry_run:
                print(f"[DRY-RUN eval] {' '.join(cmd)}")
            else:
                print(f"[EVAL] {entry['run_id']}: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
    except Exception as exc:
        status = "failed"
        print(f"[FAIL] {entry['run_id']}: {exc}")

    duration = time.time() - t0
    print(f"[RUN] run_id={entry['run_id']} tier={entry['tier']} "
          f"status={status} duration={duration:.1f}s")
    return status


def main():
    parser = argparse.ArgumentParser(
        description="Manifest-driven, resumable runner for the publication experiment sweep.")
    parser.add_argument("--ckpt", required=True, help="Path to the DiT checkpoint")
    parser.add_argument("--tiers", default="must,baseline",
                         help="Comma-separated subset of {must,baseline,should}")
    parser.add_argument("--only", default=None,
                         help="Only run entries whose run_id contains this substring")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print commands without executing them")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                         help="Default step count; E3's NFE-matched-control rows "
                              "keep their own literal step counts regardless of this")
    args = parser.parse_args()

    tiers = {t.strip().lower() for t in args.tiers.split(",") if t.strip()}
    unknown = tiers - set(TIER_ORDER)
    if unknown:
        parser.error(f"Unknown tier(s) {sorted(unknown)}; choose from {sorted(TIER_ORDER)}")

    experiments = build_experiments(args.samples, args.batch, args.steps)
    experiments = [e for e in experiments if e["tier"] in tiers]
    if args.only:
        experiments = [e for e in experiments if args.only in e["run_id"]]
    experiments.sort(key=lambda e: (TIER_ORDER[e["tier"]], e["run_id"]))

    print("=" * 70)
    print(f"Selected {len(experiments)} run(s) "
          f"(tiers={sorted(tiers)}" + (f", only={args.only!r}" if args.only else "") + ")")
    for t in ("must", "baseline", "should"):
        n = sum(1 for e in experiments if e["tier"] == t)
        if n:
            print(f"  {t:8s}: {n}")
    print("=" * 70)

    python_exe = sys.executable
    rows = []
    n_ok = 0
    n_failed = 0

    for entry in experiments:
        status = run_one(entry, args.ckpt, python_exe, args.dry_run)
        if status == "ok":
            n_ok += 1
        else:
            n_failed += 1
        if not args.dry_run:
            rows.append(build_row(entry, status))

    if not args.dry_run and rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv("results_summary.csv", index=False)
        with open("results_summary.json", "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nWrote results_summary.csv / results_summary.json ({len(rows)} rows)")

    print(f"\nDone: {n_ok} ok, {n_failed} failed, {len(experiments)} total")

    if n_failed > 0:
        print("One or more runs FAILED — see [FAIL] lines above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
