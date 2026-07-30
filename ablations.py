import os
import argparse
import subprocess
import json
import pandas as pd
from datetime import datetime

# --- ABLATION CONFIGURATION ---
REAL_DIR = "./data/local_celeba/real_images"
CLASSIFIER = "FarStryke21/celeba-resnet18-classifier"

SAMPLES = 1000
BATCH = 100
STEPS = 50

# The parameters you want to sweep
CFG_SCALES = [2.0, 4.0, 6.0, 8.0]
GATING_WINDOWS = [
    (0.0, 1.0, "Full"),
    (0.0, 0.3, "Early"),
    (0.3, 0.7, "Middle"),
    (0.7, 1.0, "Late")
]

def run_command(command):
    print(f"\nExecuting: {' '.join(command)}")
    result = subprocess.run(command, text=True)
    if result.returncode != 0:
        print(f"Error executing command: {' '.join(command)}")
        exit(1)

def generate_and_evaluate(ckpt, method, out_dir, w, extra_gen_args=()):
    fake_dir = os.path.join(out_dir, "fake")

    gen_cmd = [
        "python", "sample_generator.py",
        "--method", method,
        "--ckpt", ckpt,
        "--num-samples", str(SAMPLES),
        "--batch-size", str(BATCH),
        "--cfg-scale", str(w),
        "--num-steps", str(STEPS),
        "--out-dir", out_dir,
        *extra_gen_args,
    ]
    run_command(gen_cmd)

    eval_cmd = [
        "python", "evaluate_metrics.py",
        "--fake-dir", fake_dir,
        "--real-dir", REAL_DIR,
        "--classifier", CLASSIFIER,
        "--out-dir", out_dir,
    ]
    run_command(eval_cmd)

def collect_row(out_dir, ablation, w, gate_label):
    eval_file = os.path.join(out_dir, "evaluation_results.json")
    stats_file = os.path.join(out_dir, "generation_stats.json")

    if not (os.path.exists(eval_file) and os.path.exists(stats_file)):
        print(f"WARNING: missing results in {out_dir} — row skipped")
        return None

    with open(eval_file, "r") as f: eval_data = json.load(f)
    with open(stats_file, "r") as f: stats_data = json.load(f)

    return {
        "Ablation": ablation,
        "CFG_Scale (w)": w,
        "Gate_Window": gate_label,
        "NFE": stats_data["avg_nfe_per_sample"],
        "FID": round(eval_data.get("fid_score", 0), 4),
        "Exact_Acc (%)": round(eval_data.get("exact_match_accuracy", 0) * 100, 2),
        "Elem_Acc (%)": round(eval_data.get("elementwise_accuracy", 0) * 100, 2)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="results/000-DiT-B-2/checkpoints/0072000.pt")
    args = parser.parse_args()

    print("==================================================")
    print("STARTING CFG-MP+ ABLATION SWEEP")
    print(f"Checkpoint: {args.ckpt}")
    print("==================================================")

    all_results = []

    # Loop 1: CFG Scale Sweep — vanilla CFG baseline vs gated corrector at every w,
    # so the trade-off curve has both arms
    print("\n--- Phase 1: Sweeping CFG Scales ---")
    optimal_tmin, optimal_tmax = 0.3, 0.7

    for w in CFG_SCALES:
        out_dir = f"samples_cfg_w{w}_steps{STEPS}"
        generate_and_evaluate(args.ckpt, "cfg", out_dir, w)
        row = collect_row(out_dir, "CFG Scale", w, "Vanilla (no corrector)")
        if row: all_results.append(row)

        out_dir = f"samples_cfg_mp_anderson_gated_Middle_w{w}_steps{STEPS}"
        generate_and_evaluate(
            args.ckpt, "cfg_mp_anderson_gated", out_dir, w,
            extra_gen_args=("--tmin", str(optimal_tmin), "--tmax", str(optimal_tmax)),
        )
        row = collect_row(out_dir, "CFG Scale", w, "Middle [0.3, 0.7]")
        if row: all_results.append(row)

    # Loop 2: Time-Gating Sweep (using a standard CFG scale of 4.0)
    print("\n--- Phase 2: Sweeping Time-Gating Windows ---")
    standard_w = 4.0

    for tmin, tmax, name in GATING_WINDOWS:
        # Skip the 'Middle' gate if we already ran it in Phase 1 for w=4.0
        if name == "Middle" and standard_w in CFG_SCALES:
            continue

        out_dir = f"samples_cfg_mp_anderson_gated_{name}_w{standard_w}_steps{STEPS}"
        generate_and_evaluate(
            args.ckpt, "cfg_mp_anderson_gated", out_dir, standard_w,
            extra_gen_args=("--tmin", str(tmin), "--tmax", str(tmax)),
        )
        row = collect_row(out_dir, "Time Gating", standard_w, f"{name} [{tmin}, {tmax}]")
        if row: all_results.append(row)

    # --- SAVE AND PRINT RESULTS ---
    print("\n==================================================")
    print("ABLATION SWEEP COMPLETE")
    print("==================================================")

    df = pd.DataFrame(all_results)

    # Sort for cleaner output
    df = df.sort_values(by=["Ablation", "CFG_Scale (w)", "Gate_Window"])

    print("\n" + df.to_string(index=False))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"ablation_summary_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults successfully saved to {csv_path}")

if __name__ == "__main__":
    main()
