import os
import subprocess
import json
import pandas as pd
from datetime import datetime

# --- ABLATION CONFIGURATION ---
CKPT = "results/000-DiT-B-2/checkpoints/0072000.pt"
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

def main():
    print("==================================================")
    print("STARTING CFG-MP+ ABLATION SWEEP")
    print("==================================================")

    all_results = []

    # Loop 1: CFG Scale Sweep (using the optimal 'Middle' gate)
    print("\n--- Phase 1: Sweeping CFG Scales ---")
    optimal_tmin, optimal_tmax = 0.3, 0.7
    
    for w in CFG_SCALES:
        out_dir = f"samples_cfg_mp_anderson_gated_w{w}_steps{STEPS}"
        fake_dir = os.path.join(out_dir, "fake")
        
        # 1. Generate
        gen_cmd = [
            "python", "sample_generator.py",
            "--method", "cfg_mp_anderson_gated",
            "--ckpt", CKPT,
            "--num-samples", str(SAMPLES),
            "--batch-size", str(BATCH),
            "--cfg-scale", str(w),
            "--num-steps", str(STEPS),
            "--tmin", str(optimal_tmin),
            "--tmax", str(optimal_tmax)
        ]
        run_command(gen_cmd)

        # 2. Evaluate
        eval_cmd = [
            "python", "evaluate_metrics.py",
            "--fake-dir", fake_dir,
            "--real-dir", REAL_DIR,
            "--classifier", CLASSIFIER
        ]
        run_command(eval_cmd)

        # 3. Collect Data
        eval_file = os.path.join(out_dir, "evaluation_results.json")
        stats_file = os.path.join(out_dir, "generation_stats.json")
        
        if os.path.exists(eval_file) and os.path.exists(stats_file):
            with open(eval_file, "r") as f: eval_data = json.load(f)
            with open(stats_file, "r") as f: stats_data = json.load(f)
            
            all_results.append({
                "Ablation": "CFG Scale",
                "CFG_Scale (w)": w,
                "Gate_Window": "Middle [0.3, 0.7]",
                "NFE": stats_data.get("avg_nfe_per_sample", 140),
                "FID": round(eval_data.get("fid_score", 0), 4),
                "Exact_Acc (%)": round(eval_data.get("exact_match_accuracy", 0) * 100, 2),
                "Elem_Acc (%)": round(eval_data.get("elementwise_accuracy", 0) * 100, 2)
            })

    # Loop 2: Time-Gating Sweep (using a standard CFG scale of 4.0)
    print("\n--- Phase 2: Sweeping Time-Gating Windows ---")
    standard_w = 4.0
    
    for tmin, tmax, name in GATING_WINDOWS:
        # Skip the 'Middle' gate if we already ran it in Phase 1 for w=4.0
        if name == "Middle" and standard_w in CFG_SCALES:
            continue 
            
        out_dir = f"samples_cfg_mp_anderson_gated_{name}_w{standard_w}_steps{STEPS}"
        fake_dir = os.path.join(out_dir, "fake")
        
        # 1. Generate
        gen_cmd = [
            "python", "sample_generator.py",
            "--method", "cfg_mp_anderson_gated",
            "--ckpt", CKPT,
            "--num-samples", str(SAMPLES),
            "--batch-size", str(BATCH),
            "--cfg-scale", str(standard_w),
            "--num-steps", str(STEPS),
            "--tmin", str(tmin),
            "--tmax", str(tmax)
        ]
        run_command(gen_cmd)

        # 2. Evaluate
        eval_cmd = [
            "python", "evaluate_metrics.py",
            "--fake-dir", fake_dir,
            "--real-dir", REAL_DIR,
            "--classifier", CLASSIFIER
        ]
        run_command(eval_cmd)

        # 3. Collect Data
        eval_file = os.path.join(out_dir, "evaluation_results.json")
        stats_file = os.path.join(out_dir, "generation_stats.json")
        
        if os.path.exists(eval_file) and os.path.exists(stats_file):
            with open(eval_file, "r") as f: eval_data = json.load(f)
            with open(stats_file, "r") as f: stats_data = json.load(f)
            
            all_results.append({
                "Ablation": "Time Gating",
                "CFG_Scale (w)": standard_w,
                "Gate_Window": f"{name} [{tmin}, {tmax}]",
                "NFE": stats_data.get("avg_nfe_per_sample", 0),
                "FID": round(eval_data.get("fid_score", 0), 4),
                "Exact_Acc (%)": round(eval_data.get("exact_match_accuracy", 0) * 100, 2),
                "Elem_Acc (%)": round(eval_data.get("elementwise_accuracy", 0) * 100, 2)
            })

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