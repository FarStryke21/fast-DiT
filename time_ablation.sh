#!/bin/bash

# --- CONFIGURATION ---
CKPT="results/000-DiT-B-2/checkpoints/0072000.pt"
REAL_DIR="./data/local_celeba/real_images"
CLASSIFIER="FarStryke21/celeba-resnet18-classifier"

SAMPLES=1000
BATCH=100
STEPS=50
CFG=4.0

# --- TIME GATING ARRAYS ---
NAMES=("Early" "Middle" "Late" "Full")
TMINS=(0.0 0.3 0.7 0.0)
TMAXS=(0.3 0.7 1.0 1.0)

echo "=================================================="
echo "STARTING TIME-GATING ABLATION SWEEP"
echo "=================================================="

# Loop through the 4 gating configurations
for i in ${!NAMES[@]}; do
    NAME=${NAMES[$i]}
    TMIN=${TMINS[$i]}
    TMAX=${TMAXS[$i]}
    
    # CRITICAL: Unique output directory for each run to prevent overwriting
    OUT_DIR="samples_gated_${NAME}_w${CFG}"
    FAKE_DIR="${OUT_DIR}/fake"
    
    echo ""
    echo "--------------------------------------------------"
    echo "Running Phase: $NAME Gate [tmin=$TMIN, tmax=$TMAX]"
    echo "Output Directory: $OUT_DIR"
    echo "--------------------------------------------------"
    
    # 1. Generate Samples
    echo "Generating samples..."
    python sample_generator.py \
        --method cfg_mp_anderson_gated \
        --ckpt $CKPT \
        --num-samples $SAMPLES \
        --batch-size $BATCH \
        --cfg-scale $CFG \
        --num-steps $STEPS \
        --tmin $TMIN \
        --tmax $TMAX \
        --out-dir $OUT_DIR  # Ensure your Python script accepts this argument!

    # 2. Evaluate Metrics
    echo "Evaluating metrics..."
    python evaluate_metrics.py \
        --fake-dir $FAKE_DIR \
        --real-dir $REAL_DIR \
        --classifier $CLASSIFIER \
        --out-dir $OUT_DIR
        
    echo "Finished $NAME Gate."
done

echo ""
echo "=================================================="
echo "ALL GATING ABLATIONS COMPLETE"
echo "=================================================="