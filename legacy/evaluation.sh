#!/bin/bash

# Exit immediately if any command fails
set -e

# Define common variables to keep things clean
CKPT="results/000-DiT-B-2/checkpoints/0066000.pt"
REAL_DIR="./data/local_celeba/real_images"
CLASSIFIER="FarStryke21/celeba-resnet18-classifier" # <-- Update this to your actual HF model!
SAMPLES=1000
BATCH=100
STEPS=50
CFG=4.0

# echo -e "\n=================================================="
# echo "STEP 1: GENERATING SAMPLES ACROSS ALL METHODS"
# echo "=================================================="

# # 1. Unconditional Baseline
# python sample_generator.py --method uncond --ckpt $CKPT --num-samples $SAMPLES --batch-size $BATCH --num-steps $STEPS

# # 2. Standard CFG Baseline
# python sample_generator.py --method cfg --ckpt $CKPT --num-samples $SAMPLES --batch-size $BATCH --cfg-scale $CFG --num-steps $STEPS

# # 3. CFG + Standard Manifold Projection (K=3)
# python sample_generator.py --method cfg_mp_std --ckpt $CKPT --num-samples $SAMPLES --batch-size $BATCH --cfg-scale $CFG --num-steps $STEPS --proj-K 3

# # 4. CFG + Anderson Acceleration
# python sample_generator.py --method cfg_mp_anderson --ckpt $CKPT --num-samples $SAMPLES --batch-size $BATCH --cfg-scale $CFG --num-steps $STEPS

# # 5. CFG + Time-Gated Anderson Acceleration
# python sample_generator.py --method cfg_mp_anderson_gated --ckpt $CKPT --num-samples $SAMPLES --batch-size $BATCH --cfg-scale $CFG --num-steps $STEPS --tmin 0.3 --tmax 0.7


echo -e "\n=================================================="
echo "STEP 3: EVALUATING FID AND ACCURACY METRICS"
echo "=================================================="

# Evaluate Unconditional
echo -e "\n---> Evaluating UNCONDITIONAL"
python evaluate_metrics.py --fake-dir samples_uncond_w${CFG}_steps${STEPS}/fake --real-dir $REAL_DIR --classifier $CLASSIFIER

# Evaluate Standard CFG
echo -e "\n---> Evaluating STANDARD CFG"
python evaluate_metrics.py --fake-dir samples_cfg_w${CFG}_steps${STEPS}/fake --real-dir $REAL_DIR --classifier $CLASSIFIER

# Evaluate CFG + Standard MP
echo -e "\n---> Evaluating CFG + STANDARD MP"
python evaluate_metrics.py --fake-dir samples_cfg_mp_std_w${CFG}_steps${STEPS}/fake --real-dir $REAL_DIR --classifier $CLASSIFIER

# Evaluate CFG + Anderson MP
echo -e "\n---> Evaluating CFG + ANDERSON MP"
python evaluate_metrics.py --fake-dir samples_cfg_mp_anderson_w${CFG}_steps${STEPS}/fake --real-dir $REAL_DIR --classifier $CLASSIFIER

# Evaluate CFG + Gated Anderson MP
echo -e "\n---> Evaluating CFG + GATED ANDERSON MP"
python evaluate_metrics.py --fake-dir samples_cfg_mp_anderson_gated_w${CFG}_steps${STEPS}/fake --real-dir $REAL_DIR --classifier $CLASSIFIER

echo -e "\n=================================================="
echo "SWEEP COMPLETE! Check generation_stats.json in each folder for NFE metrics."
# echo "=================================================="