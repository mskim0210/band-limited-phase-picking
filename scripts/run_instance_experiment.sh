#!/bin/bash
# ============================================================================
# Second-dataset generalization experiment (INSTANCE): trains the full
# pipeline on a second archive to test whether the findings generalize.
#
# Reuses the STEAD pipeline unchanged via configs/instance_experiment.yaml:
#   preprocess INSTANCE subset -> memmap  ->  encode wavelet CR=4
#   -> train baseline (L2) + wavelet CR=4 (L3)  -> evaluate on INSTANCE test
#
# Question: does the CR=4 accuracy gain reproduce when training on INSTANCE?
#
# Usage:
#   nohup bash scripts/run_instance_experiment.sh > outputs/experiment_logs/instance_exp.log 2>&1 &
# ============================================================================

set -e

CFG="configs/instance_experiment.yaml"
MM="data/instance_memmap"
SEEDS=(42 123 7)
LOG_DIR="outputs/experiment_logs"
CKPT="outputs/checkpoints"
RES="outputs/results"
mkdir -p "$LOG_DIR" "$RES"
ts() { date "+%Y-%m-%d %H:%M:%S"; }

# --- Step 1: preprocess INSTANCE subset -> STEAD-style memmap ---
if [ -f "$MM/train_waveforms.npy" ] && [ -f "$MM/metadata.json" ]; then
    echo "[$(ts)] INSTANCE memmap exists ($MM) — skipping preprocess."
else
    echo "[$(ts)] Preprocessing INSTANCE subset -> $MM ..."
    python scripts/preprocess_instance.py --out "$MM" \
        --n-train 150000 --n-val 20000 --n-test 30000
fi

# --- Step 2: encode wavelet CR=4 ---
if [ -f "data/cs_encoded_instance/wavelet_cr4/train_waveforms.npy" ]; then
    echo "[$(ts)] INSTANCE wavelet_cr4 encoding exists — skipping."
else
    echo "[$(ts)] Encoding INSTANCE wavelet CR=4 ..."
    python scripts/preprocess_cs.py --config "$CFG" --matrix wavelet --cr 4
fi

# --- Step 3+4: train baseline (L2) and wavelet CR=4 (L3), per seed ---
for seed in "${SEEDS[@]}"; do
    BOUT="$CKPT/instance_baseline_seed${seed}"
    if [ -f "$BOUT/best_model.pt" ]; then
        echo "[$(ts)] instance baseline seed=$seed exists — skipping."
    else
        echo "[$(ts)] TRAIN instance baseline (L2) seed=$seed"
        python scripts/train_cs_picker.py --config "$CFG" --level L2 --seed "$seed" \
            --output-dir "$BOUT" 2>&1 | tee "$LOG_DIR/instance_baseline_seed${seed}.log"
    fi

    WOUT="$CKPT/instance_wavelet_cr4_seed${seed}"
    if [ -f "$WOUT/best_model.pt" ]; then
        echo "[$(ts)] instance wavelet cr4 seed=$seed exists — skipping."
    else
        echo "[$(ts)] TRAIN instance wavelet CR=4 (L3) seed=$seed"
        python scripts/train_cs_picker.py --config "$CFG" --level L3 --matrix wavelet --cr 4 \
            --seed "$seed" --output-dir "$WOUT" 2>&1 | tee "$LOG_DIR/instance_wavelet_cr4_seed${seed}.log"
    fi
done

# --- Step 5: evaluate on INSTANCE test (memmap path, not seisbench) ---
for seed in "${SEEDS[@]}"; do
    for tag in baseline wavelet_cr4; do
        C="$CKPT/instance_${tag}_seed${seed}/best_model.pt"
        if [ -f "$C" ]; then
            echo "[$(ts)] EVAL instance_${tag}_seed${seed}"
            python scripts/evaluate.py --config "$CFG" --checkpoint "$C" \
                --output "$RES/instance_train_${tag}_seed${seed}.json" \
                2>&1 | tee "$LOG_DIR/instance_eval_${tag}_seed${seed}.log"
        fi
    done
done

echo "[$(ts)] INSTANCE_EXP_DONE. Results: $RES/instance_train_*.json"
