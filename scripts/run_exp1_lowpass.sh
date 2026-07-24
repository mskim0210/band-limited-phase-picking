#!/bin/bash
# ============================================================================
# LP-only control: low-pass WITHOUT downsampling (denoising vs. dimensionality reduction)
#
# Isolates the implicit-denoising effect from dimensionality reduction by
# applying the SAME 12.5 Hz zero-phase Butterworth low-pass as CR=4, but
# keeping all 6000 samples (M = N = 6000, no resampling).
#
#   Baseline      : no denoise, no dim-reduction, M=6000   (l2_baseline)
#   LP-only (this): denoise,    no dim-reduction, M=6000   (lowpass_cr1)
#   Butterworth   : denoise,    dim-reduction,    M=1500   (butterworth_cr4)
#   Wavelet CR=4  : denoise,    dim-reduction,    M=1506   (wavelet_cr4)
#
# Decision (vs. baseline P-F1=0.905, CR=4 P-F1=0.976):
#   LP-only P-F1 >= 0.95  → denoising is the dominant cause (Scenario A)
#   0.92 <= P-F1 < 0.95   → denoising + dim-reduction combined (Scenario B)
#   P-F1 < 0.92           → information-bottleneck dominates  (Scenario C)
#
# Cost note: M=6000 means per-step compute == baseline (NO speedup). Each
# seed trains at ~baseline speed (early stopping decides total). Budget for
# several hours across 3 seeds, not the optimistic ~2.5 h.
#
# Usage:
#   nohup bash scripts/run_exp1_lowpass.sh > outputs/experiment_logs/exp1_lowpass.log 2>&1 &
# ============================================================================

set -e

CONFIG="configs/stead_experiment.yaml"
SEEDS=(42 123 7)
LOG_DIR="outputs/experiment_logs"
mkdir -p "$LOG_DIR"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

# ---------------------------------------------------------------------------
# Step 1: Pre-encode LP-only set (skip if already present)
# ---------------------------------------------------------------------------
ENC_DIR="data/cs_encoded/lowpass_cr1"
if [ -f "$ENC_DIR/metadata.json" ] && [ -f "$ENC_DIR/train_waveforms.npy" ]; then
    echo "[$(timestamp)] LP-only encoding already exists at $ENC_DIR — skipping."
else
    echo "[$(timestamp)] Pre-encoding LP-only (lowpass, cr=1, M=6000)..."
    python scripts/preprocess_cs.py --config "$CONFIG" --matrix lowpass --cr 1
fi

# ---------------------------------------------------------------------------
# Step 2: Train 3 seeds (L3 path, matrix=lowpass, cr=1)
# ---------------------------------------------------------------------------
for seed in "${SEEDS[@]}"; do
    OUT="outputs/checkpoints/lowpass_cr1_seed${seed}"
    if [ -f "$OUT/best_model.pt" ]; then
        echo "[$(timestamp)] seed=$seed already trained ($OUT) — skipping."
        continue
    fi
    echo "=================================================================="
    echo "[$(timestamp)] TRAIN: lowpass cr=1 seed=$seed"
    echo "=================================================================="
    python scripts/train_cs_picker.py --config "$CONFIG" \
        --level L3 --matrix lowpass --cr 1 --seed "$seed" \
        2>&1 | tee "$LOG_DIR/lowpass_cr1_seed${seed}.log"
done

# ---------------------------------------------------------------------------
# Step 3: Evaluate each checkpoint on STEAD test set
# ---------------------------------------------------------------------------
for seed in "${SEEDS[@]}"; do
    CKPT="outputs/checkpoints/lowpass_cr1_seed${seed}/best_model.pt"
    if [ -f "$CKPT" ]; then
        echo "[$(timestamp)] EVAL: $CKPT"
        python scripts/evaluate.py --config "$CONFIG" --checkpoint "$CKPT" \
            --output "outputs/results/lowpass_cr1_seed${seed}_test.json" \
            2>&1 | tee "$LOG_DIR/lowpass_cr1_seed${seed}_eval.log"
    fi
done

echo "[$(timestamp)] LP-only control complete. Results: outputs/results/lowpass_cr1_seed*_test.json"
