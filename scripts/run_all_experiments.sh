#!/bin/bash
# ============================================================================
# Full training: 21 runs for the CR sweep (Table 5) + encoder ablation (Table 8)
#
# Schedule:
#   Phase 1 (Night 1-3): L2 baseline × 3 seeds      (~54h)
#   Phase 2 (Day 4):     CR=2,4 wavelet × 3 seeds    (~12h)
#   Phase 3 (Day 5):     CR=8,16 + gaussian,dct × 3   (~11h)
#
# Usage:
#   # Run all 21 experiments sequentially
#   bash scripts/run_all_experiments.sh
#
#   # Run a specific phase only
#   bash scripts/run_all_experiments.sh phase1   # L2 baseline only
#   bash scripts/run_all_experiments.sh phase2   # CR=2,4 wavelet
#   bash scripts/run_all_experiments.sh phase3   # CR=8,16 + matrix ablation
#
#   # Run in background with nohup
#   nohup bash scripts/run_all_experiments.sh > experiments.log 2>&1 &
# ============================================================================

set -e  # Exit on error

CONFIG="configs/stead_experiment.yaml"
TRAIN="python scripts/train_cs_picker.py --config $CONFIG"
ENCODE="python scripts/preprocess_cs.py --config $CONFIG"
SEEDS=(42 123 7)

LOG_DIR="outputs/experiment_logs"
mkdir -p "$LOG_DIR"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

run_experiment() {
    local desc="$1"
    shift
    echo ""
    echo "=================================================================="
    echo "[$(timestamp)] START: $desc"
    echo "  Command: $TRAIN $@"
    echo "=================================================================="
    $TRAIN "$@" 2>&1 | tee "$LOG_DIR/$(echo $desc | tr ' /' '_').log"
    echo "[$(timestamp)] DONE: $desc"
    echo ""
}

# ── Pre-encoding: ensure all required CS datasets exist ──────────────

ensure_encoded() {
    local matrix=$1
    local cr=$2
    local dir="data/cs_encoded/${matrix}_cr${cr}"
    if [ ! -d "$dir" ] || [ ! -f "$dir/train_waveforms.npy" ]; then
        echo "[$(timestamp)] Encoding: $matrix CR=$cr ..."
        $ENCODE --cr $cr --matrix $matrix --num-workers 8
    else
        echo "[$(timestamp)] Already encoded: $dir"
    fi
}

# ============================================================================
# Phase 1: L2 Baseline (no CS) — longest runs, schedule for nights
# ============================================================================
phase1() {
    echo ""
    echo "########################################"
    echo "# Phase 1: L2 Baseline × 3 seeds"
    echo "# Estimated: ~54 hours"
    echo "########################################"

    for seed in "${SEEDS[@]}"; do
        run_experiment "L2_baseline_seed${seed}" \
            --level L2 --seed $seed
    done
}

# ============================================================================
# Phase 2: CR sweep (wavelet) — CR=2,4
# ============================================================================
phase2() {
    echo ""
    echo "########################################"
    echo "# Phase 2: Wavelet CR=2,4 × 3 seeds"
    echo "# Estimated: ~12 hours"
    echo "########################################"

    # Ensure wavelet pre-encoding exists
    ensure_encoded wavelet 2
    ensure_encoded wavelet 4

    for cr in 2 4; do
        for seed in "${SEEDS[@]}"; do
            run_experiment "wavelet_cr${cr}_seed${seed}" \
                --level L3 --matrix wavelet --cr $cr --seed $seed
        done
    done
}

# ============================================================================
# Phase 3: CR=8,16 wavelet + matrix ablation (gaussian,dct at CR=4)
# ============================================================================
phase3() {
    echo ""
    echo "########################################"
    echo "# Phase 3: CR=8,16 + Matrix Ablation"
    echo "# Estimated: ~11 hours"
    echo "########################################"

    # Ensure wavelet CR=8,16 pre-encoding
    ensure_encoded wavelet 8
    ensure_encoded wavelet 16

    # CR=8 wavelet
    for seed in "${SEEDS[@]}"; do
        run_experiment "wavelet_cr8_seed${seed}" \
            --level L3 --matrix wavelet --cr 8 --seed $seed
    done

    # CR=16 wavelet
    for seed in "${SEEDS[@]}"; do
        run_experiment "wavelet_cr16_seed${seed}" \
            --level L3 --matrix wavelet --cr 16 --seed $seed
    done

    # Gaussian CR=4 (encoder ablation)
    ensure_encoded gaussian 4
    for seed in "${SEEDS[@]}"; do
        run_experiment "gaussian_cr4_seed${seed}" \
            --level L3 --matrix gaussian --cr 4 --seed $seed
    done

    # DCT CR=4 (encoder ablation)
    ensure_encoded dct 4
    for seed in "${SEEDS[@]}"; do
        run_experiment "dct_cr4_seed${seed}" \
            --level L3 --matrix dct --cr 4 --seed $seed
    done
}

# ============================================================================
# Main dispatcher
# ============================================================================

echo "============================================"
echo " P2 Full Training Experiment Runner"
echo " Total: 21 runs (~77 GPU-hours)"
echo " Start: $(timestamp)"
echo "============================================"

case "${1:-all}" in
    phase1) phase1 ;;
    phase2) phase2 ;;
    phase3) phase3 ;;
    all)
        phase1
        phase2
        phase3
        ;;
    *)
        echo "Usage: $0 [phase1|phase2|phase3|all]"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo " All experiments complete!"
echo " End: $(timestamp)"
echo " Results: outputs/checkpoints/"
echo " Logs:    $LOG_DIR/"
echo "============================================"
