#!/bin/bash
# ============================================================================
# Remaining experiments: 18 runs (L2 baseline already done)
#
# CR sweep (wavelet, Table 5) — 12 runs
# Encoder ablation at CR=4 (Table 8) — 6 runs
#
# Usage:
#   nohup bash scripts/run_remaining_experiments.sh > outputs/experiment_logs/run_remaining.log 2>&1 &
# ============================================================================

set -e

CONFIG="configs/stead_experiment.yaml"
TRAIN="python scripts/train_cs_picker.py --config $CONFIG"
SEEDS=(42 123 7)

LOG_DIR="outputs/experiment_logs"
mkdir -p "$LOG_DIR"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

run_experiment() {
    local desc="$1"
    shift
    local log_file="$LOG_DIR/$(echo $desc | tr ' /' '_').log"

    echo ""
    echo "=================================================================="
    echo "[$(timestamp)] START: $desc"
    echo "  Command: $TRAIN $@"
    echo "=================================================================="

    $TRAIN "$@" 2>&1 | tee "$log_file"

    echo "[$(timestamp)] DONE: $desc"
    echo ""
}

echo "============================================"
echo " Remaining experiments (18 runs)"
echo " Start: $(timestamp)"
echo "============================================"

# ── CR sweep (wavelet, 12 runs, Table 5) ──────────────────────────────

echo ""
echo "########################################"
echo "# CR sweep: wavelet × 3 seeds"
echo "########################################"

for cr in 2 4 8 16; do
    for seed in "${SEEDS[@]}"; do
        run_experiment "wavelet_cr${cr}_seed${seed}" \
            --level L3 --matrix wavelet --cr $cr --seed $seed
    done
done

# ── Encoder ablation at CR=4 (6 runs, Table 8) ───────────────────────

echo ""
echo "########################################"
echo "# Encoder ablation: Gaussian + DCT at CR=4"
echo "########################################"

for matrix in gaussian dct; do
    for seed in "${SEEDS[@]}"; do
        run_experiment "${matrix}_cr4_seed${seed}" \
            --level L3 --matrix $matrix --cr 4 --seed $seed
    done
done

echo ""
echo "============================================"
echo " All 18 experiments complete!"
echo " End: $(timestamp)"
echo " Results: outputs/checkpoints/"
echo "============================================"
