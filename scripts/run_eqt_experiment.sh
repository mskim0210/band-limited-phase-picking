#!/usr/bin/env bash
# EQTransformer (SeisBench) full-band vs wavelet CR=4, 3 seeds, then test
# evaluation, event-cluster bootstrap and the revision summary.
# Response to C&G Reviewer 1 (architecture independence beyond PhaseNet).
#
# lr = 1e-4 (EQT_LR to override): at the shared lr = 1e-3 EQTransformer's
# training loss rises monotonically after epoch ~5 and the best checkpoint
# lands at epoch 3-8 (outputs/experiment_logs/archive_eqt_lr1e-3/), whereas at
# 1e-4 both losses decrease throughout (outputs/experiment_logs/diag_eqt_*).
# PhaseNet and CSPhaseNet are stable at 1e-3 and keep the shared recipe.
# Run alone on the machine: wall-clock per run is a reported quantity.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/experiment_logs
LR="${EQT_LR:-1e-4}"
for seed in 42 123 7; do
  python scripts/train_cs_picker.py --level L2 --seed "$seed" --arch eqtransformer --lr "$LR" \
      2>&1 | tee "outputs/experiment_logs/eqt_baseline_seed${seed}.log"
  python scripts/train_cs_picker.py --level L3 --matrix wavelet --cr 4 --seed "$seed" --arch eqtransformer --lr "$LR" \
      2>&1 | tee "outputs/experiment_logs/eqt_wavelet_cr4_seed${seed}.log"
done
for run in eqt_baseline eqt_wavelet_cr4; do for seed in 42 123 7; do
  python scripts/evaluate.py --checkpoint "outputs/checkpoints/${run}_seed${seed}/best_model.pt" \
      --output "outputs/results/${run}_seed${seed}_test.json"
done; done
python scripts/cluster_bootstrap.py --runs eqt_baseline eqt_wavelet_cr4 \
    --output outputs/results/arch_bootstrap_results_eqt.json 2>&1 | tee outputs/experiment_logs/bootstrap_eqt.log
python scripts/summarize_revision.py
echo "EQT_DONE"
