#!/usr/bin/env bash
# EQTransformer (SeisBench) full-band vs wavelet CR=4, 3 seeds.
# Response to C&G Reviewer 1 (architecture independence beyond PhaseNet).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/experiment_logs
for seed in 42 123 7; do
  python scripts/train_cs_picker.py --level L2 --seed "$seed" --arch eqtransformer \
      2>&1 | tee "outputs/experiment_logs/eqt_baseline_seed${seed}.log"
  python scripts/train_cs_picker.py --level L3 --matrix wavelet --cr 4 --seed "$seed" --arch eqtransformer \
      2>&1 | tee "outputs/experiment_logs/eqt_wavelet_cr4_seed${seed}.log"
done
for run in eqt_baseline eqt_wavelet_cr4; do for seed in 42 123 7; do
  python scripts/evaluate.py --checkpoint "outputs/checkpoints/${run}_seed${seed}/best_model.pt" \
      --output "outputs/results/${run}_seed${seed}_test.json"
done; done
echo "EQT_DONE"
