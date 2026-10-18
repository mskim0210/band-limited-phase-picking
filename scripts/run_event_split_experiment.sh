#!/usr/bin/env bash
# Event-level (source_id-grouped) split control: baseline vs wavelet CR=4, 3 seeds.
# Response to C&G Reviewer 1 (possible event leakage under the trace-level split).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/experiment_logs
for seed in 42 123 7; do
  python scripts/train_cs_picker.py --level L2 --seed "$seed" --split event \
      2>&1 | tee "outputs/experiment_logs/l2_baseline_evsplit_seed${seed}.log"
  python scripts/train_cs_picker.py --level L3 --matrix wavelet --cr 4 --seed "$seed" --split event \
      2>&1 | tee "outputs/experiment_logs/wavelet_cr4_evsplit_seed${seed}.log"
done
for run in l2_baseline_evsplit wavelet_cr4_evsplit; do for seed in 42 123 7; do
  python scripts/evaluate.py --checkpoint "outputs/checkpoints/${run}_seed${seed}/best_model.pt" \
      --split event --output "outputs/results/${run}_seed${seed}_test.json"
done; done
echo "EVSPLIT_DONE"
