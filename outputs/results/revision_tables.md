# Revision tables (draft, auto-generated)

## Table A — Architecture generalization (test set, 3 seeds)

| Model | Params (M) | Input | P-F1 | P-F1 95% CI | S-F1 | P-MAE (ms) | Wall-clock (min) | Speedup |
|---|---|---|---|---|---|---|---|---|
| CSPhaseNet | 1.45 | full-band (M=6000) | 0.9047 ± 0.0008 | [0.9030, 0.9065] | 0.8036 ± 0.0004 | 32 | 161 | 1.0× |
| CSPhaseNet | 1.45 | wavelet CR=4 (M=1506) | 0.9755 ± 0.0004 | [0.9746, 0.9764] | 0.7959 ± 0.0002 | 14 | 76 | 2.1× |
| PhaseNet | 0.27 | full-band (M=6000) | 0.9799 ± 0.0001 | [0.9790, 0.9807] | 0.8037 ± 0.0006 | 7 | 56 | 1.0× |
| PhaseNet | 0.27 | wavelet CR=4 (M=1506) | 0.9731 ± 0.0003 | [0.9722, 0.9741] | 0.7921 ± 0.0004 | 14 | 27 | 2.1× |
| EQTransformer | 0.38 | full-band (M=6000) | 0.9803 ± 0.0002 | [0.9795, 0.9812] | 0.8036 ± 0.0010 | 8 | 317 | 1.0× |
| EQTransformer | 0.38 | wavelet CR=4 (M=1506) | 0.9683 ± 0.0009 | [0.9673, 0.9693] | 0.7729 ± 0.0008 | 15 | 267 | 1.2× |

## Table B — Leakage control: trace-level vs event-level split (CSPhaseNet, 3 seeds)

| Split | Model | P-F1 | S-F1 | ΔP-F1 (pts) | P-MAE (ms) | Test traces |
|---|---|---|---|---|---|---|
| trace-level | baseline | 0.9047 ± 0.0008 | 0.8036 ± 0.0004 | +0.0 | 32 | 126567 |
| trace-level | wavelet CR=4 | 0.9755 ± 0.0004 | 0.7959 ± 0.0002 | +7.1 | 14 | 126567 |
| event-level | baseline | 0.9042 ± 0.0006 | 0.8029 ± 0.0006 | +0.0 | 32 | 126566 |
| event-level | wavelet CR=4 | 0.9740 ± 0.0002 | 0.7945 ± 0.0006 | +7.0 | 14 | 126566 |

## Table C — Waveform fidelity of band truncation (10,000 test traces)

| CR | Nominal cutoff (Hz) | −3 dB (Hz) | Power kept @8/10/15 Hz | NRMSE full | NRMSE P-win (1 s) | NRMSE S-win | Onset lag median / p95 (ms) | P bias (ms) | S bias (ms) | P-F1 | S-F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 (baseline) | 50 | — | 1/1/1 | 0 | 0 | 0 | 0 / 0 | +5.2 | +2.5 | 0.9047 | 0.8036 |
| 2 | 25 | 24.8 | 1.00/1.00/0.95 | 0.26 | 0.25 | 0.14 | 0 / 0 | +0.9 | +4.8 | 0.9595 | 0.8069 |
| 4 | 12.5 | 12.5 | 0.94/0.82/0.19 | 0.51 | 0.63 | 0.45 | 0 / 10 | -3.5 | +0.4 | 0.9755 | 0.7959 |
| 8 | 6.25 | 6.2 | 0.10/0.00/0.00 | 0.73 | 0.91 | 0.81 | 0 / 234 | +0.6 | +5.4 | 0.9360 | 0.7273 |
| 16 | 3.12 | 3.3 | 0.00/0.00/0.00 | 0.87 | 0.98 | 0.97 | 0 / 400 | -5.9 | +10.2 | 0.8161 | 0.5407 |

## Table D — Fixed-recipe label-free distillation vs. student CR (Korean borehole, held-out events, 3 splits; Table 17 fixed-recipe rows)

| Student | Distilled P-F1 | Distilled S-F1 | Distillation time (s, mean) |
|---|---|---|---|
| CR = 2 | 0.765 ± 0.017 | 0.803 ± 0.005 | 8.6 |
| CR = 4 | 0.586 ± 0.013 | 0.817 ± 0.010 | 6.5 |
| CR = 8 | 0.367 ± 0.021 | 0.697 ± 0.011 | 5.7 |
