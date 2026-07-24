# Package Manifest

Complete inventory of this repository and the role of every file.
Paper references (tables/figures/sections) follow the *Computers &
Geosciences* manuscript. See `README.md` for installation, data setup, and
the reproduction quick-start.

## Top level

| File | Role |
|---|---|
| `README.md` | Overview, installation, data setup, paper-to-script reproduction map |
| `MANIFEST.md` | This file — full inventory with per-file roles |
| `LICENSE` | MIT license |
| `requirements.txt` | Python dependencies (Python 3.11, PyTorch 2.x) |
| `.gitignore` | Excludes generated data/outputs and caches |
| `instrument data_pick_time.csv` | Blind manual pick catalog for the SE-Korea deep-borehole network (272 events; 1,845 P / 1,842 S analyst picks). Pick times, trace-start timestamps, and instrument indices only. Reference data for Section 3.7 (Tables 13–15) |

## `configs/` — experiment configuration

| File | Role |
|---|---|
| `stead_experiment.yaml` | Main configuration: STEAD paths, 85/5/10 split (seed 42), training hyperparameters, CR grid, profiling settings |
| `instance_experiment.yaml` | Configuration for the INSTANCE generalization experiment (Section 3.6, Tables 11–12) |

## `cs_pipeline/` — core library

| File | Role |
|---|---|
| `sensing_matrix.py` | All encoders (Eqs. 1–2): wavelet band-level truncation (main method), DCT, Butterworth resampling, length-preserving low-pass control (LP-only), Gaussian random projection; factory `create_sensing_matrix()` |
| `cs_encoder.py` | Offline pre-encoding of memmap splits into compressed memmap datasets |
| `cs_dataset.py` | PyTorch dataset for compressed-domain training (per-trace normalization, Gaussian labels) |
| `cs_aware_model.py` | CSPhaseNet U-Net (1.45 M params; compressed input → full-length output). `dilation` argument implements the receptive-field control of Section 3.4 |
| `phasenet_wrapper.py` | SeisBench PhaseNet adapter (pad/trim/upsample) for the modern-picker control experiment (Section 3.4) |
| `profiler.py` | StepProfiler + GPU monitor used for all per-step timing (Fig. 3, Tables 1–3) |
| `sparsity_analysis.py` | Wavelet/DCT coefficient-decay and energy-retention analysis (Fig. 1) |
| `metrics.py` | Peak-picking precision/recall/F1 and residual metrics |
| `__init__.py` | Package marker |

## `scripts/` — experiment drivers

### Data preparation

| File | Role |
|---|---|
| `preprocess_stead_memmap.py` | STEAD `merge.hdf5` → per-split float32 memmap (L0 → L2); deterministic seed-42 split |
| `preprocess_cs.py` | Memmap splits → band-limited pre-encoded datasets (L3) for each matrix/CR |
| `preprocess_instance.py` | INSTANCE subset download/conversion for Section 3.6 |

### I/O profiling (Section 3.2; Fig. 3–4, Tables 1–4)

| File | Role |
|---|---|
| `profile_baseline.py` | L0/L2/L3 per-step decomposition; also defines `STEADMemmapDataset` |
| `measure_io_stats.py` | System-call-level I/O statistics (Table 2) |
| `exp2_hdf5_chunked.py` | Chunked-HDF5 + cache vs. memmap random-read latency (Table 4) |
| `exp3_num_workers_sweep.py` | DataLoader-worker sweep, L0 vs. L2 (Fig. 4 data) |

### Training and evaluation (Sections 3.3–3.5; Tables 5–10, Figs. 5, 7–8)

| File | Role |
|---|---|
| `train_cs_picker.py` | Main training loop (baseline / compressed / LP-only; `--dilation` for the receptive-field control; `--arch phasenet` for the SeisBench PhaseNet control) |
| `evaluate.py` | Test-set evaluation (tolerances 0.1/0.5 s, residuals, `--snr-stratified`) |
| `cluster_bootstrap.py` | Trace-level vs. event-cluster bootstrap CIs (Table 6) |
| `analyze_psd.py` | STEAD P/S-onset and noise PSD (Fig. 6) |
| `analyze_s_snr_waveform.py` | Waveform-derived SNR stratification of S-F1 for all four Section-3.4 models (Table 10) |
| `band_energy_retention.py` | Retained-band energy of the encoders (Eq. 2 values in Section 2.2 and Table 8) |
| `measure_erf.py` | Gradient-based effective-receptive-field measurement (5.3 s baseline vs. 19.1/18.0 s; Section 3.4) |
| `run_sparsity_analysis.py` | Driver for `sparsity_analysis.py` (Fig. 1) |
| `run_all_experiments.sh`, `run_remaining_experiments.sh` | Batch drivers for the CR-sweep/ablation training runs |
| `run_exp1_lowpass.sh` | LP-only control training (Table 8) |
| `run_instance_experiment.sh` | INSTANCE retraining experiment (Tables 11–12) |

### Real-data validation and adaptation (Section 3.7; Tables 13–15)

| File | Role |
|---|---|
| `manual2023_catalog.py` | InSite manual picks → per-(event, station) reference catalog + dataset statistics |
| `apply_manual2023.py` | Zero-shot evaluation of baseline/CR=2/4/8 against the manual catalog (P/S recall, precision, residuals, false triggers/hour); builds the shared window cache |
| `adapt_manual2023.py` | Cluster-aware event-split fine-tuning and label-free distillation with threshold/tolerance sweeps (Table 15) |
| `sweep_manual2023.py` | Symmetric hyperparameter sweep for fine-tune vs. distillation (inner-validation selection) + adaptation-cost timing |
| `station_holdout_manual2023.py` | Five-fold borehole-site hold-out for cross-station generalization of distillation |
| `distill_fixed_recipe.py` | Fully label-free distillation with the fixed recipe (lr 1e-4, T = 1, 30 epochs; no labeled validation) |

### Figure generation

| File | Role |
|---|---|
| `generate_figures.py` | CR-vs-F1 and time-to-accuracy figures (Figs. 5, 8) |
| `make_fig2_cr4.py` | Step-time decomposition figure with corrected CR=4 bar (Fig. 3) |
| `make_fig_numworkers.py` | Worker-sweep figure (Fig. 4) |
| `make_waveform_figure.py` | Representative-trace mechanism figure (Fig. 7) |

Every data-derived figure in the paper (Figs. 1, 3–8) is regenerated by the
scripts above. Fig. 2 is a hand-drawn architecture schematic (no underlying
data) and therefore has no generation script.

## `examples/` — self-contained test case

| File | Role |
|---|---|
| `synthetic_demo.py` | Runs the complete pipeline (synthetic waveforms → CR=4 encoding → training → evaluation → label-free distillation) with no external data, in about a minute. The reproducible test case required by the journal's code policy, given that the borehole waveforms are available on request only |

## `tests/` — unit tests (78 tests, `pytest tests/`)

| File | Role |
|---|---|
| `test_sensing_matrix.py` | Encoder correctness: band truncation, energy retention, shapes, LP-only control |
| `test_cs_encoder.py` | Offline pre-encoding round-trip |
| `test_cs_dataset.py` | Dataset shapes, normalization, label generation |
| `test_profiler.py` | StepProfiler timing accounting |
| `test_sparsity_analysis.py` | Sparsity/energy computations |

## Not included (by design)

| Item | Reason |
|---|---|
| STEAD / INSTANCE archives | Public benchmarks; download and preprocessing instructions in `README.md` |
| SE-Korea borehole waveforms and event/station metadata | Subject to KIGAM's data policy; request-based access pending institutional approval. `examples/synthetic_demo.py` substitutes for testing |
| `outputs/` checkpoints and large arrays | Reproducible from the scripts above; a snapshot of the summary result JSONs and the epoch logs behind every table and figure IS included under `outputs/` so reported numbers can be checked without retraining |

