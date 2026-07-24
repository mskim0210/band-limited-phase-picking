# band-limited-phase-picking

Code and data package for the manuscript:

> **Band-Limited Compression for Training Deep Seismic Phase Pickers: I/O
> Bottleneck Diagnosis, Receptive-Field Mechanism, and Label-Free Domain
> Adaptation**
> Myungsun Kim and Jiho Park (KIGAM), submitted to *Computers & Geosciences*.

The pipeline (1) profiles and removes the storage-I/O bottleneck that
dominates deep phase-picker training on archive-scale seismic data,
(2) pre-encodes waveforms by wavelet band-level truncation and trains
directly on the compressed representation, (3) isolates *why* compression
improves accuracy (effective receptive-field enlargement) with
length-preserving low-pass, dilated-convolution, and PhaseNet controls, and
(4) recovers the compressed model on out-of-domain field data by label-free
teacher–student self-distillation.

## Quick start (no external data required)

```bash
pip install -r requirements.txt
python examples/synthetic_demo.py   # full pipeline on synthetic waveforms, ~1 min
pytest tests/                       # 78 unit tests
```

`examples/synthetic_demo.py` generates synthetic borehole-like 3-component
windows with known P/S arrivals, encodes them at CR = 4, trains a compressed
CSPhaseNet, evaluates peak-picking recall/precision, and reproduces the
label-free distillation recipe — exercising the same code paths as every
experiment in the paper.

## Repository layout

See [MANIFEST.md](MANIFEST.md) for the complete per-file inventory with roles.

```
cs_pipeline/            core library
  profiler.py             StepProfiler + GPU monitor (Fig. 3, Tables 1-3)
  sensing_matrix.py       wavelet band truncation, DCT, Butterworth, LP-only,
                          Gaussian encoders (Eq. 1-2)
  cs_encoder.py           offline pre-encoding to memmap
  cs_dataset.py           compressed-domain PyTorch dataset
  cs_aware_model.py       CSPhaseNet (U-Net, 1.45 M params; --dilation for the
                          receptive-field control)
  phasenet_wrapper.py     SeisBench PhaseNet adapter (padding/trim/upsample)
scripts/                experiment drivers (see reproduction map below)
configs/                YAML configuration (paths, splits, seeds)
tests/                  pytest unit tests
examples/               synthetic end-to-end demo
```

## Data setup

**STEAD** (Mousavi et al., 2019): download `merge.hdf5` + `merge.csv` from
https://github.com/smousavi05/STEAD, set the paths in
`configs/stead_experiment.yaml`, then build the memmap splits (deterministic
seed-42 permutation, 85/5/10):

```bash
python scripts/preprocess_stead_memmap.py --config configs/stead_experiment.yaml
python scripts/preprocess_cs.py --matrix wavelet --cr 4     # pre-encoding
```

**INSTANCE** (Michelini et al., 2021): via SeisBench;
`scripts/preprocess_instance.py`.

**SE-Korea manual pick catalog**: `instrument data_pick_time.csv` (repo root)
holds the 272-event blind analyst catalog (1,845 P / 1,842 S picks — pick times, trace-start timestamps, and instrument indices only) used in
Section 3.7. The associated event/station metadata and the borehole waveforms
are subject to KIGAM's data policy (available on request, pending institutional 
approval); the synthetic demo above provides a self-contained substitute for
testing the adaptation code path.


## Reproduction map (paper -> script)

| Paper item | Script | Output |
|---|---|---|
| Fig. 3, Tables 1-3 (I/O profiling) | `scripts/profile_baseline.py`, `scripts/measure_io_stats.py` | `outputs/profiles/` |
| Table 4 (chunked HDF5) | `scripts/exp2_hdf5_chunked.py` | `outputs/profiles/exp2_*.json` |
| Fig. 4 (workers sweep) | `scripts/exp3_num_workers_sweep.py`, `scripts/make_fig_numworkers.py` | `fig_num_workers.png` |
| Fig. 1 (sparsity) | `scripts/run_sparsity_analysis.py` | `outputs/sparsity/` |
| Tables 5-6, Figs. 5 and 8 (CR sweep, bootstrap, time-to-accuracy) | `scripts/train_cs_picker.py`, `scripts/evaluate.py`, `scripts/cluster_bootstrap.py` | `outputs/results/` |
| Table 8 (ablation incl. LP-only, dilated) | `scripts/run_exp1_lowpass.sh`; `train_cs_picker.py --dilation 4` | `outputs/results/` |
| Table 7 (SNR stratification) | `scripts/evaluate.py --snr-stratified` | `*_snr.json` |
| Table 9 (inference throughput) | timing snippet documented in `scripts/make_fig2_cr4.py` header; `outputs/results/inference_throughput.json` | |
| Table 10 (waveform-derived SNR) | `scripts/analyze_s_snr_waveform.py` | `s_snr_waveform.json` |
| Er values (Section 2.2, Table 8) | `scripts/band_energy_retention.py` | `band_energy_retention.json` |
| Fixed-recipe label-free check (Section 3.7) | `scripts/distill_fixed_recipe.py` | `manual2023_distill_fixed.json` |
| Fig. 6 (PSD), Fig. 7 (example trace) | `scripts/analyze_psd.py`, `scripts/make_waveform_figure.py` | |
| Table 11 (INSTANCE retraining) | `scripts/run_instance_experiment.sh` | `instance_*.json` |
| Table 12 (zero-shot STEAD -> INSTANCE) | `scripts/evaluate.py --dataset instance --checkpoint <STEAD ckpt> --output outputs/results/instance_<config>.json` | `instance_l2_baseline*.json`, `instance_wavelet_cr4*.json` |
| PhaseNet control (Section 3.4) | `scripts/train_cs_picker.py --arch phasenet` | `phasenet_*_test.json` |
| Tables 13-15 (SE-Korea real data) | `scripts/manual2023_catalog.py`, `scripts/apply_manual2023.py`, `scripts/adapt_manual2023.py`, `scripts/sweep_manual2023.py`, `scripts/station_holdout_manual2023.py` | `manual2023_*.json` |

Hardware used in the paper: one NVIDIA RTX 5090 (32 GB), Python 3.11,
PyTorch 2.10 + CUDA 12.8. All experiments run on any CUDA GPU; the synthetic
demo and unit tests also run on CPU.

## Result snapshot

`outputs/` ships a snapshot of the summary JSONs and epoch logs that produced
every table and figure (profiling, CR sweep, ablations, ERF, INSTANCE,
real-data adaptation), so the reported numbers can be inspected without
retraining. The epoch logs cover all training runs; wall-clock and epoch
counts quoted in the paper are three-seed means of these logs (e.g.,
baseline 161 min / 32 epochs, CR = 4 76 min / 29 epochs). Checkpoints and
encoded arrays are excluded; rerunning any script overwrites the
corresponding snapshot file.

## License

MIT — see [LICENSE](LICENSE).
