"""Aggregate the C&G revision experiments into one JSON + Markdown table draft.

Tables produced (mean ± std over seeds, test set, tolerance 0.1 s):
  A  Architecture generalization: CSPhaseNet / PhaseNet / EQTransformer
     × {full-band, wavelet CR=4}, with bootstrap 95% CIs when available.
  B  Leakage controls: trace-level vs event-level split × {baseline, CR=4}.
  C  Waveform fidelity per CR (from waveform_distortion.json).
  D  Distillation ceiling per CR (from manual2023_decomposition.json).

Usage:
    python scripts/summarize_revision.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

SEEDS = (42, 123, 7)


def fmt_pm(mean: float, std: float, digits: int = 4) -> str:
    """'mean ± std' with fixed decimals."""
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def aggregate_runs(name: str, seeds: tuple[int, ...] = SEEDS,
                   root: str | Path = "outputs") -> dict | None:
    """Mean/std of test metrics, wall-clock and epochs over the seeds of one run.

    Reads ``{root}/results/{name}_seed{s}_test.json`` and
    ``{root}/checkpoints/{name}_seed{s}/epoch_log.csv``.  Returns None when no
    seed has a result file yet.
    """
    root = Path(root)
    rows = []
    for s in seeds:
        rp = root / "results" / f"{name}_seed{s}_test.json"
        lp = root / "checkpoints" / f"{name}_seed{s}" / "epoch_log.csv"
        if not rp.exists():
            continue
        d = json.load(open(rp))
        t = d["tolerance_0.1s"]
        tm = d.get("timing", {})
        row = {"seed": s, "p_f1": t["P"]["f1"], "s_f1": t["S"]["f1"],
               "p_prec": t["P"]["precision"], "p_rec": t["P"]["recall"],
               "p_mae_ms": 1000.0 * (tm.get("P", {}).get("mae_sec") or np.nan),
               "p_bias_ms": 1000.0 * (tm.get("P", {}).get("mean_residual_sec") or np.nan),
               "s_mae_ms": 1000.0 * (tm.get("S", {}).get("mae_sec") or np.nan)}
        if lp.exists():
            log = np.genfromtxt(lp, delimiter=",", names=True)
            log = np.atleast_1d(log)
            row["wall_min"] = float(log["wall_clock_sec"][-1]) / 60.0
            row["epochs"] = int(log["epoch"][-1])
        rows.append(row)
    if not rows:
        return None
    out: dict = {"name": name, "n_seeds": len(rows), "seeds": [r["seed"] for r in rows]}
    for k in ("p_f1", "s_f1", "p_prec", "p_rec", "p_mae_ms", "p_bias_ms", "s_mae_ms",
              "wall_min", "epochs"):
        v = np.array([r[k] for r in rows if k in r], dtype=float)
        if v.size:
            out[f"{k}_mean"] = float(np.nanmean(v))
            out[f"{k}_std"] = float(np.nanstd(v))
    return out


def _ci(boot: dict | None, run: str, phase: str = "P") -> str:
    """Pooled 95% CI text over seeds from a cluster_bootstrap JSON (trace-level CI)."""
    if not boot or run not in boot:
        return "—"
    lows = [v[phase]["trace_CI"]["ci_low"] for v in boot[run].values()]
    highs = [v[phase]["trace_CI"]["ci_high"] for v in boot[run].values()]
    return f"[{np.mean(lows):.4f}, {np.mean(highs):.4f}]"


def _load(path: str) -> dict | None:
    p = Path(path)
    return json.load(open(p)) if p.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs")
    ap.add_argument("--json-out", default="outputs/results/revision_summary.json")
    ap.add_argument("--md-out", default="manuscript/revision_tables_v8.md")
    args = ap.parse_args()
    root = Path(args.root)

    arch_runs = [("CSPhaseNet", "l2_baseline", "wavelet_cr4", 1.45),
                 ("PhaseNet", "phasenet_baseline", "phasenet_wavelet_cr4", 0.27),
                 ("EQTransformer", "eqt_baseline", "eqt_wavelet_cr4", 0.38)]
    boot = {}
    for f in ("cluster_bootstrap_results.json", "arch_bootstrap_results_phasenet.json",
              "arch_bootstrap_results_eqt.json"):
        boot.update(_load(root / "results" / f) or {})

    summary: dict = {"table_A": [], "table_B": [], "table_C": None, "table_D": None}
    md = ["# Revision tables (draft, auto-generated)\n"]

    # ---- Table A ----
    md.append("## Table A — Architecture generalization (test set, 3 seeds)\n")
    md.append("| Model | Params (M) | Input | P-F1 | P-F1 95% CI | S-F1 | P-MAE (ms) | Wall-clock (min) | Speedup |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for label, full, cr4, params in arch_runs:
        a_full, a_cr4 = aggregate_runs(full, root=root), aggregate_runs(cr4, root=root)
        for inp, a, run in (("full-band (M=6000)", a_full, full), ("wavelet CR=4 (M=1506)", a_cr4, cr4)):
            if a is None:
                md.append(f"| {label} | {params} | {inp} | pending | | | | | |")
                continue
            speed = (a_full["wall_min_mean"] / a["wall_min_mean"]) if (a_full and "wall_min_mean" in a) else np.nan
            summary["table_A"].append({"model": label, "run": run, "input": inp, **a,
                                       "ci_p": _ci(boot, run), "speedup": speed})
            md.append(f"| {label} | {params} | {inp} | {fmt_pm(a['p_f1_mean'], a['p_f1_std'])} | "
                      f"{_ci(boot, run)} | {fmt_pm(a['s_f1_mean'], a['s_f1_std'])} | "
                      f"{a.get('p_mae_ms_mean', np.nan):.0f} | {a.get('wall_min_mean', np.nan):.0f} | "
                      f"{speed:.1f}× |")
    md.append("")

    # ---- Table B ----
    md.append("## Table B — Leakage control: trace-level vs event-level split (CSPhaseNet, 3 seeds)\n")
    md.append("| Split | Model | P-F1 | S-F1 | ΔP-F1 (pts) | P-MAE (ms) | Test traces |")
    md.append("|---|---|---|---|---|---|---|")
    for split_label, base, cr4, ntest in (("trace-level", "l2_baseline", "wavelet_cr4", 126567),
                                           ("event-level", "l2_baseline_evsplit", "wavelet_cr4_evsplit", 126566)):
        ab, ac = aggregate_runs(base, root=root), aggregate_runs(cr4, root=root)
        for lab, a in (("baseline", ab), ("wavelet CR=4", ac)):
            if a is None:
                md.append(f"| {split_label} | {lab} | pending | | | | {ntest} |")
                continue
            delta = 100 * (a["p_f1_mean"] - ab["p_f1_mean"]) if (ab and lab != "baseline") else 0.0
            summary["table_B"].append({"split": split_label, "model": lab, "run": a["name"], **a,
                                       "delta_p_pts": delta})
            md.append(f"| {split_label} | {lab} | {fmt_pm(a['p_f1_mean'], a['p_f1_std'])} | "
                      f"{fmt_pm(a['s_f1_mean'], a['s_f1_std'])} | {delta:+.1f} | "
                      f"{a.get('p_mae_ms_mean', np.nan):.0f} | {ntest} |")
    md.append("")

    # ---- Table C ----
    wd = _load(root / "results" / "waveform_distortion.json")
    if wd:
        summary["table_C"] = wd
        md.append("## Table C — Waveform fidelity of band truncation (10,000 test traces)\n")
        md.append("| CR | Nominal cutoff (Hz) | −3 dB (Hz) | Power kept @8/10/15 Hz | NRMSE full | NRMSE P-win (1 s) | NRMSE S-win | Onset lag median / p95 (ms) | P bias (ms) | S bias (ms) | P-F1 | S-F1 |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        b = wd["baseline"]
        md.append(f"| 1 (baseline) | 50 | — | 1/1/1 | 0 | 0 | 0 | 0 / 0 | {b['p_bias_ms']:+.1f} | {b['s_bias_ms']:+.1f} | {b['p_f1']:.4f} | {b['s_f1']:.4f} |")
        for cr, r in wd["per_cr"].items():
            tf = wd["noise_transfer_function"][cr]
            pr = tf["power_ratio_at"]
            md.append(f"| {cr} | {tf['nominal_cutoff_hz']:.3g} | {tf['f_minus3dB_hz']:.1f} | "
                      f"{pr['8']:.2f}/{pr['10']:.2f}/{pr['15']:.2f} | {r['nrmse_full_mean']:.2f} | "
                      f"{r['nrmse_p_win_median']:.2f} | {r['nrmse_s_win_median']:.2f} | "
                      f"{r['onset_lag_ms_median']:.0f} / {r['onset_lag_ms_p95_abs']:.0f} | "
                      f"{r['p_bias_ms']:+.1f} | {r['s_bias_ms']:+.1f} | {r['p_f1']:.4f} | {r['s_f1']:.4f} |")
        md.append("")

    # ---- Table D ----
    dc = _load(root / "results" / "manual2023_decomposition.json")
    if dc:
        summary["table_D"] = dc
        md.append("## Table D — Distillation ceiling vs CR (Korean borehole, held-out events, 3 splits)\n")
        md.append("| Student | Zero-shot P-F1 | Distilled P-F1 | Distilled S-F1 | Representation recovery (pts) | Residual vs baseline-ft 0.82 (pts) |")
        md.append("|---|---|---|---|---|---|")
        for cr, r in dc["per_cr"].items():
            md.append(f"| CR = {cr} | {r['zero_shot_p_f1']:.2f} | {fmt_pm(*r['distill_p_f1'], digits=3)} | "
                      f"{fmt_pm(*r['distill_s_f1'], digits=3)} | {100 * r['representation_recovery_p']:+.0f} | "
                      f"{100 * r['spectral_residual_p_vs_baseline_ft']:.0f} |")
        md.append("")

    Path(args.json_out).write_text(json.dumps(summary, indent=1, default=float))
    Path(args.md_out).write_text("\n".join(md))
    logger.info("saved %s and %s", args.json_out, args.md_out)
    print("\n".join(md))


if __name__ == "__main__":
    main()
