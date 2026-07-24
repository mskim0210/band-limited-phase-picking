"""Build the manual-pick reference catalog for the SE-Korea 2023 Q4 dataset.

Source picks: 'instrument data_pick_time.csv' (InSite export). ALL rows are
analyst manual picks; the ``Comp`` column is only the analyst's bookkeeping
category (manual = borehole natural catalog, artificial = suspected blasts,
eqnet = additional auto-detected events that were then manually picked).

Maps each pick group (Comp, Event, Date, Time = trace start, OT-10 s) to its
waveform folder of the borehole export (see --waveform-dir) by trace-start time, and each
``Inst`` (1-10) to a station via receivers_insite.csv Instrument_Number.

Outputs:
    outputs/results/manual2023_catalog.csv   one row per (event, station) trace
        with p_sec / s_sec (seconds from trace start) and event metadata
    outputs/results/manual2023_stats.json    dataset statistics for the paper

Usage:
    python scripts/manual2023_catalog.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

PICKS = Path("instrument data_pick_time.csv")
BASE = Path("data/2023Q4_p8")  # borehole waveform export (from KIGAM, on request)
OUT_CSV = Path("outputs/results/manual2023_catalog.csv")
OUT_STATS = Path("outputs/results/manual2023_stats.json")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--waveform-dir", type=Path, default=BASE,
                    help="Directory of the borehole waveform export")
    args = ap.parse_args()
    base = args.waveform_dir

    picks = pd.read_csv(PICKS)
    recv = pd.read_csv(base / "receivers_insite.csv")
    events = pd.read_csv(base / "event_list_categorized.csv", parse_dates=["ot"])

    inst2sta = (recv.drop_duplicates("Instrument_Number")
                .set_index("Instrument_Number")["Instrument_Label"].to_dict())
    logger.info("Instrument map: %s", inst2sta)

    # trace start = OT - 10 s; pick CSV Date/Time is the start truncated to 1 s
    events["start"] = events.ot - pd.Timedelta(seconds=10)
    events["start_sec"] = events.start.dt.floor("s")
    ev_by_start = events.set_index("start_sec")

    picks["start"] = pd.to_datetime(picks.Date + " " + picks.Time, dayfirst=True)

    rows, unmatched = [], []
    for (comp, evno, start), g in picks.groupby(["Comp", "Event", "start"]):
        hit = None
        for dt in (0, 1, -1):  # allow 1-s truncation slack
            key = start + pd.Timedelta(seconds=dt)
            if key in ev_by_start.index:
                hit = ev_by_start.loc[key]
                if isinstance(hit, pd.DataFrame):
                    hit = hit.iloc[0]
                break
        if hit is None:
            unmatched.append((comp, evno, str(start)))
            continue
        frac = (hit.start - hit.start.floor("s")).total_seconds()
        for _, r in g.iterrows():
            if pd.isna(r.Ptimepick) and pd.isna(r.Stimepick):
                continue
            sta = inst2sta[int(r.Inst)]
            rows.append({
                "comp": comp, "event_no": int(evno),
                "folder": hit.folder, "path": str(base / hit.path),
                "category": hit.category, "ot": str(hit.ot),
                "event_id": f"{comp}_{int(evno)}",
                "station": sta,
                # pick seconds are relative to the true (fractional) trace
                # start, which equals the mseed start time -> use directly
                "p_sec": r.Ptimepick if pd.notna(r.Ptimepick) else np.nan,
                "s_sec": r.Stimepick if pd.notna(r.Stimepick) else np.nan,
                "lat": hit.lat, "lon": hit.lon, "depth_km": hit.depth_km,
                "start_frac_offset": round(frac, 3),
            })

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    n_ev = df.event_id.nunique()
    stats = {
        "n_events": n_ev,
        "n_events_by_comp": df.groupby("comp").event_id.nunique().to_dict(),
        "n_traces": len(df),
        "n_p_picks": int(df.p_sec.notna().sum()),
        "n_s_picks": int(df.s_sec.notna().sum()),
        "n_stations": int(df.station.nunique()),
        "stations": sorted(df.station.unique().tolist()),
        "period": [str(df.ot.min()), str(df.ot.max())],
        "depth_km_range": [round(float(df.depth_km.min()), 1),
                           round(float(df.depth_km.max()), 1)],
        "unmatched_pick_groups": unmatched,
    }
    json.dump(stats, open(OUT_STATS, "w"), indent=2, ensure_ascii=False)

    logger.info("Catalog: %d traces, %d events (%s), P %d / S %d",
                len(df), n_ev, stats["n_events_by_comp"],
                stats["n_p_picks"], stats["n_s_picks"])
    if unmatched:
        logger.warning("UNMATCHED pick groups: %d -> %s",
                       len(unmatched), unmatched[:10])
    logger.info("Stations: %s", stats["stations"])


if __name__ == "__main__":
    main()
