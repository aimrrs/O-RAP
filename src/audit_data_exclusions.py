from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "25544_ISS_2025.csv"
LABELS = ROOT / "data" / "processed" / "orap_multisatellite_labels_2025.csv"
OUT = ROOT / "results" / "experiments" / "audit"


def main():
    raw = pd.read_csv(RAW)
    raw["CREATION_DATE"] = pd.to_datetime(raw["CREATION_DATE"], utc=True)
    raw["EPOCH"] = pd.to_datetime(raw["EPOCH"], format="mixed", utc=True)
    keys = ["NORAD_CAT_ID", "CREATION_DATE", "EPOCH"]
    unique_records = raw.drop_duplicates(["NORAD_CAT_ID", "TLE_LINE1", "TLE_LINE2"])
    duplicate_keys = unique_records.groupby(keys).size().loc[lambda counts: counts > 1].index
    if len(duplicate_keys) != 1:
        raise RuntimeError(f"Expected one ambiguous ISS source key, found {len(duplicate_keys)}")
    norad, created, epoch = duplicate_keys[0]
    tle_rows = unique_records.loc[
        (unique_records.NORAD_CAT_ID == norad)
        & (unique_records.CREATION_DATE == created)
        & (unique_records.EPOCH == epoch),
        ["NORAD_CAT_ID", "OBJECT_NAME", "CREATION_DATE", "EPOCH", "GP_ID", "TLE_LINE1", "TLE_LINE2"],
    ].copy()
    labels = pd.read_csv(LABELS)
    labels["source_creation"] = pd.to_datetime(labels["source_creation"], format="mixed", utc=True)
    labels["source_epoch"] = pd.to_datetime(labels["source_epoch"], format="mixed", utc=True)
    excluded = labels.loc[
        (labels.norad_id == norad)
        & (labels.source_creation == created)
        & (labels.source_epoch == epoch),
    ].copy()
    OUT.mkdir(parents=True, exist_ok=True)
    tle_rows.to_csv(OUT / "ambiguous_source_tles.csv", index=False)
    excluded.to_csv(OUT / "ambiguous_excluded_labels.csv", index=False)
    summary = {
        "reason": "Different complete TLE records share the source-event key; feature matching cannot uniquely choose a source TLE, so all matching labels are excluded.",
        "source_key": {"norad_id": int(norad), "creation_utc": created.isoformat(), "epoch_utc": epoch.isoformat()},
        "ambiguous_unique_tle_records": int(len(tle_rows)),
        "excluded_label_rows": int(len(excluded)),
        "excluded_labels_by_horizon_hours": {str(int(k)): int(v) for k, v in excluded.horizon_hours.value_counts().sort_index().items()},
    }
    (OUT / "ambiguous_exclusion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
