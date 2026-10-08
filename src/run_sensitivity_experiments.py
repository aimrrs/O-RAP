from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
LABELS = ROOT / "data" / "processed" / "orap_multisatellite_labels_2025.csv"
OUT = ROOT / "results" / "experiments" / "sensitivity"
MODEL = ROOT / "src" / "run_final_tuned_xgboost.py"
SAT_FILES = sorted(RAW.glob("*_2025.csv"))
FEATURE_COLUMNS = [
    "norad_id", "satellite_name", "horizon_hours", "source_tle_age_hours",
    "mean_motion", "eccentricity", "inclination", "ra_of_asc_node",
    "arg_of_pericenter", "mean_anomaly", "bstar", "mean_motion_dot",
    "mean_motion_ddot", "semimajor_axis", "period", "apoapsis", "periapsis",
    "unreliable_1km",
]
RAW_COLUMNS = [
    "NORAD_CAT_ID", "CREATION_DATE", "EPOCH", "TLE_LINE1", "TLE_LINE2",
    "MEAN_MOTION", "ECCENTRICITY", "INCLINATION", "RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER", "MEAN_ANOMALY", "BSTAR", "MEAN_MOTION_DOT",
    "MEAN_MOTION_DDOT", "SEMIMAJOR_AXIS", "PERIOD", "APOAPSIS", "PERIAPSIS",
]


def load_data():
    labels = pd.read_csv(LABELS)
    for col in ("source_creation", "source_epoch"):
        labels[col] = pd.to_datetime(labels[col], format="mixed", utc=True)
    frames = []
    for path in SAT_FILES:
        raw = pd.read_csv(path, usecols=RAW_COLUMNS)
        raw["CREATION_DATE"] = pd.to_datetime(raw["CREATION_DATE"], format="mixed", utc=True)
        raw["EPOCH"] = pd.to_datetime(raw["EPOCH"], format="mixed", utc=True)
        frames.append(raw)
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.drop_duplicates(["NORAD_CAT_ID", "TLE_LINE1", "TLE_LINE2"], keep="first")
    keys = ["NORAD_CAT_ID", "CREATION_DATE", "EPOCH"]
    ambiguous = raw.groupby(keys).size().loc[lambda s: s > 1].index
    if len(ambiguous):
        amb = pd.MultiIndex.from_tuples(ambiguous, names=keys)
        raw_index = pd.MultiIndex.from_frame(raw[keys])
        raw = raw.loc[~raw_index.isin(amb)].copy()
        label_index = pd.MultiIndex.from_arrays(
            [labels["norad_id"], labels["source_creation"], labels["source_epoch"]],
            names=["NORAD_CAT_ID", "CREATION_DATE", "EPOCH"],
        )
        labels = labels.loc[~label_index.isin(amb)].copy()
    labels["norad_id"] = labels["norad_id"].astype(int)
    merged = labels.merge(
        raw,
        left_on=["norad_id", "source_creation", "source_epoch"],
        right_on=["NORAD_CAT_ID", "CREATION_DATE", "EPOCH"],
        how="left",
        validate="many_to_one",
    )
    if merged["TLE_LINE1"].isna().any():
        raise RuntimeError("Some label rows do not map to a unique source TLE.")
    data = pd.DataFrame({
        "norad_id": merged["norad_id"],
        "satellite_name": merged["satellite_name"],
        "horizon_hours": merged["horizon_hours"],
        "source_tle_age_hours": merged["source_tle_age_hours"],
        "mean_motion": merged["MEAN_MOTION"],
        "eccentricity": merged["ECCENTRICITY"],
        "inclination": merged["INCLINATION"],
        "ra_of_asc_node": merged["RA_OF_ASC_NODE"],
        "arg_of_pericenter": merged["ARG_OF_PERICENTER"],
        "mean_anomaly": merged["MEAN_ANOMALY"],
        "bstar": merged["BSTAR"],
        "mean_motion_dot": merged["MEAN_MOTION_DOT"],
        "mean_motion_ddot": merged["MEAN_MOTION_DDOT"],
        "semimajor_axis": merged["SEMIMAJOR_AXIS"],
        "period": merged["PERIOD"],
        "apoapsis": merged["APOAPSIS"],
        "periapsis": merged["PERIAPSIS"],
        "proxy_error_km": merged["proxy_error_km"],
        "reference_delay_hours": merged["reference_delay_hours"],
        "source_creation": merged["source_creation"],
    })
    return data


def save_splits(data: pd.DataFrame, folder: Path, target_km: float):
    data = data.copy()
    data["unreliable_1km"] = (data["proxy_error_km"] > target_km).astype(int)
    month = data["source_creation"].dt.month
    masks = {
        "train_temporal": month.between(1, 8),
        "validation_temporal": month.between(9, 10),
        "test_temporal": month.between(11, 12),
    }
    folder.mkdir(parents=True, exist_ok=True)
    rows = {}
    for name, mask in masks.items():
        frame = data.loc[mask, FEATURE_COLUMNS].copy()
        frame.to_csv(folder / f"{name}.csv", index=False)
        rows[name] = len(frame)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    cases = [("error_threshold_0p5km", 0.5, 6.0),
             ("error_threshold_2km", 2.0, 6.0),
             ("error_threshold_5km", 5.0, 6.0),
             ("reference_delay_1h", 1.0, 1.0),
             ("reference_delay_2h", 1.0, 2.0),
             ("reference_delay_4h", 1.0, 4.0)]
    all_results, all_configs, manifests = [], [], []
    for name, target_km, delay_h in cases:
        case_data = data.loc[data["reference_delay_hours"] <= delay_h].copy()
        case_dir = OUT / name
        input_dir = case_dir / "inputs"
        result_dir = case_dir / "tables"
        result_dir.mkdir(parents=True, exist_ok=True)
        counts = save_splits(case_data, input_dir, target_km)
        env = os.environ.copy()
        env["ORAP_DATA_DIR"] = str(input_dir)
        env["ORAP_RESULTS_DIR"] = str(result_dir)
        env["PYTHONIOENCODING"] = "utf-8"
        log_path = case_dir / "run.log"
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run([sys.executable, str(MODEL)], cwd=ROOT, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        loso = pd.read_csv(result_dir / "final_calibrated_tuned_xgboost_results.csv")
        loso.insert(0, "experiment", name)
        all_results.append(loso)
        config = pd.read_csv(result_dir / "final_global_xgboost_configuration_selection.csv")
        config.insert(0, "experiment", name)
        all_configs.append(config)
        manifests.append({
            "experiment": name,
            "error_threshold_km": target_km,
            "maximum_reference_delay_hours": delay_h,
            "samples_by_split": counts,
            "selected_configuration": str(config.iloc[0]["configuration"]),
            "loso_folds": int(len(loso)),
            "test_labels_used_for_selection": False,
            "metrics": "six held-out-satellite folds; configuration, calibration, and F1 threshold use non-held-out training/validation only",
        })
        print(f"Completed {name}: split sizes {counts}; configuration {config.iloc[0]['configuration']}")
    results = pd.concat(all_results, ignore_index=True)
    configs = pd.concat(all_configs, ignore_index=True)
    results.to_csv(OUT / "sensitivity_loso_by_satellite.csv", index=False)
    configs.to_csv(OUT / "sensitivity_configuration_selection.csv", index=False)
    mean_cols = [c for c in results.columns if c.startswith(("raw_", "calibrated_"))]
    summary = results.groupby("experiment", as_index=False)[mean_cols].mean()
    summary.to_csv(OUT / "sensitivity_loso_summary.csv", index=False)
    manifest = {
        "python": sys.version,
        "packages": {},
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_data_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in SAT_FILES},
        "protocol": "Temporal source-creation splits Jan-Aug train, Sep-Oct validation, Nov-Dec test. LOSO folds exclude the held-out satellite from train and validation; hyperparameters pooled across validation folds; Platt calibration and F1 threshold fitted on validation only.",
        "cases": manifests,
    }
    (OUT / "sensitivity_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    plot = summary.set_index("experiment")[["calibrated_pr_auc", "calibrated_roc_auc", "calibrated_brier"]]
    ax = plot.plot(kind="bar", figsize=(12, 6), rot=25)
    ax.set_title("O-RAP sensitivity: LOSO test means")
    ax.set_ylabel("Metric value")
    ax.set_xlabel("")
    ax.legend(title="Metric")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT / "sensitivity_loso_summary.png", dpi=180)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
