from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "experiments" / "sensitivity"
NAMES = [
    "error_threshold_0p5km", "error_threshold_1km", "error_threshold_2km",
    "error_threshold_5km", "reference_delay_1h", "reference_delay_2h",
    "reference_delay_4h", "reference_delay_6h",
]
METRICS = [
    "raw_roc_auc", "calibrated_roc_auc", "raw_pr_auc", "calibrated_pr_auc",
    "raw_brier", "calibrated_brier", "raw_logloss", "calibrated_logloss",
    "raw_f1", "calibrated_f1",
]


def read_case(name):
    if name in {"error_threshold_1km", "reference_delay_6h"}:
        folder = ROOT / "results" / "experiments" / "loso_reproduction" / "tables"
    else:
        folder = OUT / name / "tables"
    results = pd.read_csv(folder / "final_calibrated_tuned_xgboost_results.csv")
    results.insert(0, "experiment", name)
    configs = pd.read_csv(folder / "final_global_xgboost_configuration_selection.csv")
    configs.insert(0, "experiment", name)
    return results, configs


def main():
    all_results, all_configs = [], []
    for name in NAMES:
        results, configs = read_case(name)
        all_results.append(results)
        all_configs.append(configs)
    by_satellite = pd.concat(all_results, ignore_index=True)
    configurations = pd.concat(all_configs, ignore_index=True)
    summary = by_satellite.groupby("experiment", as_index=False)[METRICS].mean()
    by_satellite.to_csv(OUT / "sensitivity_loso_by_satellite.csv", index=False)
    configurations.to_csv(OUT / "sensitivity_configuration_selection.csv", index=False)
    summary.to_csv(OUT / "sensitivity_loso_summary.csv", index=False)
    original_manifest_path = OUT / "sensitivity_manifest.json"
    manifest = json.loads(original_manifest_path.read_text(encoding="utf-8"))
    baseline_names = {"error_threshold_1km", "reference_delay_6h"}
    manifest["cases"] = [case for case in manifest["cases"] if case["experiment"] not in baseline_names]
    for name, target, delay in [
        ("error_threshold_1km", 1.0, 6.0), ("reference_delay_6h", 1.0, 6.0)
    ]:
        folder = ROOT / "results" / "experiments" / "loso_reproduction" / "tables"
        cfg = pd.read_csv(folder / "final_global_xgboost_configuration_selection.csv")
        manifest["cases"].append({
            "experiment": name,
            "error_threshold_km": target,
            "maximum_reference_delay_hours": delay,
            "samples_by_split": {"train_temporal": 12657, "validation_temporal": 2011, "test_temporal": 2595},
            "selected_configuration": str(cfg.iloc[0]["configuration"]),
            "loso_folds": 6,
            "source": "Independent reproduction of the 1 km / 6 h baseline, recorded in loso_reproduction/tables.",
            "test_labels_used_for_selection": False,
        })
    manifest["summary_metrics_are_unweighted_means_of_six_held_out_satellite_folds"] = True
    manifest["reproducibility_scripts_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [ROOT / "src" / "run_final_tuned_xgboost.py", ROOT / "src" / "run_sensitivity_experiments.py", Path(__file__)]
    }
    manifest["source_git_head_during_aggregation"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest["package_versions"] = {name: metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "scipy", "sgp4", "xgboost", "matplotlib"]}
    manifest["source_code_was_modified_since_base_commit"] = True
    original_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    ax = summary.set_index("experiment")[["calibrated_pr_auc", "calibrated_roc_auc", "calibrated_brier"]].plot(kind="bar", figsize=(13, 6), rot=25)
    ax.set_title("O-RAP sensitivity: mean LOSO test metrics")
    ax.set_ylabel("Mean across held-out satellites")
    ax.set_xlabel("")
    ax.legend(title="Metric")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT / "sensitivity_loso_summary.png", dpi=180)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
