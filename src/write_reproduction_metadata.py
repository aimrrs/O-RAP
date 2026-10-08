from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ENV_PACKAGES = ["numpy", "pandas", "scikit-learn", "scipy", "sgp4", "xgboost", "matplotlib"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    temporal_dir = ROOT / "results" / "experiments" / "temporal_reproduction"
    loso_dir = ROOT / "results" / "experiments" / "loso_reproduction"
    temporal = pd.read_csv(temporal_dir / "tables" / "xgboost_temporal_results.csv")
    loso = pd.read_csv(loso_dir / "tables" / "final_calibrated_tuned_xgboost_results.csv")
    best = pd.read_csv(loso_dir / "tables" / "final_global_xgboost_configuration_selection.csv").iloc[0]
    packages = {name: metadata.version(name) for name in ENV_PACKAGES}
    common = {
        "python": sys.version,
        "packages": packages,
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_tree_had_uncommitted_audit_changes": True,
        "input_split_sha256": {
            path.name: sha256(path)
            for path in sorted((ROOT / "data" / "processed" / "splits").glob("*.csv"))
        },
    }
    temporal_record = {
        **common,
        "command": ".\\venv\\Scripts\\python.exe src\\run_xgboost.py with ORAP_RESULTS_DIR set to results/experiments/temporal_reproduction/tables",
        "script_sha256": {"run_xgboost.py": sha256(ROOT / "src" / "run_xgboost.py")},
        "threshold_selection": "training split only",
        "result_rows": temporal.to_dict(orient="records"),
    }
    loso_record = {
        **common,
        "command": ".\\venv\\Scripts\\python.exe src\\run_final_tuned_xgboost.py with ORAP_RESULTS_DIR set to results/experiments/loso_reproduction/tables and PYTHONIOENCODING=utf-8",
        "script_sha256": {"run_final_tuned_xgboost.py": sha256(ROOT / "src" / "run_final_tuned_xgboost.py")},
        "selected_configuration": str(best["configuration"]),
        "held_out_satellite_folds": int(len(loso)),
        "mean_test_metrics": {col: float(loso[col].mean()) for col in loso.select_dtypes(include="number").columns if col.startswith(("raw_", "calibrated_"))},
        "threshold_and_calibration_data": "per-fold non-held-out validation rows only; no held-out test labels used",
    }
    (temporal_dir / "metadata.json").write_text(json.dumps(temporal_record, indent=2), encoding="utf-8")
    (loso_dir / "metadata.json").write_text(json.dumps(loso_record, indent=2), encoding="utf-8")
    print("Wrote temporal and LOSO metadata JSON.")


if __name__ == "__main__":
    main()
