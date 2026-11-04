
import csv
import json
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]

SPLIT_DIR = ROOT / "results" / "modeling" / "splits"
FEATURE_DIR = ROOT / "results" / "modeling" / "features"
OUTPUT_DIR = (
    ROOT / "results" / "modeling" / "experiments"
    / "threshold_sensitivity"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "validation", "test"]
THRESHOLDS_KM = [1.0, 5.0, 10.0]

FEATURES = [
    "mean_motion",
    "eccentricity",
    "inclination",
    "raan",
    "argument_of_pericenter",
    "mean_anomaly",
    "bstar",
    "mean_motion_dot",
    "mean_motion_ddot",
    "semimajor_axis",
    "orbital_period",
    "apoapsis",
    "periapsis",
]

RANDOM_STATE = 42


def read_csv(path):
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def select_threshold(y, probabilities):
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in np.linspace(0.01, 0.99, 199):
        predictions = (probabilities >= threshold).astype(int)
        score = f1_score(y, predictions, zero_division=0)

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold


def metrics(y, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    result = {
        "samples": int(len(y)),
        "positives": int(np.sum(y)),
        "positive_rate_percent": float(np.mean(y) * 100),
        "pr_auc": (
            float(average_precision_score(y, probabilities))
            if len(np.unique(y)) == 2 else float("nan")
        ),
        "roc_auc": (
            float(roc_auc_score(y, probabilities))
            if len(np.unique(y)) == 2 else float("nan")
        ),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "log_loss": float(log_loss(
            y,
            np.clip(probabilities, 1e-7, 1 - 1e-7),
            labels=[0, 1],
        )),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "threshold": float(threshold),
    }

    return result


def main():
    print("=" * 72)
    print("O-RAP THRESHOLD SENSITIVITY EXPERIMENT")
    print("=" * 72)

    datasets = {}

    for split in SPLITS:
        split_rows = read_csv(SPLIT_DIR / f"{split}.csv")
        feature_rows = read_csv(
            FEATURE_DIR / f"{split}_features.csv"
        )

        # Map disagreement values using pair_id.
        disagreement_map = {
            row["pair_id"]: float(row["disagreement_km"])
            for row in split_rows
        }

        feature_ids = {row["pair_id"] for row in feature_rows}
        split_ids = set(disagreement_map)

        if feature_ids != split_ids:
            raise RuntimeError(
                f"Pair ID mismatch in {split}: "
                f"features={len(feature_ids)}, "
                f"split={len(split_ids)}"
            )

        datasets[split] = {
            "rows": feature_rows,
            "disagreement": disagreement_map,
        }

        print(f"{split.capitalize():12s}: {len(feature_rows):,} rows")

    results = []
    satellite_results = []

    for threshold_km in THRESHOLDS_KM:
        print("\n" + "=" * 72)
        print(f"DISAGREEMENT THRESHOLD: > {threshold_km:g} km")
        print("=" * 72)

        arrays = {}

        for split in SPLITS:
            rows = datasets[split]["rows"]
            disagreement = datasets[split]["disagreement"]

            X = np.asarray(
                [[float(row[col]) for col in FEATURES] for row in rows],
                dtype=np.float64,
            )

            y = np.asarray(
                [
                    int(disagreement[row["pair_id"]] > threshold_km)
                    for row in rows
                ],
                dtype=np.int32,
            )

            arrays[split] = (X, y, rows)

            print(
                f"{split.capitalize():12s}: "
                f"{len(y):,} samples, "
                f"{int(np.sum(y)):,} positives "
                f"({np.mean(y) * 100:.3f}%)"
            )

        X_train, y_train, _ = arrays["train"]
        X_val, y_val, _ = arrays["validation"]
        X_test, y_test, test_rows = arrays["test"]

        positives = int(np.sum(y_train))
        negatives = int(len(y_train) - positives)

        if positives == 0 or negatives == 0:
            raise RuntimeError(
                f"Training set has insufficient classes at "
                f"{threshold_km} km."
            )

        if len(np.unique(y_val)) < 2:
            raise RuntimeError(
                f"Validation set has insufficient classes at "
                f"{threshold_km} km."
            )

        model = XGBClassifier(
            n_estimators=500,
            max_depth=3,
            learning_rate=0.03,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=5,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            scale_pos_weight=negatives / positives,
            n_jobs=-1,
            tree_method="hist",
        )

        model.fit(X_train, y_train)

        # Fit calibration on validation predictions only.
        val_raw = model.predict_proba(X_val)[:, 1]
        test_raw = model.predict_proba(X_test)[:, 1]

        calibrator = LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=3000,
        )

        calibrator.fit(val_raw.reshape(-1, 1), y_val)

        val_calibrated = calibrator.predict_proba(
            val_raw.reshape(-1, 1)
        )[:, 1]

        test_calibrated = calibrator.predict_proba(
            test_raw.reshape(-1, 1)
        )[:, 1]

        chosen_threshold = select_threshold(
            y_val, val_calibrated
        )

        raw_result = metrics(y_test, test_raw, 0.5)
        calibrated_result = metrics(
            y_test, test_calibrated, chosen_threshold
        )

        result = {
            "disagreement_threshold_km": threshold_km,
            "validation_selected_probability_threshold": chosen_threshold,
        }

        for key, value in raw_result.items():
            result[f"raw_{key}"] = value

        for key, value in calibrated_result.items():
            result[f"calibrated_{key}"] = value

        results.append(result)

        # Satellite-level calibrated results.
        satellite_groups = {}

        for index, row in enumerate(test_rows):
            satellite = row["satellite_key"]
            satellite_groups.setdefault(satellite, []).append(index)

        for satellite, indices in sorted(satellite_groups.items()):
            indices = np.asarray(indices)
            y_sat = y_test[indices]
            p_sat = test_calibrated[indices]
            sat_predictions = (p_sat >= chosen_threshold).astype(int)

            satellite_results.append({
                "disagreement_threshold_km": threshold_km,
                "satellite_key": satellite,
                "satellite_name": test_rows[int(indices[0])]["satellite_name"],
                "samples": int(len(indices)),
                "positives": int(np.sum(y_sat)),
                "positive_rate_percent": float(np.mean(y_sat) * 100),
                "pr_auc": (
                    float(average_precision_score(y_sat, p_sat))
                    if len(np.unique(y_sat)) == 2 else float("nan")
                ),
                "roc_auc": (
                    float(roc_auc_score(y_sat, p_sat))
                    if len(np.unique(y_sat)) == 2 else float("nan")
                ),
                "brier_score": float(brier_score_loss(y_sat, p_sat)),
                "f1": float(
                    f1_score(y_sat, sat_predictions, zero_division=0)
                ),
            })

        print("\nCalibrated test performance:")
        for key, value in calibrated_result.items():
            print(f"{key:28s}: {value:.6f}" if isinstance(value, float)
                  else f"{key:28s}: {value}")

        print(
            f"\nValidation-selected probability threshold: "
            f"{chosen_threshold:.6f}"
        )

    write_csv(
        OUTPUT_DIR / "threshold_sensitivity_summary.csv",
        results,
    )

    write_csv(
        OUTPUT_DIR / "threshold_sensitivity_by_satellite.csv",
        satellite_results,
    )

    with open(
        OUTPUT_DIR / "threshold_sensitivity_configuration.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump({
            "disagreement_thresholds_km": THRESHOLDS_KM,
            "label_rule": "disagreement_km > threshold_km",
            "features": FEATURES,
            "excluded_feature": "interval_hours",
            "model": "XGBoost",
            "calibration": "Platt scaling fitted on validation only",
            "threshold_selection": "Maximum validation F1",
            "random_state": RANDOM_STATE,
        }, f, indent=2)

    print("\n" + "=" * 72)
    print("THRESHOLD SENSITIVITY COMPLETED")
    print("=" * 72)
    print(f"Output directory: {OUTPUT_DIR}")
    print("[PASS] Existing data splits preserved.")
    print("[PASS] Labels derived from disagreement_km.")
    print("[PASS] Only source orbital features used.")
    print("[PASS] Calibration fitted on validation only.")
    print("[PASS] Test data used only for evaluation.")
    print("[PASS] 1 km, 5 km and 10 km experiments completed.")


if __name__ == "__main__":
    main()
