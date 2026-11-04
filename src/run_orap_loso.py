
import csv
import json
import math
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

FEATURE_DIR = ROOT / "results" / "modeling" / "features"
OUTPUT_DIR = ROOT / "results" / "modeling" / "experiments" / "loso"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "train": FEATURE_DIR / "train_features.csv",
    "validation": FEATURE_DIR / "validation_features.csv",
    "test": FEATURE_DIR / "test_features.csv",
}

# Operational orbital-only inputs.
# interval_hours is deliberately excluded because it depends on
# the future reference TLE epoch.
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

TARGET = "label_1km"
SATELLITE = "satellite_key"
SATELLITE_NAME = "satellite_name"

RANDOM_STATE = 42


def load_csv(path):
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def prepare(rows):
    X = np.asarray(
        [[float(row[col]) for col in FEATURES] for row in rows],
        dtype=np.float64,
    )
    y = np.asarray(
        [int(row[TARGET]) for row in rows],
        dtype=np.int32,
    )
    return X, y


def safe_metric(function, y, probabilities):
    if len(y) == 0:
        return float("nan")

    if function in (average_precision_score, roc_auc_score):
        if len(np.unique(y)) < 2:
            return float("nan")

    try:
        return float(function(y, probabilities))
    except Exception:
        return float("nan")


def calculate_metrics(y, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "samples": int(len(y)),
        "positives": int(np.sum(y)),
        "positive_rate_percent": float(np.mean(y) * 100),
        "pr_auc": safe_metric(
            average_precision_score, y, probabilities
        ),
        "roc_auc": safe_metric(
            roc_auc_score, y, probabilities
        ),
        "brier_score": float(
            brier_score_loss(y, probabilities)
        ),
        "log_loss": float(
            log_loss(y, np.clip(probabilities, 1e-7, 1 - 1e-7),
                     labels=[0, 1])
        ),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y, predictions, zero_division=0)
        ),
        "threshold": float(threshold),
    }


def select_f1_threshold(y, probabilities):
    # Select the threshold using validation data only.
    thresholds = np.linspace(0.01, 0.99, 199)

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        score = f1_score(y, predictions, zero_division=0)

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return float(best_threshold)


def write_csv(path, rows):
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    print("=" * 72)
    print("O-RAP LEAVE-ONE-SATELLITE-OUT EVALUATION")
    print("=" * 72)

    datasets = {
        name: load_csv(path)
        for name, path in FILES.items()
    }

    for name, rows in datasets.items():
        print(f"{name.capitalize():12s}: {len(rows):,} rows")

    # Confirm all required features and values are present.
    for split_name, rows in datasets.items():
        if not rows:
            raise RuntimeError(f"{split_name} dataset is empty.")

        required = FEATURES + [TARGET, SATELLITE, SATELLITE_NAME]
        missing = [
            col for col in required
            if col not in rows[0]
        ]

        if missing:
            raise RuntimeError(
                f"Missing columns in {split_name}: {missing}"
            )

    # Use all 14 satellites represented in the training partition.
    satellites = sorted({
        row[SATELLITE] for row in datasets["train"]
    })

    if len(satellites) != 14:
        raise RuntimeError(
            f"Expected 14 satellites, found {len(satellites)}: "
            f"{satellites}"
        )

    print(f"\nSatellites: {len(satellites)}")
    print(f"Features: {len(FEATURES)}")
    print("Excluded feature: interval_hours")

    fold_results = []
    all_predictions = []
    all_importance = []

    for fold_number, held_out in enumerate(satellites, start=1):
        print("\n" + "-" * 72)
        print(
            f"FOLD {fold_number:02d}/14 | "
            f"Held-out satellite: {held_out}"
        )
        print("-" * 72)

        train_rows = [
            row for row in datasets["train"]
            if row[SATELLITE] != held_out
        ]

        validation_rows = [
            row for row in datasets["validation"]
            if row[SATELLITE] != held_out
        ]

        test_rows = [
            row for row in datasets["test"]
            if row[SATELLITE] == held_out
        ]

        if not train_rows or not validation_rows or not test_rows:
            raise RuntimeError(
                f"Empty partition in fold for {held_out}"
            )

        X_train, y_train = prepare(train_rows)
        X_val, y_val = prepare(validation_rows)
        X_test, y_test = prepare(test_rows)

        train_positives = int(np.sum(y_train))
        train_negatives = int(len(y_train) - train_positives)

        if train_positives == 0 or len(np.unique(y_val)) < 2:
            raise RuntimeError(
                f"Insufficient classes for training/calibration "
                f"in fold {held_out}"
            )

        scale_pos_weight = train_negatives / train_positives

        print(
            f"Training:   {len(y_train):,} "
            f"(positive {train_positives:,})"
        )
        print(
            f"Validation: {len(y_val):,} "
            f"(positive {int(np.sum(y_val)):,})"
        )
        print(
            f"Test:       {len(y_test):,} "
            f"(positive {int(np.sum(y_test)):,})"
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
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            tree_method="hist",
        )

        model.fit(X_train, y_train)

        # Calibration is fitted only on the 13 training satellites'
        # validation partition, never on the held-out test satellite.
        val_raw = model.predict_proba(X_val)[:, 1]
        test_raw = model.predict_proba(X_test)[:, 1]

        calibrator = LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=3000,
        )

        calibrator.fit(
            val_raw.reshape(-1, 1),
            y_val,
        )

        val_calibrated = calibrator.predict_proba(
            val_raw.reshape(-1, 1)
        )[:, 1]

        test_calibrated = calibrator.predict_proba(
            test_raw.reshape(-1, 1)
        )[:, 1]

        threshold = select_f1_threshold(
            y_val, val_calibrated
        )

        raw_metrics = calculate_metrics(
            y_test, test_raw, 0.5
        )

        calibrated_metrics = calculate_metrics(
            y_test, test_calibrated, threshold
        )

        name = test_rows[0][SATELLITE_NAME]

        result = {
            "held_out_satellite": held_out,
            "satellite_name": name,
            "train_samples": len(y_train),
            "validation_samples": len(y_val),
            "test_samples": len(y_test),
            "test_positives": int(np.sum(y_test)),
            "validation_selected_threshold": threshold,
        }

        for key, value in raw_metrics.items():
            result[f"raw_{key}"] = value

        for key, value in calibrated_metrics.items():
            result[f"calibrated_{key}"] = value

        fold_results.append(result)

        for feature, importance in zip(
            FEATURES, model.feature_importances_
        ):
            all_importance.append({
                "held_out_satellite": held_out,
                "feature": feature,
                "importance": float(importance),
            })

        for row, actual, raw_p, calibrated_p in zip(
            test_rows, y_test, test_raw, test_calibrated
        ):
            all_predictions.append({
                "pair_id": row["pair_id"],
                "held_out_satellite": held_out,
                "satellite_name": row[SATELLITE_NAME],
                "actual_label": int(actual),
                "raw_probability": float(raw_p),
                "calibrated_probability": float(calibrated_p),
                "validation_threshold": threshold,
                "predicted_label": int(
                    calibrated_p >= threshold
                ),
            })

        print(
            f"Calibrated PR-AUC: "
            f"{calibrated_metrics['pr_auc']:.5f}"
        )
        print(
            f"Calibrated ROC-AUC: "
            f"{calibrated_metrics['roc_auc']:.5f}"
        )
        print(
            f"Calibrated Brier: "
            f"{calibrated_metrics['brier_score']:.5f}"
        )
        print(
            f"Calibrated F1: "
            f"{calibrated_metrics['f1']:.5f}"
        )

    # Save per-satellite fold results.
    write_csv(
        OUTPUT_DIR / "loso_by_satellite.csv",
        fold_results,
    )

    write_csv(
        OUTPUT_DIR / "loso_test_predictions.csv",
        all_predictions,
    )

    write_csv(
        OUTPUT_DIR / "loso_feature_importance.csv",
        all_importance,
    )

    # Pooled out-of-satellite test predictions.
    pooled_y = np.asarray(
        [row["actual_label"] for row in all_predictions]
    )

    pooled_p = np.asarray(
        [row["calibrated_probability"] for row in all_predictions]
    )

    pooled_raw = np.asarray(
        [row["raw_probability"] for row in all_predictions]
    )

    pooled_thresholded = np.asarray(
        [row["predicted_label"] for row in all_predictions]
    )

    pooled = {
        "samples": int(len(pooled_y)),
        "positives": int(np.sum(pooled_y)),
        "positive_rate_percent": float(np.mean(pooled_y) * 100),
        "raw_pr_auc": safe_metric(
            average_precision_score, pooled_y, pooled_raw
        ),
        "raw_roc_auc": safe_metric(
            roc_auc_score, pooled_y, pooled_raw
        ),
        "calibrated_pr_auc": safe_metric(
            average_precision_score, pooled_y, pooled_p
        ),
        "calibrated_roc_auc": safe_metric(
            roc_auc_score, pooled_y, pooled_p
        ),
        "calibrated_brier_score": float(
            brier_score_loss(pooled_y, pooled_p)
        ),
        "calibrated_log_loss": float(
            log_loss(
                pooled_y,
                np.clip(pooled_p, 1e-7, 1 - 1e-7),
                labels=[0, 1],
            )
        ),
        "calibrated_precision": float(
            precision_score(
                pooled_y, pooled_thresholded, zero_division=0
            )
        ),
        "calibrated_recall": float(
            recall_score(
                pooled_y, pooled_thresholded, zero_division=0
            )
        ),
        "calibrated_f1": float(
            f1_score(
                pooled_y, pooled_thresholded, zero_division=0
            )
        ),
    }

    with open(
        OUTPUT_DIR / "loso_pooled_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(pooled, f, indent=2)

    print("\n" + "=" * 72)
    print("POOLED LOSO RESULTS")
    print("=" * 72)

    for key, value in pooled.items():
        if isinstance(value, float):
            print(f"{key:32s}: {value:.6f}")
        else:
            print(f"{key:32s}: {value}")

    print("\n" + "=" * 72)
    print("LOSO EVALUATION COMPLETED")
    print("=" * 72)
    print(f"Output: {OUTPUT_DIR}")
    print("[PASS] 14 held-out satellite folds completed.")
    print("[PASS] Calibration fitted using validation data only.")
    print("[PASS] Held-out satellite test data used only for evaluation.")
    print("[PASS] Pooled out-of-satellite predictions saved.")
    print("[PASS] Per-satellite results and feature importance saved.")


if __name__ == "__main__":
    main()
