
import csv
import json
import joblib
import numpy as np

from pathlib import Path
from collections import defaultdict

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)

# ============================================================
# O-RAP: ORBITAL-ONLY CALIBRATION AND ANALYSIS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FEATURE_DIR = ROOT / "results" / "modeling" / "features"
EXPERIMENT_DIR = ROOT / "results" / "modeling" / "experiments"
MODEL_DIR = EXPERIMENT_DIR / "models"

OUTPUT_DIR = EXPERIMENT_DIR / "orbital_only_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "orbital_only_xgboost.joblib"

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


def load_data(partition):

    path = FEATURE_DIR / f"{partition}_features.csv"

    X_values = []
    y_values = []
    satellites = []
    pair_ids = []

    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:

        reader = csv.DictReader(file)

        required = set(
            FEATURES + [TARGET, "satellite_name", "pair_id"]
        )

        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"Missing columns in {partition}: {sorted(missing)}"
            )

        for row in reader:

            X_values.append([
                float(row[feature])
                for feature in FEATURES
            ])

            y_values.append(int(row[TARGET]))
            satellites.append(row["satellite_name"])
            pair_ids.append(row["pair_id"])

    X = np.asarray(X_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.int32)

    if not np.isfinite(X).all():
        raise ValueError(f"Invalid features in {partition}")

    return X, y, satellites, pair_ids


def logits(probabilities):

    p = np.clip(probabilities, 1e-7, 1 - 1e-7)

    return np.log(p / (1 - p)).reshape(-1, 1)


def choose_threshold(y, probabilities):

    precision, recall, thresholds = precision_recall_curve(
        y, probabilities
    )

    if len(thresholds) == 0:
        return 0.5

    denominator = precision[:-1] + recall[:-1]

    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )

    return float(thresholds[int(np.argmax(f1))])


def calculate_metrics(y, probabilities, threshold):

    p = np.clip(probabilities, 1e-7, 1 - 1e-7)
    predictions = (p >= threshold).astype(int)

    result = {
        "samples": len(y),
        "positives": int(y.sum()),
        "positive_rate_percent": float(y.mean() * 100),
        "pr_auc": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
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

    result["roc_auc"] = (
        float(roc_auc_score(y, p))
        if len(np.unique(y)) == 2
        else None
    )

    return result


def save_csv(path, rows, fields):

    with path.open(
        "w", encoding="utf-8", newline=""
    ) as file:

        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():

    print("=" * 72)
    print("O-RAP ORBITAL-ONLY CALIBRATION AND ANALYSIS")
    print("=" * 72)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Previously trained model not found: {MODEL_PATH}"
        )

    print("\nLoading datasets...")

    X_train, y_train, _, _ = load_data("train")
    X_val, y_val, _, _ = load_data("validation")
    X_test, y_test, test_satellites, test_pair_ids = load_data("test")

    print(f"Training:   {len(y_train):,}")
    print(f"Validation: {len(y_val):,}")
    print(f"Testing:    {len(y_test):,}")

    print("\nLoading existing orbital-only XGBoost model...")

    model = joblib.load(MODEL_PATH)

    # --------------------------------------------------------
    # 1. Generate raw validation and test probabilities
    # --------------------------------------------------------

    raw_val = model.predict_proba(X_val)[:, 1]
    raw_test = model.predict_proba(X_test)[:, 1]

    # --------------------------------------------------------
    # 2. Fit Platt calibration using validation only
    # --------------------------------------------------------

    print("\nFitting Platt calibration on validation predictions...")

    calibrator = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        random_state=42,
    )

    calibrator.fit(logits(raw_val), y_val)

    calibrated_val = calibrator.predict_proba(
        logits(raw_val)
    )[:, 1]

    calibrated_test = calibrator.predict_proba(
        logits(raw_test)
    )[:, 1]

    threshold = choose_threshold(y_val, calibrated_val)

    joblib.dump(
        calibrator,
        OUTPUT_DIR / "orbital_only_platt_calibrator.joblib",
    )

    # --------------------------------------------------------
    # 3. Overall test metrics before and after calibration
    # --------------------------------------------------------

    raw_metrics = calculate_metrics(
        y_test, raw_test, 0.5
    )

    calibrated_metrics = calculate_metrics(
        y_test, calibrated_test, threshold
    )

    results = [
        {
            "model": "orbital_only_xgboost_raw",
            **raw_metrics,
        },
        {
            "model": "orbital_only_xgboost_platt",
            **calibrated_metrics,
        },
    ]

    save_csv(
        OUTPUT_DIR / "orbital_only_calibration_metrics.csv",
        results,
        list(results[0].keys()),
    )

    print("\nOVERALL TEST RESULTS")
    print("-" * 72)

    for result in results:

        print(f"\n{result['model']}")
        for key, value in result.items():

            if key == "model":
                continue

            if isinstance(value, float):
                print(f"{key:28s}: {value:.6f}")
            else:
                print(f"{key:28s}: {value}")

    print(
        f"\nValidation-selected calibrated F1 threshold: "
        f"{threshold:.6f}"
    )

    # --------------------------------------------------------
    # 4. Per-satellite test performance
    # --------------------------------------------------------

    print("\nPER-SATELLITE TEST PERFORMANCE")
    print("-" * 72)

    groups = defaultdict(list)

    for index, satellite in enumerate(test_satellites):
        groups[satellite].append(index)

    satellite_results = []

    for satellite, indices in sorted(groups.items()):

        idx = np.asarray(indices, dtype=int)

        y = y_test[idx]
        p = calibrated_test[idx]

        if len(np.unique(y)) == 2:
            pr_auc = float(average_precision_score(y, p))
            roc_auc = float(roc_auc_score(y, p))
        else:
            pr_auc = None
            roc_auc = None

        predictions = (p >= threshold).astype(int)

        result = {
            "satellite": satellite,
            "samples": len(idx),
            "positives": int(y.sum()),
            "positive_rate_percent": float(y.mean() * 100),
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "brier_score": float(brier_score_loss(y, p)),
            "precision": float(
                precision_score(y, predictions, zero_division=0)
            ),
            "recall": float(
                recall_score(y, predictions, zero_division=0)
            ),
            "f1": float(
                f1_score(y, predictions, zero_division=0)
            ),
        }

        satellite_results.append(result)

        print(
            f"{satellite:22s} "
            f"N={len(idx):6,} "
            f"Pos={int(y.sum()):5,} "
            f"PR-AUC={pr_auc if pr_auc is not None else 'N/A'} "
            f"Brier={result['brier_score']:.5f}"
        )

    save_csv(
        OUTPUT_DIR / "orbital_only_per_satellite.csv",
        satellite_results,
        list(satellite_results[0].keys()),
    )

    # --------------------------------------------------------
    # 5. Feature importance
    # --------------------------------------------------------

    print("\nORBITAL FEATURE IMPORTANCE")
    print("-" * 72)

    importance = model.feature_importances_

    importance_rows = sorted(
        [
            {
                "feature": feature,
                "importance": float(value),
            }
            for feature, value in zip(FEATURES, importance)
        ],
        key=lambda row: row["importance"],
        reverse=True,
    )

    save_csv(
        OUTPUT_DIR / "orbital_only_feature_importance.csv",
        importance_rows,
        ["feature", "importance"],
    )

    for row in importance_rows:
        print(
            f"{row['feature']:30s} "
            f"{row['importance']:.6f}"
        )

    # --------------------------------------------------------
    # 6. Calibration reliability bins on test set
    # --------------------------------------------------------

    print("\nTEST CALIBRATION BINS")
    print("-" * 72)

    bin_rows = []

    bin_edges = np.linspace(0.0, 1.0, 11)

    for bin_index in range(10):

        lower = bin_edges[bin_index]
        upper = bin_edges[bin_index + 1]

        if bin_index == 9:
            mask = (
                (calibrated_test >= lower)
                & (calibrated_test <= upper)
            )
        else:
            mask = (
                (calibrated_test >= lower)
                & (calibrated_test < upper)
            )

        count = int(mask.sum())

        if count == 0:
            continue

        mean_predicted = float(
            calibrated_test[mask].mean()
        )

        observed_rate = float(
            y_test[mask].mean()
        )

        bin_rows.append({
            "bin": bin_index + 1,
            "lower_probability": lower,
            "upper_probability": upper,
            "samples": count,
            "mean_predicted_probability": mean_predicted,
            "observed_positive_rate": observed_rate,
            "absolute_gap": abs(
                mean_predicted - observed_rate
            ),
        })

        print(
            f"{lower:.1f}-{upper:.1f} "
            f"N={count:6,} "
            f"Predicted={mean_predicted:.5f} "
            f"Observed={observed_rate:.5f}"
        )

    save_csv(
        OUTPUT_DIR / "orbital_only_calibration_bins.csv",
        bin_rows,
        [
            "bin",
            "lower_probability",
            "upper_probability",
            "samples",
            "mean_predicted_probability",
            "observed_positive_rate",
            "absolute_gap",
        ],
    )

    # --------------------------------------------------------
    # 7. Save test predictions
    # --------------------------------------------------------

    prediction_rows = []

    for index in range(len(y_test)):

        prediction_rows.append({
            "pair_id": test_pair_ids[index],
            "satellite": test_satellites[index],
            "actual_label": int(y_test[index]),
            "raw_probability": float(raw_test[index]),
            "calibrated_probability": float(
                calibrated_test[index]
            ),
            "predicted_label": int(
                calibrated_test[index] >= threshold
            ),
        })

    save_csv(
        OUTPUT_DIR / "orbital_only_test_predictions.csv",
        prediction_rows,
        [
            "pair_id",
            "satellite",
            "actual_label",
            "raw_probability",
            "calibrated_probability",
            "predicted_label",
        ],
    )

    # --------------------------------------------------------
    # 8. Save calibration parameters and configuration
    # --------------------------------------------------------

    configuration = {
        "model": "orbital_only_xgboost",
        "features": FEATURES,
        "calibration": "Platt scaling",
        "calibration_training_partition": "validation",
        "calibration_test_partition": "test",
        "platt_coefficient": float(calibrator.coef_[0][0]),
        "platt_intercept": float(calibrator.intercept_[0]),
        "validation_selected_f1_threshold": threshold,
        "test_metrics_raw": raw_metrics,
        "test_metrics_calibrated": calibrated_metrics,
    }

    with (
        OUTPUT_DIR / "orbital_only_analysis_configuration.json"
    ).open("w", encoding="utf-8") as file:

        json.dump(configuration, file, indent=4)

    print("\n" + "=" * 72)
    print("ORBITAL-ONLY ANALYSIS COMPLETED")
    print("=" * 72)

    print("\nOUTPUT DIRECTORY:")
    print(OUTPUT_DIR)

    print("\n[PASS] Calibration fitted on validation only.")
    print("[PASS] Test set used only for evaluation.")
    print("[PASS] Per-satellite results generated.")
    print("[PASS] Feature importance generated.")
    print("[PASS] Calibration bins generated.")
    print("\nSUCCESS.")


if __name__ == "__main__":
    main()
