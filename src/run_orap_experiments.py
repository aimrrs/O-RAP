
import csv
import json
import joblib
import numpy as np

from pathlib import Path
from collections import Counter

from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    log_loss,
)


# ============================================================
# O-RAP: MAIN EXPERIMENT PIPELINE
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "results" / "modeling" / "features"
OUTPUT_DIR = ROOT / "results" / "modeling" / "experiments"
MODEL_DIR = OUTPUT_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

TARGET = "label_1km"

INTERVAL_FEATURES = [
    "interval_hours",
]

ORBITAL_FEATURES = [
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

COMBINED_FEATURES = INTERVAL_FEATURES + ORBITAL_FEATURES

PARTITIONS = ["train", "validation", "test"]


# ============================================================
# CSV LOADING
# ============================================================

def load_partition(partition, features):

    path = DATA_DIR / f"{partition}_features.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    X_values = []
    y_values = []
    pair_ids = []
    satellite_names = []

    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:

        reader = csv.DictReader(file)

        required = set(features + [TARGET, "pair_id", "satellite_name"])

        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"Missing columns in {path}: {sorted(missing)}"
            )

        for row in reader:

            values = [
                float(row[name])
                for name in features
            ]

            label = int(row[TARGET])

            if label not in (0, 1):
                raise ValueError(
                    f"Invalid target in pair {row['pair_id']}"
                )

            X_values.append(values)
            y_values.append(label)
            pair_ids.append(row["pair_id"])
            satellite_names.append(row["satellite_name"])

    X = np.asarray(X_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.int32)

    if not np.isfinite(X).all():
        raise ValueError(
            f"Non-finite feature values in {partition}"
        )

    if len(X) == 0:
        raise ValueError(f"Empty partition: {partition}")

    return X, y, pair_ids, satellite_names


# ============================================================
# METRIC HELPERS
# ============================================================

def select_f1_threshold(y_true, probabilities):

    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )

    if len(thresholds) == 0:
        return 0.5

    denominator = precision[:-1] + recall[:-1]

    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )

    best_index = int(np.argmax(f1_values))

    return float(thresholds[best_index])


def calculate_metrics(
    y_true,
    probabilities,
    threshold,
    model_name,
    partition,
):

    probabilities = np.clip(
        probabilities, 1e-7, 1 - 1e-7
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    if len(np.unique(y_true)) == 2:
        roc_auc = roc_auc_score(y_true, probabilities)
    else:
        roc_auc = float("nan")

    return {
        "model": model_name,
        "partition": partition,
        "samples": len(y_true),
        "positives": int(np.sum(y_true)),
        "positive_rate_percent": round(
            float(np.mean(y_true) * 100), 5
        ),
        "pr_auc": float(
            average_precision_score(y_true, probabilities)
        ),
        "roc_auc": float(roc_auc),
        "precision": float(
            precision_score(
                y_true, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true, predictions, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true, predictions, zero_division=0
            )
        ),
        "brier_score": float(
            brier_score_loss(y_true, probabilities)
        ),
        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=[0, 1],
            )
        ),
        "decision_threshold": float(threshold),
    }


# ============================================================
# PLATT CALIBRATION
# ============================================================

def probability_logits(probabilities):

    probabilities = np.clip(
        probabilities, 1e-7, 1 - 1e-7
    )

    return np.log(
        probabilities / (1 - probabilities)
    ).reshape(-1, 1)


def fit_platt_calibrator(y_validation, validation_probabilities):

    calibrator = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    calibrator.fit(
        probability_logits(validation_probabilities),
        y_validation,
    )

    return calibrator


def apply_platt_calibration(calibrator, probabilities):

    return calibrator.predict_proba(
        probability_logits(probabilities)
    )[:, 1]


# ============================================================
# XGBOOST CONFIGURATION
# ============================================================

def create_xgb_model(scale_pos_weight):

    return XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=5.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0,
    )


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 76)
    print("O-RAP MAIN EXPERIMENT PIPELINE")
    print("=" * 76)

    datasets = {}

    # --------------------------------------------------------
    # Load all partitions
    # --------------------------------------------------------

    print("\nLOADING DATASETS")

    for partition in PARTITIONS:

        X, y, pair_ids, satellites = load_partition(
            partition, COMBINED_FEATURES
        )

        datasets[partition] = {
            "X": X,
            "y": y,
            "pair_ids": pair_ids,
            "satellites": satellites,
        }

        positives = int(y.sum())
        negatives = len(y) - positives

        print(
            f"{partition.upper():12s} "
            f"samples={len(y):,} "
            f"positive={positives:,} "
            f"negative={negatives:,} "
            f"positive_rate={np.mean(y)*100:.4f}%"
        )

    X_train = datasets["train"]["X"]
    y_train = datasets["train"]["y"]

    X_val = datasets["validation"]["X"]
    y_val = datasets["validation"]["y"]

    X_test = datasets["test"]["X"]
    y_test = datasets["test"]["y"]

    if len(np.unique(y_train)) != 2:
        raise ValueError("Training set must contain both classes.")

    if len(np.unique(y_val)) != 2:
        raise ValueError("Validation set must contain both classes.")

    if len(np.unique(y_test)) != 2:
        raise ValueError("Test set must contain both classes.")

    scale_pos_weight = (
        np.sum(y_train == 0) / np.sum(y_train == 1)
    )

    print(
        f"\nTraining class weight ratio: "
        f"{scale_pos_weight:.4f}"
    )

    results = []
    prediction_outputs = {}

    # --------------------------------------------------------
    # Experiment definitions
    # --------------------------------------------------------

    experiments = [
        {
            "name": "interval_only_xgboost",
            "features": INTERVAL_FEATURES,
        },
        {
            "name": "orbital_only_xgboost",
            "features": ORBITAL_FEATURES,
        },
        {
            "name": "combined_xgboost",
            "features": COMBINED_FEATURES,
        },
    ]

    trained_models = {}

    # --------------------------------------------------------
    # E1, E2, E3: XGBoost comparisons
    # --------------------------------------------------------

    for experiment in experiments:

        name = experiment["name"]
        features = experiment["features"]

        print("\n" + "-" * 76)
        print(f"TRAINING: {name}")
        print(f"Feature count: {len(features)}")
        print("-" * 76)

        train_indices = [
            COMBINED_FEATURES.index(feature)
            for feature in features
        ]

        Xtr = X_train[:, train_indices]
        Xva = X_val[:, train_indices]
        Xte = X_test[:, train_indices]

        model = create_xgb_model(scale_pos_weight)

        model.fit(Xtr, y_train)

        trained_models[name] = model

        joblib.dump(
            model,
            MODEL_DIR / f"{name}.joblib"
        )

        val_probabilities = model.predict_proba(Xva)[:, 1]
        test_probabilities = model.predict_proba(Xte)[:, 1]

        threshold = select_f1_threshold(
            y_val, val_probabilities
        )

        print(
            f"Validation-selected F1 threshold: {threshold:.6f}"
        )

        for partition, labels, probabilities in [
            ("validation", y_val, val_probabilities),
            ("test", y_test, test_probabilities),
        ]:

            metrics = calculate_metrics(
                labels,
                probabilities,
                threshold,
                name,
                partition,
            )

            results.append(metrics)

            print(
                f"{partition.upper():12s} "
                f"PR-AUC={metrics['pr_auc']:.5f} "
                f"ROC-AUC={metrics['roc_auc']:.5f} "
                f"F1={metrics['f1']:.5f} "
                f"Brier={metrics['brier_score']:.5f}"
            )

        prediction_outputs[name] = test_probabilities

    # --------------------------------------------------------
    # E4: Logistic Regression comparison
    # --------------------------------------------------------

    print("\n" + "-" * 76)
    print("TRAINING: combined_logistic_regression")
    print("-" * 76)

    logistic_model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=3000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    logistic_model.fit(X_train, y_train)

    joblib.dump(
        logistic_model,
        MODEL_DIR / "combined_logistic_regression.joblib"
    )

    logistic_val = logistic_model.predict_proba(X_val)[:, 1]
    logistic_test = logistic_model.predict_proba(X_test)[:, 1]

    logistic_threshold = select_f1_threshold(
        y_val, logistic_val
    )

    for partition, labels, probabilities in [
        ("validation", y_val, logistic_val),
        ("test", y_test, logistic_test),
    ]:

        metrics = calculate_metrics(
            labels,
            probabilities,
            logistic_threshold,
            "combined_logistic_regression",
            partition,
        )

        results.append(metrics)

        print(
            f"{partition.upper():12s} "
            f"PR-AUC={metrics['pr_auc']:.5f} "
            f"ROC-AUC={metrics['roc_auc']:.5f} "
            f"F1={metrics['f1']:.5f} "
            f"Brier={metrics['brier_score']:.5f}"
        )

    prediction_outputs[
        "combined_logistic_regression"
    ] = logistic_test

    # --------------------------------------------------------
    # E5: Platt calibration of combined XGBoost
    # --------------------------------------------------------

    print("\n" + "-" * 76)
    print("PLATT CALIBRATION: combined_xgboost")
    print("-" * 76)

    combined_model = trained_models["combined_xgboost"]

    combined_val = combined_model.predict_proba(
        X_val
    )[:, 1]

    combined_test = prediction_outputs[
        "combined_xgboost"
    ]

    calibrator = fit_platt_calibrator(
        y_val, combined_val
    )

    calibrated_val = apply_platt_calibration(
        calibrator, combined_val
    )

    calibrated_test = apply_platt_calibration(
        calibrator, combined_test
    )

    calibrated_threshold = select_f1_threshold(
        y_val, calibrated_val
    )

    joblib.dump(
        calibrator,
        MODEL_DIR / "combined_xgboost_platt_calibrator.joblib"
    )

    for partition, labels, probabilities in [
        ("validation", y_val, calibrated_val),
        ("test", y_test, calibrated_test),
    ]:

        metrics = calculate_metrics(
            labels,
            probabilities,
            calibrated_threshold,
            "combined_xgboost_platt",
            partition,
        )

        results.append(metrics)

        print(
            f"{partition.upper():12s} "
            f"PR-AUC={metrics['pr_auc']:.5f} "
            f"ROC-AUC={metrics['roc_auc']:.5f} "
            f"F1={metrics['f1']:.5f} "
            f"Brier={metrics['brier_score']:.5f} "
            f"LogLoss={metrics['log_loss']:.5f}"
        )

    prediction_outputs[
        "combined_xgboost_platt"
    ] = calibrated_test

    # --------------------------------------------------------
    # Save all metrics
    # --------------------------------------------------------

    metrics_file = OUTPUT_DIR / "model_comparison.csv"

    metric_fields = list(results[0].keys())

    with metrics_file.open(
        "w", encoding="utf-8", newline=""
    ) as file:

        writer = csv.DictWriter(
            file, fieldnames=metric_fields
        )

        writer.writeheader()
        writer.writerows(results)

    # --------------------------------------------------------
    # Save test predictions
    # --------------------------------------------------------

    prediction_file = OUTPUT_DIR / "test_predictions.csv"

    prediction_fields = [
        "pair_id",
        "satellite_name",
        "actual_label",
    ] + [
        f"{name}_probability"
        for name in prediction_outputs
    ]

    with prediction_file.open(
        "w", encoding="utf-8", newline=""
    ) as file:

        writer = csv.DictWriter(
            file, fieldnames=prediction_fields
        )

        writer.writeheader()

        for index in range(len(y_test)):

            row = {
                "pair_id": datasets["test"]["pair_ids"][index],
                "satellite_name": datasets["test"]["satellites"][index],
                "actual_label": int(y_test[index]),
            }

            for name, probabilities in prediction_outputs.items():
                row[f"{name}_probability"] = float(
                    probabilities[index]
                )

            writer.writerow(row)

    # --------------------------------------------------------
    # Save combined XGBoost feature importance
    # --------------------------------------------------------

    importance_file = (
        OUTPUT_DIR / "combined_feature_importance.csv"
    )

    importance = combined_model.feature_importances_

    importance_rows = sorted(
        zip(COMBINED_FEATURES, importance),
        key=lambda item: item[1],
        reverse=True,
    )

    with importance_file.open(
        "w", encoding="utf-8", newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(["feature", "importance"])

        for feature, value in importance_rows:
            writer.writerow([feature, float(value)])

    # --------------------------------------------------------
    # Save experiment configuration
    # --------------------------------------------------------

    configuration = {
        "random_state": RANDOM_STATE,
        "target": TARGET,
        "positive_class": 1,
        "features": COMBINED_FEATURES,
        "training_scale_pos_weight": float(scale_pos_weight),
        "xgboost": {
            "n_estimators": 500,
            "max_depth": 3,
            "learning_rate": 0.03,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.5,
            "reg_lambda": 5.0,
        },
        "calibration": "Platt scaling fitted on validation predictions",
        "threshold_selection": (
            "Maximum validation F1, applied unchanged to test"
        ),
        "test_usage": "Final evaluation only",
    }

    with (
        OUTPUT_DIR / "experiment_configuration.json"
    ).open("w", encoding="utf-8") as file:

        json.dump(configuration, file, indent=4)

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 76)
    print("O-RAP EXPERIMENTS COMPLETED")
    print("=" * 76)

    print("\nTEST SET RESULTS")
    print("-" * 76)

    print(
        f"{'MODEL':38s} {'PR-AUC':>10s} "
        f"{'ROC-AUC':>10s} {'F1':>10s} {'BRIER':>10s}"
    )

    for row in results:

        if row["partition"] != "test":
            continue

        print(
            f"{row['model']:38s} "
            f"{row['pr_auc']:10.5f} "
            f"{row['roc_auc']:10.5f} "
            f"{row['f1']:10.5f} "
            f"{row['brier_score']:10.5f}"
        )

    print("\nOUTPUT FILES")
    print(metrics_file)
    print(prediction_file)
    print(importance_file)
    print(MODEL_DIR)

    print("\nSUCCESS: All experiments and outputs completed.")


if __name__ == "__main__":
    main()
