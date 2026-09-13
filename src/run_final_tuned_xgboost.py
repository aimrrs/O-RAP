from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "processed" / "splits"
RESULTS_DIR = BASE_DIR / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "horizon_hours",
    "source_tle_age_hours",
    "mean_motion",
    "eccentricity",
    "inclination",
    "ra_of_asc_node",
    "arg_of_pericenter",
    "mean_anomaly",
    "bstar",
    "mean_motion_dot",
    "mean_motion_ddot",
    "semimajor_axis",
    "period",
    "apoapsis",
    "periapsis",
]

TARGET = "unreliable_1km"

SATELLITES = [
    "GPS BIIR-2",
    "Hubble Space Telescope",
    "ISS",
    "NOAA 19",
    "Sentinel-1A",
    "TDRS-5",
]


# ============================================================
# XGBOOST CONFIGURATIONS
# ============================================================

CONFIGS = {
    "baseline": {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },

    "shallow_regularized": {
        "n_estimators": 400,
        "max_depth": 3,
        "learning_rate": 0.05,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
    },

    "medium_regularized": {
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
    },

    "strong_regularization": {
        "n_estimators": 500,
        "max_depth": 3,
        "learning_rate": 0.03,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 5.0,
    },

    "low_depth": {
        "n_estimators": 500,
        "max_depth": 2,
        "learning_rate": 0.05,
        "min_child_weight": 3,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
    },

    "more_capacity": {
        "n_estimators": 400,
        "max_depth": 5,
        "learning_rate": 0.05,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
    },
}


# ============================================================
# HELPERS
# ============================================================

def load_split(name):
    path = DATA_DIR / f"{name}.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing split: {path}")

    df = pd.read_csv(path)

    print(f"{name}: {len(df)} samples")

    return df


def train_model(train_df, config):
    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)

    if positives == 0:
        raise ValueError("Training set contains no positive samples.")

    scale_pos_weight = negatives / positives

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        **config,
    )

    model.fit(X_train, y_train)

    return model


def get_predictions(model, df):
    return model.predict_proba(df[FEATURES])[:, 1]


def find_f1_threshold(y_true, probabilities):
    thresholds = np.linspace(0.01, 0.99, 197)

    best_threshold = 0.5
    best_f1 = -1

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return best_threshold, best_f1


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    result = {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "brier": brier_score_loss(
            y_true,
            probabilities,
        ),
        "logloss": log_loss(
            y_true,
            probabilities,
            labels=[0, 1],
        ),
    }

    if pd.Series(y_true).nunique() == 2:
        result["roc_auc"] = roc_auc_score(
            y_true,
            probabilities,
        )
        result["pr_auc"] = average_precision_score(
            y_true,
            probabilities,
        )
    else:
        result["roc_auc"] = np.nan
        result["pr_auc"] = np.nan

    return result


def fit_platt_calibrator(y_true, probabilities):
    """
    Platt/sigmoid calibration using validation data only.

    The XGBoost probability is converted to a logit.
    Logistic regression then learns:

        calibrated probability =
        sigmoid(a * logit(p) + b)
    """

    probabilities = np.clip(
        probabilities,
        1e-7,
        1 - 1e-7,
    )

    logits = np.log(
        probabilities / (1 - probabilities)
    ).reshape(-1, 1)

    calibrator = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=1000,
    )

    calibrator.fit(logits, y_true)

    return calibrator


def apply_calibrator(calibrator, probabilities):
    probabilities = np.clip(
        probabilities,
        1e-7,
        1 - 1e-7,
    )

    logits = np.log(
        probabilities / (1 - probabilities)
    ).reshape(-1, 1)

    return calibrator.predict_proba(logits)[:, 1]


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("O-RAP — FINAL GLOBAL-TUNING + CALIBRATION EXPERIMENT")
print("=" * 70)

train = load_split("train_temporal")
validation = load_split("validation_temporal")
test = load_split("test_temporal")

print("\nTemporal protocol:")
print("  Train      : January-August 2025")
print("  Validation : September-October 2025")
print("  Test       : November-December 2025")

print("\nHeld-out satellite protocol:")
for satellite in SATELLITES:
    print(f"  - {satellite}")


# ============================================================
# STEP 1
# GLOBAL CONFIGURATION SELECTION
#
# For every held-out satellite:
#
#   training = other 5 satellites, Jan-Aug
#   validation = other 5 satellites, Sep-Oct
#
# Importantly, the held-out satellite is NOT used for selecting
# the configuration.
#
# Validation predictions from all six folds are pooled.
# One configuration is then selected globally.
# ============================================================

print("\n" + "=" * 70)
print("STEP 1 — GLOBAL HYPERPARAMETER SELECTION")
print("=" * 70)

validation_records = []

for config_name, config in CONFIGS.items():

    print(f"\nTesting configuration: {config_name}")

    for held_out in SATELLITES:

        fold_train = train[
            train["satellite_name"] != held_out
        ].copy()

        fold_validation = validation[
            validation["satellite_name"] != held_out
        ].copy()

        model = train_model(
            fold_train,
            config,
        )

        probabilities = get_predictions(
            model,
            fold_validation,
        )

        validation_records.append({
            "configuration": config_name,
            "held_out_satellite": held_out,
            "y_true": fold_validation[TARGET].to_numpy(),
            "probabilities": probabilities,
        })

        metrics = calculate_metrics(
            fold_validation[TARGET].to_numpy(),
            probabilities,
            0.5,
        )

        print(
            f"  {held_out:25s} "
            f"PR-AUC={metrics['pr_auc']:.4f} "
            f"ROC-AUC={metrics['roc_auc']:.4f} "
            f"Brier={metrics['brier']:.4f}"
        )


# ============================================================
# AGGREGATE VALIDATION PERFORMANCE
# ============================================================

global_validation_results = []

for config_name in CONFIGS:

    rows = [
        r for r in validation_records
        if r["configuration"] == config_name
    ]

    y_all = np.concatenate(
        [r["y_true"] for r in rows]
    )

    p_all = np.concatenate(
        [r["probabilities"] for r in rows]
    )

    metrics = calculate_metrics(
        y_all,
        p_all,
        0.5,
    )

    global_validation_results.append({
        "configuration": config_name,
        "validation_pr_auc": metrics["pr_auc"],
        "validation_roc_auc": metrics["roc_auc"],
        "validation_brier": metrics["brier"],
        "validation_logloss": metrics["logloss"],
    })


global_validation_df = pd.DataFrame(
    global_validation_results
)

# Primary selection criterion:
#   highest pooled validation PR-AUC
#
# Tie breaker:
#   lowest Brier score

global_validation_df = global_validation_df.sort_values(
    by=[
        "validation_pr_auc",
        "validation_brier",
    ],
    ascending=[
        False,
        True,
    ],
).reset_index(drop=True)

best_configuration = global_validation_df.iloc[0]["configuration"]

print("\n" + "-" * 70)
print("GLOBAL CONFIGURATION RANKING")
print("-" * 70)

print(
    global_validation_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print(
    f"\nSelected global configuration: "
    f"{best_configuration}"
)

global_validation_df.to_csv(
    RESULTS_DIR / "final_global_xgboost_configuration_selection.csv",
    index=False,
)


# ============================================================
# STEP 2
# FINAL LEAVE-ONE-SATELLITE-OUT EVALUATION
#
# One configuration is now frozen.
#
# For every held-out satellite:
#
#   Train:
#       other 5 satellites, Jan-Aug
#
#   Calibration:
#       other 5 satellites, Sep-Oct
#
#   Test:
#       held-out satellite, Nov-Dec
#
# Calibration NEVER sees held-out satellite data.
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 — FINAL CALIBRATED UNSEEN-SATELLITE EVALUATION")
print("=" * 70)

final_config = CONFIGS[best_configuration]

final_results = []

for held_out in SATELLITES:

    print("\n" + "-" * 70)
    print(f"Held-out satellite: {held_out}")
    print("-" * 70)

    fold_train = train[
        train["satellite_name"] != held_out
    ].copy()

    fold_validation = validation[
        validation["satellite_name"] != held_out
    ].copy()

    fold_test = test[
        test["satellite_name"] == held_out
    ].copy()

    print(
        f"Train       : {len(fold_train)}"
    )

    print(
        f"Validation  : {len(fold_validation)}"
    )

    print(
        f"Test        : {len(fold_test)}"
    )

    print(
        f"Test positive rate: "
        f"{fold_test[TARGET].mean():.4f}"
    )

    # --------------------------------------------------------
    # Train final XGBoost model for this fold
    # --------------------------------------------------------

    model = train_model(
        fold_train,
        final_config,
    )

    # --------------------------------------------------------
    # Raw validation probabilities
    # --------------------------------------------------------

    validation_probabilities = get_predictions(
        model,
        fold_validation,
    )

    validation_y = fold_validation[TARGET].to_numpy()

    # --------------------------------------------------------
    # Fit calibration ONLY on validation
    # --------------------------------------------------------

    calibrator = fit_platt_calibrator(
        validation_y,
        validation_probabilities,
    )

    calibrated_validation_probabilities = apply_calibrator(
        calibrator,
        validation_probabilities,
    )

    # --------------------------------------------------------
    # Threshold selected ONLY from validation
    # --------------------------------------------------------

    threshold, validation_f1 = find_f1_threshold(
        validation_y,
        calibrated_validation_probabilities,
    )

    # --------------------------------------------------------
    # Held-out test probabilities
    # --------------------------------------------------------

    test_probabilities = get_predictions(
        model,
        fold_test,
    )

    test_y = fold_test[TARGET].to_numpy()

    calibrated_test_probabilities = apply_calibrator(
        calibrator,
        test_probabilities,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    raw_metrics = calculate_metrics(
        test_y,
        test_probabilities,
        threshold,
    )

    calibrated_metrics = calculate_metrics(
        test_y,
        calibrated_test_probabilities,
        threshold,
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"Calibration coefficient: "
        f"{calibrator.coef_[0][0]:.6f}"
    )

    print(
        f"Calibration intercept: "
        f"{calibrator.intercept_[0]:.6f}"
    )

    print(
        f"Validation calibrated F1: "
        f"{validation_f1:.4f}"
    )

    print(
        f"Test threshold: "
        f"{threshold:.3f}"
    )

    print("\nRaw XGBoost:")
    print(
        f"  ROC-AUC : {raw_metrics['roc_auc']:.4f}"
    )
    print(
        f"  PR-AUC  : {raw_metrics['pr_auc']:.4f}"
    )
    print(
        f"  Brier   : {raw_metrics['brier']:.4f}"
    )
    print(
        f"  LogLoss : {raw_metrics['logloss']:.4f}"
    )
    print(
        f"  F1      : {raw_metrics['f1']:.4f}"
    )

    print("\nCalibrated XGBoost:")
    print(
        f"  ROC-AUC : {calibrated_metrics['roc_auc']:.4f}"
    )
    print(
        f"  PR-AUC  : {calibrated_metrics['pr_auc']:.4f}"
    )
    print(
        f"  Brier   : {calibrated_metrics['brier']:.4f}"
    )
    print(
        f"  LogLoss : {calibrated_metrics['logloss']:.4f}"
    )
    print(
        f"  F1      : {calibrated_metrics['f1']:.4f}"
    )

    final_results.append({
        "held_out_satellite": held_out,
        "configuration": best_configuration,
        "threshold": threshold,

        "test_positive_rate": test_y.mean(),

        "raw_accuracy": raw_metrics["accuracy"],
        "raw_precision": raw_metrics["precision"],
        "raw_recall": raw_metrics["recall"],
        "raw_f1": raw_metrics["f1"],
        "raw_roc_auc": raw_metrics["roc_auc"],
        "raw_pr_auc": raw_metrics["pr_auc"],
        "raw_brier": raw_metrics["brier"],
        "raw_logloss": raw_metrics["logloss"],

        "calibrated_accuracy": calibrated_metrics["accuracy"],
        "calibrated_precision": calibrated_metrics["precision"],
        "calibrated_recall": calibrated_metrics["recall"],
        "calibrated_f1": calibrated_metrics["f1"],
        "calibrated_roc_auc": calibrated_metrics["roc_auc"],
        "calibrated_pr_auc": calibrated_metrics["pr_auc"],
        "calibrated_brier": calibrated_metrics["brier"],
        "calibrated_logloss": calibrated_metrics["logloss"],

        "calibration_coefficient": calibrator.coef_[0][0],
        "calibration_intercept": calibrator.intercept_[0],
    })


# ============================================================
# SAVE FINAL RESULTS
# ============================================================

final_results_df = pd.DataFrame(
    final_results
)

final_results_df.to_csv(
    RESULTS_DIR / "final_calibrated_tuned_xgboost_results.csv",
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(
    final_results_df[
        [
            "held_out_satellite",
            "configuration",
            "raw_roc_auc",
            "calibrated_roc_auc",
            "raw_pr_auc",
            "calibrated_pr_auc",
            "raw_brier",
            "calibrated_brier",
            "raw_logloss",
            "calibrated_logloss",
            "raw_f1",
            "calibrated_f1",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\nMean metrics across held-out satellites:")

for column in [
    "raw_roc_auc",
    "calibrated_roc_auc",
    "raw_pr_auc",
    "calibrated_pr_auc",
    "raw_brier",
    "calibrated_brier",
    "raw_logloss",
    "calibrated_logloss",
    "raw_f1",
    "calibrated_f1",
]:

    print(
        f"  {column:25s}: "
        f"{final_results_df[column].mean(skipna=True):.4f}"
    )

print("\nSaved:")
print(
    RESULTS_DIR
    / "final_global_xgboost_configuration_selection.csv"
)

print(
    RESULTS_DIR
    / "final_calibrated_tuned_xgboost_results.csv"
)

print("\n✓ Final experiment completed.")