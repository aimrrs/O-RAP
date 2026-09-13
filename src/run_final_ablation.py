from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS / CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "processed" / "splits"
RESULTS_DIR = BASE_DIR / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "unreliable_1km"

SATELLITES = [
    "GPS BIIR-2",
    "Hubble Space Telescope",
    "ISS",
    "NOAA 19",
    "Sentinel-1A",
    "TDRS-5",
]

# The globally selected configuration from the previous experiment.
XGB_CONFIG = {
    "n_estimators": 500,
    "max_depth": 3,
    "learning_rate": 0.03,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.5,
    "reg_lambda": 5.0,
}

FEATURE_SETS = {
    "horizon_only": [
        "horizon_hours",
    ],

    "horizon_age": [
        "horizon_hours",
        "source_tle_age_hours",
    ],

    "full_orbital": [
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
    ],
}


# ============================================================
# LOAD
# ============================================================

print("\n" + "=" * 70)
print("O-RAP — FINAL GLOBAL ABLATION")
print("=" * 70)

train = pd.read_csv(DATA_DIR / "train_temporal.csv")
validation = pd.read_csv(DATA_DIR / "validation_temporal.csv")
test = pd.read_csv(DATA_DIR / "test_temporal.csv")

print(f"Train:      {len(train)}")
print(f"Validation: {len(validation)}")
print(f"Test:       {len(test)}")


# ============================================================
# HELPERS
# ============================================================

def train_xgb(train_df, features):
    X = train_df[features]
    y = train_df[TARGET]

    positives = int(y.sum())
    negatives = len(y) - positives

    if positives == 0:
        raise ValueError("No positive samples in training data.")

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=negatives / positives,
        **XGB_CONFIG,
    )

    model.fit(X, y)

    return model


def predict_xgb(model, df, features):
    return model.predict_proba(df[features])[:, 1]


def train_logistic(train_df, features):
    X = train_df[features].to_numpy()
    y = train_df[TARGET].to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    positives = int(y.sum())
    negatives = len(y) - positives

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X_scaled, y)

    return model, scaler


def predict_logistic(model, scaler, df, features):
    X = df[features].to_numpy()
    X_scaled = scaler.transform(X)

    return model.predict_proba(X_scaled)[:, 1]


def find_threshold(y, probabilities):
    thresholds = np.linspace(0.01, 0.99, 197)

    best_threshold = 0.5
    best_f1 = -1

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        score = f1_score(
            y,
            predictions,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return best_threshold, best_f1


def metrics(y, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    result = {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(
            y, predictions, zero_division=0
        ),
        "recall": recall_score(
            y, predictions, zero_division=0
        ),
        "f1": f1_score(
            y, predictions, zero_division=0
        ),
        "brier": brier_score_loss(
            y, probabilities
        ),
        "logloss": log_loss(
            y, probabilities, labels=[0, 1]
        ),
    }

    if len(np.unique(y)) == 2:
        result["roc_auc"] = roc_auc_score(
            y, probabilities
        )
        result["pr_auc"] = average_precision_score(
            y, probabilities
        )
    else:
        result["roc_auc"] = np.nan
        result["pr_auc"] = np.nan

    return result


def calibrate(y, probabilities):
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

    calibrator.fit(logits, y)

    return calibrator


def apply_calibration(calibrator, probabilities):
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
# EXPERIMENT
# ============================================================

all_results = []

for model_name, features in FEATURE_SETS.items():

    print("\n" + "=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)

    fold_results = []

    for held_out in SATELLITES:

        print(f"\nHeld-out: {held_out}")

        fold_train = train[
            train["satellite_name"] != held_out
        ].copy()

        fold_validation = validation[
            validation["satellite_name"] != held_out
        ].copy()

        fold_test = test[
            test["satellite_name"] == held_out
        ].copy()

        # ----------------------------------------------------
        # Train model
        # ----------------------------------------------------

        model = train_xgb(
            fold_train,
            features,
        )

        validation_prob = predict_xgb(
            model,
            fold_validation,
            features,
        )

        test_prob = predict_xgb(
            model,
            fold_test,
            features,
        )

        validation_y = fold_validation[TARGET].to_numpy()
        test_y = fold_test[TARGET].to_numpy()

        # ----------------------------------------------------
        # Calibration on validation ONLY
        # ----------------------------------------------------

        calibrator = calibrate(
            validation_y,
            validation_prob,
        )

        calibrated_validation_prob = apply_calibration(
            calibrator,
            validation_prob,
        )

        calibrated_test_prob = apply_calibration(
            calibrator,
            test_prob,
        )

        # ----------------------------------------------------
        # Threshold from validation ONLY
        # ----------------------------------------------------

        threshold, validation_f1 = find_threshold(
            validation_y,
            calibrated_validation_prob,
        )

        # ----------------------------------------------------
        # Test metrics
        # ----------------------------------------------------

        raw = metrics(
            test_y,
            test_prob,
            threshold,
        )

        calibrated = metrics(
            test_y,
            calibrated_test_prob,
            threshold,
        )

        print(
            f"  ROC-AUC: "
            f"{raw['roc_auc']:.4f}"
        )

        print(
            f"  PR-AUC:  "
            f"{raw['pr_auc']:.4f}"
        )

        print(
            f"  Brier:   "
            f"{raw['brier']:.4f} -> "
            f"{calibrated['brier']:.4f}"
        )

        print(
            f"  LogLoss: "
            f"{raw['logloss']:.4f} -> "
            f"{calibrated['logloss']:.4f}"
        )

        print(
            f"  F1:      "
            f"{raw['f1']:.4f} -> "
            f"{calibrated['f1']:.4f}"
        )

        row = {
            "model": model_name,
            "held_out_satellite": held_out,
            "threshold": threshold,
            "validation_f1": validation_f1,

            "raw_roc_auc": raw["roc_auc"],
            "raw_pr_auc": raw["pr_auc"],
            "raw_brier": raw["brier"],
            "raw_logloss": raw["logloss"],
            "raw_f1": raw["f1"],

            "calibrated_roc_auc": calibrated["roc_auc"],
            "calibrated_pr_auc": calibrated["pr_auc"],
            "calibrated_brier": calibrated["brier"],
            "calibrated_logloss": calibrated["logloss"],
            "calibrated_f1": calibrated["f1"],
        }

        fold_results.append(row)
        all_results.append(row)

    # --------------------------------------------------------
    # Model-level summary
    # --------------------------------------------------------

    fold_df = pd.DataFrame(fold_results)

    print("\nMODEL MEAN:")

    print(
        f"  Raw ROC-AUC: "
        f"{fold_df['raw_roc_auc'].mean(skipna=True):.4f}"
    )

    print(
        f"  Raw PR-AUC:  "
        f"{fold_df['raw_pr_auc'].mean(skipna=True):.4f}"
    )

    print(
        f"  Calibrated Brier: "
        f"{fold_df['calibrated_brier'].mean(skipna=True):.4f}"
    )

    print(
        f"  Calibrated LogLoss: "
        f"{fold_df['calibrated_logloss'].mean(skipna=True):.4f}"
    )

    print(
        f"  Calibrated F1: "
        f"{fold_df['calibrated_f1'].mean(skipna=True):.4f}"
    )


# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(all_results)

output_path = (
    RESULTS_DIR /
    "final_ablation_results.csv"
)

results_df.to_csv(
    output_path,
    index=False,
)


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

summary = (
    results_df
    .groupby("model")
    .agg(
        mean_raw_roc_auc=("raw_roc_auc", "mean"),
        mean_raw_pr_auc=("raw_pr_auc", "mean"),
        mean_calibrated_brier=("calibrated_brier", "mean"),
        mean_calibrated_logloss=("calibrated_logloss", "mean"),
        mean_calibrated_f1=("calibrated_f1", "mean"),
    )
    .reset_index()
)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\nPer-satellite results:")

print(
    results_df[
        [
            "model",
            "held_out_satellite",
            "raw_roc_auc",
            "raw_pr_auc",
            "calibrated_brier",
            "calibrated_logloss",
            "calibrated_f1",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print(f"\nSaved: {output_path}")
print("\n✓ Final ablation completed.")