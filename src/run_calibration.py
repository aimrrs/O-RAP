
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

SPLIT_DIR = BASE_DIR / "data" / "processed" / "splits"
RESULTS_DIR = BASE_DIR / "results" / "tables"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = SPLIT_DIR / "train_temporal.csv"
VAL_FILE = SPLIT_DIR / "validation_temporal.csv"
TEST_FILE = SPLIT_DIR / "test_temporal.csv"

TARGET = "unreliable_1km"
HELD_OUT_SATELLITE = "ISS"

EXCLUDED_FEATURES = {
    "norad_id",
    "satellite_name",
    TARGET,
}


# ============================================================
# Load data
# ============================================================

print("=" * 70)
print("O-RAP PROBABILITY CALIBRATION EXPERIMENT")
print("=" * 70)

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Train samples:      {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples:       {len(test_df)}")


# ============================================================
# Feature preparation
# ============================================================

feature_columns = [
    column
    for column in train_df.columns
    if column not in EXCLUDED_FEATURES
]

X_train = train_df[feature_columns].copy()
y_train = train_df[TARGET].astype(int)

X_val = val_df[feature_columns].copy()
y_val = val_df[TARGET].astype(int)

X_test = test_df[feature_columns].copy()
y_test = test_df[TARGET].astype(int)

# Unseen ISS subset from temporal test set.
iss_mask = test_df["satellite_name"] == HELD_OUT_SATELLITE

X_test_iss = test_df.loc[iss_mask, feature_columns].copy()
y_test_iss = test_df.loc[iss_mask, TARGET].astype(int)


# ============================================================
# Sanity checks
# ============================================================

if X_train.isna().any().any():
    raise ValueError("Missing values detected in training features.")

if X_val.isna().any().any():
    raise ValueError("Missing values detected in validation features.")

if X_test.isna().any().any():
    raise ValueError("Missing values detected in test features.")

if len(X_test_iss) == 0:
    raise ValueError("No ISS samples found in temporal test set.")


# ============================================================
# Class imbalance
# ============================================================

negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())

scale_pos_weight = negative_count / positive_count

print("\nFeatures used:")
for feature in feature_columns:
    print(f"  - {feature}")

print(f"\nNumber of features: {len(feature_columns)}")

print("\nTraining class distribution:")
print(f"  Reliable:         {negative_count}")
print(f"  Unreliable:       {positive_count}")
print(f"  Positive rate:    {y_train.mean():.4f}")
print(f"  scale_pos_weight: {scale_pos_weight:.4f}")


# ============================================================
# Train base XGBoost model
# ============================================================

print("\nTraining base XGBoost model...")

base_model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
)

base_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

print("Base model training complete.")


# ============================================================
# Raw probabilities
# ============================================================

raw_train_prob = base_model.predict_proba(X_train)[:, 1]
raw_val_prob = base_model.predict_proba(X_val)[:, 1]
raw_test_prob = base_model.predict_proba(X_test)[:, 1]
raw_iss_prob = base_model.predict_proba(X_test_iss)[:, 1]


# ============================================================
# Platt / sigmoid calibration
# ============================================================
#
# We explicitly fit a logistic calibration model:
#
#     calibrated_probability =
#         sigmoid(a * logit(raw_probability) + b)
#
# The calibration model is fitted ONLY on validation data.
#
# The XGBoost model itself was fitted ONLY on training data.
#
# The temporal test and unseen-ISS test remain untouched.
# ============================================================

print("\nFitting sigmoid calibration using VALIDATION ONLY...")

# Protect against probabilities exactly equal to 0 or 1.
EPSILON = 1e-7

def probability_to_logit(probability):
    probability = np.clip(
        probability,
        EPSILON,
        1.0 - EPSILON,
    )
    return np.log(
        probability / (1.0 - probability)
    )


val_logit = probability_to_logit(raw_val_prob)
train_logit = probability_to_logit(raw_train_prob)
test_logit = probability_to_logit(raw_test_prob)
iss_logit = probability_to_logit(raw_iss_prob)

calibration_model = LogisticRegression(
    C=1e6,
    solver="lbfgs",
    max_iter=1000,
)

calibration_model.fit(
    val_logit.reshape(-1, 1),
    y_val,
)

print("Calibration complete.")

print(
    f"  Calibration coefficient: "
    f"{calibration_model.coef_[0][0]:.6f}"
)

print(
    f"  Calibration intercept:   "
    f"{calibration_model.intercept_[0]:.6f}"
)


# ============================================================
# Generate calibrated probabilities
# ============================================================

cal_train_prob = calibration_model.predict_proba(
    train_logit.reshape(-1, 1)
)[:, 1]

cal_val_prob = calibration_model.predict_proba(
    val_logit.reshape(-1, 1)
)[:, 1]

cal_test_prob = calibration_model.predict_proba(
    test_logit.reshape(-1, 1)
)[:, 1]

cal_iss_prob = calibration_model.predict_proba(
    iss_logit.reshape(-1, 1)
)[:, 1]


# ============================================================
# Probability evaluation
# ============================================================

def evaluate_probabilities(
    name,
    y_true,
    raw_prob,
    calibrated_prob,
):
    raw_brier = brier_score_loss(y_true, raw_prob)
    calibrated_brier = brier_score_loss(
        y_true,
        calibrated_prob,
    )

    raw_logloss = log_loss(y_true, raw_prob)
    calibrated_logloss = log_loss(
        y_true,
        calibrated_prob,
    )

    raw_roc_auc = roc_auc_score(y_true, raw_prob)
    calibrated_roc_auc = roc_auc_score(
        y_true,
        calibrated_prob,
    )

    raw_pr_auc = average_precision_score(
        y_true,
        raw_prob,
    )

    calibrated_pr_auc = average_precision_score(
        y_true,
        calibrated_prob,
    )

    print(f"\n{name}")
    print("-" * 70)

    print(f"Samples: {len(y_true)}")
    print(f"Actual positive rate: {y_true.mean():.4f}")

    print("\nRAW XGBoost probabilities")
    print(f"  Brier score: {raw_brier:.4f}")
    print(f"  Log loss:    {raw_logloss:.4f}")
    print(f"  ROC-AUC:     {raw_roc_auc:.4f}")
    print(f"  PR-AUC:      {raw_pr_auc:.4f}")

    print("\nCALIBRATED probabilities")
    print(f"  Brier score: {calibrated_brier:.4f}")
    print(f"  Log loss:    {calibrated_logloss:.4f}")
    print(f"  ROC-AUC:     {calibrated_roc_auc:.4f}")
    print(f"  PR-AUC:      {calibrated_pr_auc:.4f}")

    print("\nChange from raw -> calibrated")
    print(
        f"  Brier:    "
        f"{calibrated_brier - raw_brier:+.4f}"
    )
    print(
        f"  Log loss: "
        f"{calibrated_logloss - raw_logloss:+.4f}"
    )

    return {
        "dataset": name,
        "samples": len(y_true),
        "positive_rate": float(y_true.mean()),
        "raw_brier": raw_brier,
        "calibrated_brier": calibrated_brier,
        "raw_logloss": raw_logloss,
        "calibrated_logloss": calibrated_logloss,
        "raw_roc_auc": raw_roc_auc,
        "calibrated_roc_auc": calibrated_roc_auc,
        "raw_pr_auc": raw_pr_auc,
        "calibrated_pr_auc": calibrated_pr_auc,
    }


# ============================================================
# Evaluate validation, temporal test and unseen ISS
# ============================================================

results = []

results.append(
    evaluate_probabilities(
        "Validation",
        y_val,
        raw_val_prob,
        cal_val_prob,
    )
)

results.append(
    evaluate_probabilities(
        "Temporal Test",
        y_test,
        raw_test_prob,
        cal_test_prob,
    )
)

results.append(
    evaluate_probabilities(
        "Unseen ISS Test",
        y_test_iss,
        raw_iss_prob,
        cal_iss_prob,
    )
)


# ============================================================
# Reliability / calibration table
# ============================================================

def make_calibration_table(
    y_true,
    probabilities,
    dataset_name,
):
    calibration_df = pd.DataFrame(
        {
            "actual": y_true.to_numpy(),
            "probability": probabilities,
        }
    )

    bins = np.linspace(0.0, 1.0, 11)

    calibration_df["bin"] = pd.cut(
        calibration_df["probability"],
        bins=bins,
        include_lowest=True,
        labels=False,
    )

    grouped = (
        calibration_df
        .groupby("bin", observed=False)
        .agg(
            mean_predicted_probability=(
                "probability",
                "mean",
            ),
            observed_positive_rate=(
                "actual",
                "mean",
            ),
            sample_count=(
                "actual",
                "size",
            ),
        )
        .reset_index()
    )

    grouped["dataset"] = dataset_name

    return grouped[
        [
            "dataset",
            "bin",
            "mean_predicted_probability",
            "observed_positive_rate",
            "sample_count",
        ]
    ]


calibration_tables = []

calibration_tables.append(
    make_calibration_table(
        y_val,
        cal_val_prob,
        "Validation",
    )
)

calibration_tables.append(
    make_calibration_table(
        y_test,
        cal_test_prob,
        "Temporal Test",
    )
)

calibration_tables.append(
    make_calibration_table(
        y_test_iss,
        cal_iss_prob,
        "Unseen ISS Test",
    )
)

calibration_table = pd.concat(
    calibration_tables,
    ignore_index=True,
)


# ============================================================
# Save results
# ============================================================

summary = pd.DataFrame(results)

results_file = (
    RESULTS_DIR / "xgboost_calibration_results.csv"
)

calibration_file = (
    RESULTS_DIR / "xgboost_calibration_table.csv"
)

summary.to_csv(
    results_file,
    index=False,
)

calibration_table.to_csv(
    calibration_file,
    index=False,
)


# ============================================================
# Final output
# ============================================================

print("\nCalibration table:")
print(
    calibration_table.to_string(
        index=False
    )
)

print("\nSaved:")
print(f"  {results_file}")
print(f"  {calibration_file}")

print("\n" + "=" * 70)
print("PROBABILITY CALIBRATION EXPERIMENT COMPLETE")
print("=" * 70)
