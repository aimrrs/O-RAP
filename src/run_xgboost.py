from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
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

# Deliberately exclude satellite identity.
# This prevents the first experiment from simply learning
# "ISS is risky" or "GPS is safe".
EXCLUDED_FEATURES = {
    "norad_id",
    "satellite_name",
    TARGET,
}


# ============================================================
# Load data
# ============================================================

print("=" * 70)
print("O-RAP XGBOOST EXPERIMENT")
print("=" * 70)

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Train samples:      {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples:       {len(test_df)}")


# ============================================================
# Prepare features
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

print("\nFeatures used:")
for feature in feature_columns:
    print(f"  - {feature}")

print(f"\nNumber of features: {len(feature_columns)}")


# ============================================================
# Sanity checks
# ============================================================

if list(X_train.columns) != list(X_val.columns):
    raise ValueError("Train and validation feature columns do not match.")

if list(X_train.columns) != list(X_test.columns):
    raise ValueError("Train and test feature columns do not match.")

if X_train.isna().any().any():
    raise ValueError("Missing values detected in training features.")

if X_val.isna().any().any():
    raise ValueError("Missing values detected in validation features.")

if X_test.isna().any().any():
    raise ValueError("Missing values detected in test features.")

if not set(y_train.unique()).issubset({0, 1}):
    raise ValueError("Training target is not binary.")


# ============================================================
# Class imbalance
# ============================================================

negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())

scale_pos_weight = negative_count / positive_count

print("\nTraining class distribution:")
print(f"  Reliable:   {negative_count}")
print(f"  Unreliable: {positive_count}")
print(f"  Positive rate: {positive_count / len(y_train):.4f}")
print(f"  scale_pos_weight: {scale_pos_weight:.4f}")


# ============================================================
# XGBoost model
# ============================================================

model = XGBClassifier(
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


# ============================================================
# Train
# ============================================================

print("\nTraining XGBoost...")

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

print("Training complete.")


# ============================================================
# Probability predictions
# ============================================================

train_prob = model.predict_proba(X_train)[:, 1]
val_prob = model.predict_proba(X_val)[:, 1]
test_prob = model.predict_proba(X_test)[:, 1]


# ============================================================
# Threshold selection
# ============================================================
# IMPORTANT:
# We choose the classification threshold using ONLY training data.
# Validation and test remain untouched for threshold selection.

thresholds = np.linspace(0.05, 0.95, 181)

best_threshold = 0.5
best_train_f1 = -1.0

for threshold in thresholds:
    train_pred = (train_prob >= threshold).astype(int)

    score = f1_score(
        y_train,
        train_pred,
        zero_division=0,
    )

    if score > best_train_f1:
        best_train_f1 = score
        best_threshold = float(threshold)


# ============================================================
# Evaluation function
# ============================================================

def evaluate_split(name, y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "split": name,
        "threshold": threshold,
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
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),
        "brier": brier_score_loss(
            y_true,
            probabilities,
        ),
        "positive_rate": float(y_true.mean()),
        "predicted_positive_rate": float(predictions.mean()),
    }

    print(f"\n{name}")
    print("-" * 70)
    print(f"Threshold:          {threshold:.4f}")
    print(f"Accuracy:           {metrics['accuracy']:.4f}")
    print(f"Precision:          {metrics['precision']:.4f}")
    print(f"Recall:             {metrics['recall']:.4f}")
    print(f"F1:                 {metrics['f1']:.4f}")
    print(f"ROC-AUC:            {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:             {metrics['pr_auc']:.4f}")
    print(f"Brier score:        {metrics['brier']:.4f}")
    print(f"Actual positive:    {metrics['positive_rate']:.4f}")
    print(f"Predicted positive: {metrics['predicted_positive_rate']:.4f}")

    return metrics


# ============================================================
# Evaluate
# ============================================================

print("\nBest threshold selected on TRAINING data:")
print(f"  Threshold: {best_threshold:.4f}")
print(f"  Training F1: {best_train_f1:.4f}")

results = []

results.append(
    evaluate_split(
        "Train",
        y_train,
        train_prob,
        best_threshold,
    )
)

results.append(
    evaluate_split(
        "Validation",
        y_val,
        val_prob,
        best_threshold,
    )
)

results.append(
    evaluate_split(
        "Test",
        y_test,
        test_prob,
        best_threshold,
    )
)


# ============================================================
# Feature importance
# ============================================================

importance = pd.DataFrame(
    {
        "feature": feature_columns,
        "importance": model.feature_importances_,
    }
).sort_values(
    "importance",
    ascending=False,
)

print("\nFeature importance:")
print(importance.to_string(index=False))


# ============================================================
# Save results
# ============================================================

results_df = pd.DataFrame(results)

results_file = RESULTS_DIR / "xgboost_temporal_results.csv"
importance_file = RESULTS_DIR / "xgboost_feature_importance.csv"

results_df.to_csv(results_file, index=False)
importance.to_csv(importance_file, index=False)

print("\nSaved:")
print(f"  {results_file}")
print(f"  {importance_file}")

print("\n" + "=" * 70)
print("XGBOOST EXPERIMENT COMPLETE")
print("=" * 70)