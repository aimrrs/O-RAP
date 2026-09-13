from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
SPLIT_DIR = PROCESSED_DIR / "splits"
RESULTS_DIR = BASE_DIR / "results" / "tables"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = SPLIT_DIR / "train_temporal.csv"
VALIDATION_FILE = SPLIT_DIR / "validation_temporal.csv"
TEST_FILE = SPLIT_DIR / "test_temporal.csv"

XGB_FILE = (
    RESULTS_DIR
    / "xgboost_leave_one_satellite_out_results.csv"
)

HORIZON_FILE = (
    RESULTS_DIR
    / "horizon_unseen_satellite_results.csv"
)

COMPARISON_FILE = (
    RESULTS_DIR
    / "unseen_satellite_xgboost_vs_horizon.csv"
)


# ============================================================
# Configuration
# ============================================================

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
# Metrics
# ============================================================

def safe_roc_auc(y_true, scores):
    """
    ROC-AUC is undefined when only one class is present.
    """

    if pd.Series(y_true).nunique() < 2:
        return np.nan

    return roc_auc_score(
        y_true,
        scores,
    )


def safe_pr_auc(y_true, scores):
    """
    PR-AUC is undefined when only one class is present.
    """

    if pd.Series(y_true).nunique() < 2:
        return np.nan

    return average_precision_score(
        y_true,
        scores,
    )


def find_best_threshold(
    y_true,
    scores,
):
    """
    Select the threshold using TRAINING DATA ONLY.

    Threshold maximizes F1.
    """

    unique_scores = np.unique(
        np.asarray(scores, dtype=float)
    )

    best_threshold = None
    best_f1 = -1.0

    for threshold in unique_scores:

        predictions = (
            np.asarray(scores) >= threshold
        ).astype(int)

        current_f1 = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        if current_f1 > best_f1:

            best_f1 = current_f1
            best_threshold = threshold

    return (
        float(best_threshold),
        float(best_f1),
    )


# ============================================================
# Load data
# ============================================================

print("=" * 70)
print("O-RAP — UNSEEN-SATELLITE BASELINE COMPARISON")
print("=" * 70)

train = pd.read_csv(
    TRAIN_FILE
)

validation = pd.read_csv(
    VALIDATION_FILE
)

test = pd.read_csv(
    TEST_FILE
)

print(
    f"Train:      {len(train)}"
)

print(
    f"Validation: {len(validation)}"
)

print(
    f"Test:       {len(test)}"
)

print()

print("Held-out satellites:")

for satellite in SATELLITES:
    print(
        f"  - {satellite}"
    )


# ============================================================
# Horizon score
# ============================================================

def horizon_score(df):
    """
    Longer prediction horizons correspond to greater risk.

    This is a RANKING SCORE, not a calibrated probability.
    """

    return (
        df["horizon_hours"]
        .astype(float)
    )


# ============================================================
# Leave-one-satellite-out experiment
# ============================================================

results = []


for held_out in SATELLITES:

    print()
    print()
    print("=" * 70)
    print(
        f"HOLDING OUT: {held_out}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Exclude held-out satellite from training/validation.
    # Keep ONLY held-out satellite in test.
    # --------------------------------------------------------

    train_subset = train.loc[
        train["satellite_name"] != held_out
    ].copy()

    validation_subset = validation.loc[
        validation["satellite_name"] != held_out
    ].copy()

    test_subset = test.loc[
        test["satellite_name"] == held_out
    ].copy()

    print(
        f"Training samples:   "
        f"{len(train_subset)}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_subset)}"
    )

    print(
        f"Test samples:       "
        f"{len(test_subset)}"
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    y_train = (
        train_subset[TARGET]
        .astype(int)
        .to_numpy()
    )

    y_test = (
        test_subset[TARGET]
        .astype(int)
        .to_numpy()
    )

    # --------------------------------------------------------
    # Horizon scores
    # --------------------------------------------------------

    train_score = horizon_score(
        train_subset
    ).to_numpy()

    test_score = horizon_score(
        test_subset
    ).to_numpy()

    # --------------------------------------------------------
    # Training threshold
    # --------------------------------------------------------

    threshold, train_f1 = find_best_threshold(
        y_train,
        train_score,
    )

    print()
    print(
        f"Training threshold: "
        f"{threshold:.4f}"
    )

    print(
        f"Training F1:        "
        f"{train_f1:.4f}"
    )

    # --------------------------------------------------------
    # Test classification
    # --------------------------------------------------------

    test_predictions = (
        test_score >= threshold
    ).astype(int)

    test_accuracy = accuracy_score(
        y_test,
        test_predictions,
    )

    test_precision = precision_score(
        y_test,
        test_predictions,
        zero_division=0,
    )

    test_recall = recall_score(
        y_test,
        test_predictions,
        zero_division=0,
    )

    test_f1 = f1_score(
        y_test,
        test_predictions,
        zero_division=0,
    )

    # --------------------------------------------------------
    # Ranking score
    #
    # Normalize only for numerical probability-style metrics.
    #
    # IMPORTANT:
    # This is NOT a calibrated probability.
    # --------------------------------------------------------

    normalized_score = (
        test_score / 48.0
    )

    normalized_score = np.clip(
        normalized_score,
        1e-7,
        1 - 1e-7,
    )

    test_brier = brier_score_loss(
        y_test,
        normalized_score,
    )

    test_logloss = log_loss(
        y_test,
        normalized_score,
        labels=[0, 1],
    )

    test_roc_auc = safe_roc_auc(
        y_test,
        test_score,
    )

    test_pr_auc = safe_pr_auc(
        y_test,
        test_score,
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print()
    print("HORIZON-ONLY TEST")

    print(
        f"  Positive rate: "
        f"{y_test.mean():.4f}"
    )

    print(
        f"  Accuracy:  "
        f"{test_accuracy:.4f}"
    )

    print(
        f"  Precision: "
        f"{test_precision:.4f}"
    )

    print(
        f"  Recall:    "
        f"{test_recall:.4f}"
    )

    print(
        f"  F1:        "
        f"{test_f1:.4f}"
    )

    print(
        f"  Brier:     "
        f"{test_brier:.4f}"
    )

    print(
        f"  LogLoss:   "
        f"{test_logloss:.4f}"
    )

    if np.isnan(test_roc_auc):
        print(
            "  ROC-AUC:   undefined"
        )
    else:
        print(
            f"  ROC-AUC:   "
            f"{test_roc_auc:.4f}"
        )

    if np.isnan(test_pr_auc):
        print(
            "  PR-AUC:    undefined"
        )
    else:
        print(
            f"  PR-AUC:    "
            f"{test_pr_auc:.4f}"
        )

    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    results.append(
        {
            "held_out_satellite": held_out,
            "train_samples": len(train_subset),
            "validation_samples": len(validation_subset),
            "test_samples": len(test_subset),
            "test_positive_rate": y_test.mean(),
            "training_threshold": threshold,
            "training_f1": train_f1,
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1,
            "test_brier": test_brier,
            "test_logloss": test_logloss,
            "test_roc_auc": test_roc_auc,
            "test_pr_auc": test_pr_auc,
        }
    )


# ============================================================
# Horizon results DataFrame
# ============================================================

horizon_results = pd.DataFrame(
    results
)


# ============================================================
# Save horizon results
# ============================================================

horizon_results.to_csv(
    HORIZON_FILE,
    index=False,
)

print()
print()
print("=" * 70)
print("HORIZON BASELINE COMPLETE")
print("=" * 70)

print()
print("Saved:")
print(
    f"  {HORIZON_FILE}"
)


# ============================================================
# XGBoost comparison
# ============================================================

if not XGB_FILE.exists():

    print()
    print()
    print("=" * 70)
    print("XGBOOST RESULTS NOT FOUND")
    print("=" * 70)

    print()
    print("Expected:")
    print(
        f"  {XGB_FILE}"
    )

else:

    print()
    print()
    print("=" * 70)
    print("XGBOOST vs HORIZON-ONLY")
    print("=" * 70)

    # --------------------------------------------------------
    # Load existing XGBoost results.
    #
    # IMPORTANT:
    # We do NOT merge the original DataFrames with pandas
    # suffixes. We construct the comparison explicitly.
    # --------------------------------------------------------

    xgb_results = pd.read_csv(
        XGB_FILE
    )

    comparison_rows = []

    for satellite in SATELLITES:

        # Find horizon result.
        horizon_row = horizon_results.loc[
            horizon_results["held_out_satellite"]
            == satellite
        ]

        # Find XGBoost result.
        xgb_row = xgb_results.loc[
            xgb_results["held_out_satellite"]
            == satellite
        ]

        if horizon_row.empty:

            print(
                f"\nWARNING: Missing horizon result "
                f"for {satellite}"
            )

            continue

        if xgb_row.empty:

            print(
                f"\nWARNING: Missing XGBoost result "
                f"for {satellite}"
            )

            continue

        h = horizon_row.iloc[0]
        x = xgb_row.iloc[0]

        comparison_rows.append(
            {
                "held_out_satellite": satellite,

                # Horizon
                "horizon_f1": h[
                    "test_f1"
                ],

                "horizon_roc_auc": h[
                    "test_roc_auc"
                ],

                "horizon_pr_auc": h[
                    "test_pr_auc"
                ],

                "horizon_brier": h[
                    "test_brier"
                ],

                "horizon_logloss": h[
                    "test_logloss"
                ],

                # XGBoost
                "xgboost_roc_auc": x[
                    "raw_roc_auc"
                ],

                "xgboost_pr_auc": x[
                    "raw_pr_auc"
                ],

                "xgboost_brier": x[
                    "raw_brier"
                ],

                "xgboost_logloss": x[
                    "raw_logloss"
                ],
            }
        )

    # --------------------------------------------------------
    # Build comparison DataFrame.
    # --------------------------------------------------------

    comparison = pd.DataFrame(
        comparison_rows
    )

    # --------------------------------------------------------
    # Calculate gains.
    # --------------------------------------------------------

    comparison["roc_auc_gain"] = (
        comparison["xgboost_roc_auc"]
        - comparison["horizon_roc_auc"]
    )

    comparison["pr_auc_gain"] = (
        comparison["xgboost_pr_auc"]
        - comparison["horizon_pr_auc"]
    )

    comparison["brier_change"] = (
        comparison["xgboost_brier"]
        - comparison["horizon_brier"]
    )

    comparison["logloss_change"] = (
        comparison["xgboost_logloss"]
        - comparison["horizon_logloss"]
    )

    # --------------------------------------------------------
    # Save comparison.
    # --------------------------------------------------------

    comparison.to_csv(
        COMPARISON_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Display table.
    # --------------------------------------------------------

    print()

    display_columns = [
        "held_out_satellite",
        "horizon_f1",
        "horizon_roc_auc",
        "xgboost_roc_auc",
        "horizon_pr_auc",
        "xgboost_pr_auc",
        "horizon_brier",
        "xgboost_brier",
    ]

    print(
        comparison[
            display_columns
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Mean ranking metrics.
    #
    # pandas mean() ignores NaN values automatically.
    # --------------------------------------------------------

    print()
    print("Mean comparison:")

    print(
        f"  Horizon F1:       "
        f"{comparison['horizon_f1'].mean():.4f}"
    )

    print(
        f"  Horizon ROC-AUC:  "
        f"{comparison['horizon_roc_auc'].mean():.4f}"
    )

    print(
        f"  XGBoost ROC-AUC:  "
        f"{comparison['xgboost_roc_auc'].mean():.4f}"
    )

    print(
        f"  Horizon PR-AUC:   "
        f"{comparison['horizon_pr_auc'].mean():.4f}"
    )

    print(
        f"  XGBoost PR-AUC:   "
        f"{comparison['xgboost_pr_auc'].mean():.4f}"
    )

    print(
        f"  Horizon Brier:    "
        f"{comparison['horizon_brier'].mean():.4f}"
    )

    print(
        f"  XGBoost Brier:    "
        f"{comparison['xgboost_brier'].mean():.4f}"
    )

    print()
    print("Interpretation:")

    print(
        "  Positive ROC-AUC gain = XGBoost "
        "ranks unseen-satellite risk better."
    )

    print(
        "  Positive PR-AUC gain = XGBoost "
        "improves precision-recall ranking."
    )

    print(
        "  Negative Brier change = XGBoost "
        "has lower Brier score."
    )

    print(
        "  Negative LogLoss change = XGBoost "
        "has lower LogLoss."
    )

    print()
    print("Saved:")

    print(
        f"  {HORIZON_FILE}"
    )

    print(
        f"  {COMPARISON_FILE}"
    )


# ============================================================
# Complete
# ============================================================

print()
print()
print("=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)