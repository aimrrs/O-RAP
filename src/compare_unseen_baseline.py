from pathlib import Path

import numpy as np
import pandas as pd

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


# ============================================================
# Load data
# ============================================================

print("=" * 70)
print("O-RAP — UNSEEN-SATELLITE BASELINE COMPARISON")
print("=" * 70)

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Train:      {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Test:       {len(test_df)}")


# ============================================================
# Horizon-only baseline
#
# IMPORTANT:
# The threshold is learned ONLY from the non-held-out
# training satellites.
# ============================================================

def horizon_score(df):
    return df["horizon_hours"].astype(float)


def find_best_threshold(y_true, score):
    """
    Select threshold using training F1 only.
    """

    thresholds = np.sort(
        score.unique()
    )

    best_threshold = None
    best_f1 = -1.0

    for threshold in thresholds:

        prediction = (
            score >= threshold
        ).astype(int)

        tp = int(
            ((prediction == 1) & (y_true == 1)).sum()
        )

        fp = int(
            ((prediction == 1) & (y_true == 0)).sum()
        )

        fn = int(
            ((prediction == 0) & (y_true == 1)).sum()
        )

        if tp == 0:
            f1 = 0.0

        else:
            precision = (
                tp / (tp + fp)
            )

            recall = (
                tp / (tp + fn)
            )

            f1 = (
                2
                * precision
                * recall
                / (precision + recall)
            )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1


# ============================================================
# Probability metrics
# ============================================================

def probability_metrics(
    y_true,
    probability,
):
    metrics = {
        "brier": brier_score_loss(
            y_true,
            probability,
        ),
        "logloss": log_loss(
            y_true,
            probability,
            labels=[0, 1],
        ),
    }

    if y_true.nunique() >= 2:

        metrics["roc_auc"] = (
            roc_auc_score(
                y_true,
                probability,
            )
        )

        metrics["pr_auc"] = (
            average_precision_score(
                y_true,
                probability,
            )
        )

    else:

        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan

    return metrics


# ============================================================
# F1 metrics
# ============================================================

def classification_metrics(
    y_true,
    prediction,
):
    tp = int(
        ((prediction == 1) & (y_true == 1)).sum()
    )

    fp = int(
        ((prediction == 1) & (y_true == 0)).sum()
    )

    fn = int(
        ((prediction == 0) & (y_true == 1)).sum()
    )

    tn = int(
        ((prediction == 0) & (y_true == 0)).sum()
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2
        * precision
        * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    accuracy = (
        (tp + tn)
        / len(y_true)
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# Satellites
# ============================================================

satellites = sorted(
    train_df["satellite_name"].unique()
)

print("\nHeld-out satellites:")

for satellite in satellites:
    print(f"  - {satellite}")


# ============================================================
# Run experiment
# ============================================================

results = []


for held_out in satellites:

    print("\n")
    print("=" * 70)
    print(f"HOLDING OUT: {held_out}")
    print("=" * 70)

    # --------------------------------------------------------
    # Same leave-one-satellite-out protocol
    # --------------------------------------------------------

    train = train_df[
        train_df["satellite_name"] != held_out
    ].copy()

    val = val_df[
        val_df["satellite_name"] != held_out
    ].copy()

    test = test_df[
        test_df["satellite_name"] == held_out
    ].copy()

    print(
        f"Training samples:   {len(train)}"
    )

    print(
        f"Validation samples: {len(val)}"
    )

    print(
        f"Test samples:       {len(test)}"
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    y_train = train[TARGET].astype(int)
    y_val = val[TARGET].astype(int)
    y_test = test[TARGET].astype(int)

    # --------------------------------------------------------
    # Horizon score
    # --------------------------------------------------------

    train_score = horizon_score(train)
    val_score = horizon_score(val)
    test_score = horizon_score(test)

    # --------------------------------------------------------
    # Learn threshold ONLY from training
    # --------------------------------------------------------

    threshold, train_f1 = find_best_threshold(
        y_train,
        train_score,
    )

    print(
        f"\nTraining threshold: {threshold:.4f}"
    )

    print(
        f"Training F1:        {train_f1:.4f}"
    )

    # --------------------------------------------------------
    # Convert horizon score to a simple probability-like
    # ranking score for probability metrics.
    #
    # We normalize the horizon using the training range.
    # This is NOT claimed to be calibrated probability.
    # It is only used to compare ranking behavior and Brier/
    # log-loss against the ML model.
    # --------------------------------------------------------

    train_min = train_score.min()
    train_max = train_score.max()

    if train_max == train_min:

        train_probability = np.full(
            len(train),
            0.5,
        )

        val_probability = np.full(
            len(val),
            0.5,
        )

        test_probability = np.full(
            len(test),
            0.5,
        )

    else:

        train_probability = (
            (train_score - train_min)
            / (train_max - train_min)
        )

        val_probability = (
            (val_score - train_min)
            / (train_max - train_min)
        )

        test_probability = (
            (test_score - train_min)
            / (train_max - train_min)
        )

        train_probability = np.clip(
            train_probability,
            1e-6,
            1 - 1e-6,
        )

        val_probability = np.clip(
            val_probability,
            1e-6,
            1 - 1e-6,
        )

        test_probability = np.clip(
            test_probability,
            1e-6,
            1 - 1e-6,
        )

    # --------------------------------------------------------
    # Classification performance
    # --------------------------------------------------------

    train_prediction = (
        train_score >= threshold
    ).astype(int)

    val_prediction = (
        val_score >= threshold
    ).astype(int)

    test_prediction = (
        test_score >= threshold
    ).astype(int)

    train_class = classification_metrics(
        y_train,
        train_prediction,
    )

    val_class = classification_metrics(
        y_val,
        val_prediction,
    )

    test_class = classification_metrics(
        y_test,
        test_prediction,
    )

    # --------------------------------------------------------
    # Probability/ranking metrics
    # --------------------------------------------------------

    test_probability_metrics = (
        probability_metrics(
            y_test,
            test_probability,
        )
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print("\nHORIZON-ONLY TEST")

    print(
        f"  Positive rate: "
        f"{y_test.mean():.4f}"
    )

    print(
        f"  Accuracy:  "
        f"{test_class['accuracy']:.4f}"
    )

    print(
        f"  Precision: "
        f"{test_class['precision']:.4f}"
    )

    print(
        f"  Recall:    "
        f"{test_class['recall']:.4f}"
    )

    print(
        f"  F1:        "
        f"{test_class['f1']:.4f}"
    )

    print(
        f"  Brier:     "
        f"{test_probability_metrics['brier']:.4f}"
    )

    print(
        f"  LogLoss:   "
        f"{test_probability_metrics['logloss']:.4f}"
    )

    if not np.isnan(
        test_probability_metrics["roc_auc"]
    ):

        print(
            f"  ROC-AUC:   "
            f"{test_probability_metrics['roc_auc']:.4f}"
        )

        print(
            f"  PR-AUC:    "
            f"{test_probability_metrics['pr_auc']:.4f}"
        )

    else:

        print(
            "  ROC-AUC:   undefined"
        )

        print(
            "  PR-AUC:    undefined"
        )

    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    results.append(
        {
            "held_out_satellite": held_out,
            "test_samples": len(test),
            "test_positive_rate": y_test.mean(),
            "horizon_threshold": threshold,
            "train_f1": train_class["f1"],
            "test_accuracy": test_class["accuracy"],
            "test_precision": test_class["precision"],
            "test_recall": test_class["recall"],
            "test_f1": test_class["f1"],
            "test_brier": test_probability_metrics[
                "brier"
            ],
            "test_logloss": test_probability_metrics[
                "logloss"
            ],
            "test_roc_auc": test_probability_metrics[
                "roc_auc"
            ],
            "test_pr_auc": test_probability_metrics[
                "pr_auc"
            ],
        }
    )


# ============================================================
# Save
# ============================================================

results_df = pd.DataFrame(results)

output_file = (
    RESULTS_DIR
    / "horizon_unseen_satellite_results.csv"
)

results_df.to_csv(
    output_file,
    index=False,
)


# ============================================================
# Compare with XGBoost results
# ============================================================

xgb_file = (
    RESULTS_DIR
    / "xgboost_leave_one_satellite_out_results.csv"
)

if xgb_file.exists():

    xgb_df = pd.read_csv(
        xgb_file
    )

    comparison = results_df.merge(
        xgb_df,
        on="held_out_satellite",
        suffixes=(
            "_horizon",
            "_xgboost",
        ),
    )
    # --------------------------------------------------------
    # The leave-one-out XGBoost experiment reports probability
    # metrics, while this baseline reports classification F1.
    #
    # Therefore compare the metrics that both experiments
    # actually measured:
    #   - ROC-AUC
    #   - PR-AUC
    #
    # F1 is intentionally not compared here because the
    # XGBoost leave-one-out experiment did not save a
    # leave-one-out classification threshold/F1.
    # --------------------------------------------------------

    comparison["roc_auc_gain_xgboost"] = (
        comparison["raw_roc_auc"]
        - comparison["test_roc_auc"]
    )

    comparison["pr_auc_gain_xgboost"] = (
        comparison["raw_pr_auc"]
        - comparison["test_pr_auc"]
    )

    print("\n")
    print("=" * 70)
    print("XGBOOST vs HORIZON-ONLY")
    print("=" * 70)

    print(
        comparison[
            [
                "held_out_satellite",
                "test_f1",
                "test_roc_auc",
                "raw_roc_auc",
                "test_pr_auc",
                "raw_pr_auc",
            ]
        ].to_string(index=False)
    )

    print("\nMean comparison:")

    print(
        f"  Horizon F1: "
        f"{comparison['test_f1'].mean():.4f}"
    )

    print(
        f"  Horizon ROC-AUC: "
        f"{comparison['test_roc_auc'].mean():.4f}"
    )

    print(
        f"  XGBoost ROC-AUC: "
        f"{comparison['raw_roc_auc'].mean():.4f}"
    )

    print(
        f"  Horizon PR-AUC: "
        f"{comparison['test_pr_auc'].mean():.4f}"
    )

    print(
        f"  XGBoost PR-AUC: "
        f"{comparison['raw_pr_auc'].mean():.4f}"
    )

    print(
        "\nNote: F1 is reported for the horizon baseline only. "
        "A directly comparable XGBoost F1 requires a separately "
        "defined leave-one-out threshold experiment."
    )

    print(
        f"\nSaved comparison:"
    )

    print(
        f"  {output_file}"
    )

else:

    print(
        "\nXGBoost leave-one-out results were not found."
    )

    print(
        "The horizon-only results were still saved."
    )


print("\n")
print("=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)

print(
    f"\nSaved:"
)

print(
    f"  {output_file}"
)






