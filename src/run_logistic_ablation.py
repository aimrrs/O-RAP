from pathlib import Path

import numpy as np
import pandas as pd

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

HELD_OUT_SATELLITES = [
    "GPS BIIR-2",
    "Hubble Space Telescope",
    "ISS (ZARYA)",
    "NOAA 19",
    "Sentinel-1A",
    "TDRS-5",
]


# ============================================================
# Feature groups
# ============================================================

HORIZON_FEATURES = [
    "horizon_hours",
]

HORIZON_AGE_FEATURES = [
    "horizon_hours",
    "source_tle_age_hours",
]

FULL_FEATURES = [
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


MODELS = {
    "Horizon only": HORIZON_FEATURES,
    "Horizon + TLE age": HORIZON_AGE_FEATURES,
    "Full orbital/TLE": FULL_FEATURES,
}


# ============================================================
# Helpers
# ============================================================

def load_data():
    train = pd.read_csv(TRAIN_FILE)
    val = pd.read_csv(VAL_FILE)
    test = pd.read_csv(TEST_FILE)

    print(f"Train:      {len(train)}")
    print(f"Validation: {len(val)}")
    print(f"Test:       {len(test)}")

    return train, val, test


def select_training_threshold(y_true, probabilities):
    """
    Select threshold using training-set F1 only.

    Validation/test data never influence the threshold.
    """
    thresholds = np.linspace(0.01, 0.99, 197)

    best_threshold = 0.5
    best_f1 = -1.0

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

    metrics = {
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

    # ROC-AUC and PR-AUC are undefined when the test set
    # contains only one class.
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(
            y_true,
            probabilities,
        )

        metrics["pr_auc"] = average_precision_score(
            y_true,
            probabilities,
        )
    else:
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan

    return metrics


def train_logistic(X_train, y_train):
    """
    Standardization + logistic regression.

    StandardScaler is fitted only on training data.
    Logistic regression is unweighted so that its probabilities
    remain interpretable as model probabilities rather than
    probabilities distorted by class weighting.
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=5000,
        random_state=42,
    )

    model.fit(
        X_train_scaled,
        y_train,
    )

    return scaler, model


def evaluate_model(
    model_name,
    features,
    train_df,
    val_df,
    test_df,
    held_out_satellite,
):
    """
    Leave-one-satellite-out experiment.

    Train:
        non-held-out satellites, Jan-Aug

    Validation:
        non-held-out satellites, Sep-Oct

    Test:
        held-out satellite, Nov-Dec
    """

    # --------------------------------------------------------
    # Leave satellite out
    # --------------------------------------------------------

    train_subset = train_df[
        train_df["satellite_name"] != held_out_satellite
    ].copy()

    val_subset = val_df[
        val_df["satellite_name"] != held_out_satellite
    ].copy()

    test_subset = test_df[
        test_df["satellite_name"] == held_out_satellite
    ].copy()

    X_train = train_subset[features].copy()
    y_train = train_subset[TARGET].astype(int)

    X_val = val_subset[features].copy()
    y_val = val_subset[TARGET].astype(int)

    X_test = test_subset[features].copy()
    y_test = test_subset[TARGET].astype(int)

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    scaler, model = train_logistic(
        X_train,
        y_train,
    )

    train_prob = model.predict_proba(
        scaler.transform(X_train)
    )[:, 1]

    val_prob = model.predict_proba(
        scaler.transform(X_val)
    )[:, 1]

    test_prob = model.predict_proba(
        scaler.transform(X_test)
    )[:, 1]

    # --------------------------------------------------------
    # Threshold from training only
    # --------------------------------------------------------

    threshold, train_f1 = select_training_threshold(
        y_train,
        train_prob,
    )

    train_metrics = calculate_metrics(
        y_train,
        train_prob,
        threshold,
    )

    val_metrics = calculate_metrics(
        y_val,
        val_prob,
        threshold,
    )

    test_metrics = calculate_metrics(
        y_test,
        test_prob,
        threshold,
    )

    return {
        "held_out_satellite": held_out_satellite,
        "model": model_name,
        "features": len(features),

        "train_samples": len(train_subset),
        "validation_samples": len(val_subset),
        "test_samples": len(test_subset),

        "train_positive_rate": y_train.mean(),
        "validation_positive_rate": y_val.mean(),
        "test_positive_rate": y_test.mean(),

        "training_threshold": threshold,
        "training_f1": train_f1,

        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "test_brier": test_metrics["brier"],
        "test_logloss": test_metrics["logloss"],

        "validation_roc_auc": val_metrics["roc_auc"],
        "validation_pr_auc": val_metrics["pr_auc"],
        "validation_brier": val_metrics["brier"],
        "validation_logloss": val_metrics["logloss"],
    }


# ============================================================
# Main experiment
# ============================================================

def main():

    print("=" * 72)
    print("O-RAP — LOGISTIC REGRESSION ABLATION")
    print("=" * 72)

    train, val, test = load_data()

    print()
    print("Held-out satellites:")
    for satellite in HELD_OUT_SATELLITES:
        print(f"  - {satellite}")

    print()

    all_results = []

    for model_name, features in MODELS.items():

        print("-" * 72)
        print(f"MODEL: {model_name}")
        print("-" * 72)

        print("Features:")
        for feature in features:
            print(f"  - {feature}")

        print()

        for satellite in HELD_OUT_SATELLITES:

            result = evaluate_model(
                model_name=model_name,
                features=features,
                train_df=train,
                val_df=val,
                test_df=test,
                held_out_satellite=satellite,
            )

            all_results.append(result)

            print(f"{satellite}")
            print(
                f"  Test samples: "
                f"{result['test_samples']}"
            )
            print(
                f"  Test positive rate: "
                f"{result['test_positive_rate']:.4f}"
            )
            print(
                f"  Threshold: "
                f"{result['training_threshold']:.4f}"
            )
            print(
                f"  Test F1: "
                f"{result['test_f1']:.4f}"
            )
            print(
                f"  Test ROC-AUC: "
                f"{result['test_roc_auc']:.4f}"
                if not np.isnan(result["test_roc_auc"])
                else "  Test ROC-AUC: NaN"
            )
            print(
                f"  Test PR-AUC: "
                f"{result['test_pr_auc']:.4f}"
                if not np.isnan(result["test_pr_auc"])
                else "  Test PR-AUC: NaN"
            )
            print(
                f"  Test Brier: "
                f"{result['test_brier']:.4f}"
            )
            print(
                f"  Test LogLoss: "
                f"{result['test_logloss']:.4f}"
            )
            print()

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results_df = pd.DataFrame(all_results)

    output_file = (
        RESULTS_DIR
        / "logistic_ablation_leave_one_satellite_out.csv"
    )

    results_df.to_csv(
        output_file,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("=" * 72)
    print("MEAN PERFORMANCE ACROSS HELD-OUT SATELLITES")
    print("=" * 72)

    summary = (
        results_df
        .groupby("model")
        .agg(
            mean_test_f1=("test_f1", "mean"),
            mean_test_roc_auc=("test_roc_auc", "mean"),
            mean_test_pr_auc=("test_pr_auc", "mean"),
            mean_test_brier=("test_brier", "mean"),
            mean_test_logloss=("test_logloss", "mean"),
        )
        .reset_index()
    )

    print(summary.to_string(index=False))

    print()
    print(f"Saved results to:")
    print(output_file)
    print()
    print("✓ Logistic ablation completed successfully.")


if __name__ == "__main__":
    main()