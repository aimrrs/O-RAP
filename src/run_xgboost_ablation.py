from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from xgboost import XGBClassifier


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

SPLIT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "splits"
)

RESULTS_DIR = (
    BASE_DIR
    / "results"
    / "tables"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TRAIN_FILE = (
    SPLIT_DIR
    / "train_temporal.csv"
)

VALIDATION_FILE = (
    SPLIT_DIR
    / "validation_temporal.csv"
)

TEST_FILE = (
    SPLIT_DIR
    / "test_temporal.csv"
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "xgboost_ablation_leave_one_satellite_out.csv"
)


# ============================================================
# Configuration
# ============================================================

TARGET = "unreliable_1km"

RANDOM_STATE = 42

SATELLITES = [
    "GPS BIIR-2",
    "Hubble Space Telescope",
    "ISS",
    "NOAA 19",
    "Sentinel-1A",
    "TDRS-5",
]


# ============================================================
# Feature groups
# ============================================================

# Model A:
# Only prediction horizon.
FEATURES_HORIZON = [
    "horizon_hours",
]


# Model B:
# Prediction horizon + TLE age.
FEATURES_HORIZON_AGE = [
    "horizon_hours",
    "source_tle_age_hours",
]


# Model C:
# Full orbital/TLE feature set used by the previous
# XGBoost experiments.
FEATURES_FULL = [
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
    "horizon_only": FEATURES_HORIZON,
    "horizon_age": FEATURES_HORIZON_AGE,
    "full_orbital": FEATURES_FULL,
}


# ============================================================
# Helper functions
# ============================================================

def safe_roc_auc(
    y_true,
    probabilities,
):
    """
    ROC-AUC is undefined when the test set contains
    only one class.
    """

    if pd.Series(y_true).nunique() < 2:
        return np.nan

    return roc_auc_score(
        y_true,
        probabilities,
    )


def safe_pr_auc(
    y_true,
    probabilities,
):
    """
    PR-AUC is undefined when the test set contains
    only one class.
    """

    if pd.Series(y_true).nunique() < 2:
        return np.nan

    return average_precision_score(
        y_true,
        probabilities,
    )


def find_best_threshold(
    y_true,
    probabilities,
):
    """
    Select the classification threshold using TRAINING DATA ONLY.

    The threshold maximizes F1.
    """

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    thresholds = np.unique(
        probabilities
    )

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        if score > best_f1:

            best_f1 = score
            best_threshold = threshold

    return (
        float(best_threshold),
        float(best_f1),
    )


def calculate_metrics(
    y_true,
    probabilities,
    threshold,
):
    """
    Calculate threshold and ranking metrics.
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),

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

        "roc_auc": safe_roc_auc(
            y_true,
            probabilities,
        ),

        "pr_auc": safe_pr_auc(
            y_true,
            probabilities,
        ),
    }


def create_xgboost_model(
    y_train,
):
    """
    Create the same XGBoost configuration used in
    the previous O-RAP experiments.
    """

    positive_count = np.sum(
        y_train == 1
    )

    negative_count = np.sum(
        y_train == 0
    )

    if positive_count == 0:

        scale_pos_weight = 1.0

    else:

        scale_pos_weight = (
            negative_count
            / positive_count
        )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return (
        model,
        float(scale_pos_weight),
    )


# ============================================================
# Load data
# ============================================================

print("=" * 70)
print("O-RAP — XGBOOST FEATURE ABLATION")
print("LEAVE-ONE-SATELLITE-OUT + TEMPORAL TEST")
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

print()
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

print("Models:")

print(
    "  A: Horizon only"
)

print(
    "  B: Horizon + TLE age"
)

print(
    "  C: Full orbital/TLE features"
)

print()

print("Held-out satellites:")

for satellite in SATELLITES:

    print(
        f"  - {satellite}"
    )


# ============================================================
# Validate feature availability
# ============================================================

all_required_features = sorted(
    set(
        FEATURES_FULL
    )
)

for feature in all_required_features:

    if feature not in train.columns:

        raise KeyError(
            f"Missing feature in training data: "
            f"{feature}"
        )

    if feature not in validation.columns:

        raise KeyError(
            f"Missing feature in validation data: "
            f"{feature}"
        )

    if feature not in test.columns:

        raise KeyError(
            f"Missing feature in test data: "
            f"{feature}"
        )


# ============================================================
# Results
# ============================================================

results = []


# ============================================================
# Leave-one-satellite-out
# ============================================================

for held_out in SATELLITES:

    print()
    print()
    print("=" * 70)
    print(
        f"HOLDING OUT: {held_out}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Satellite split
    # --------------------------------------------------------

    train_subset = train.loc[
        train["satellite_name"]
        != held_out
    ].copy()

    validation_subset = validation.loc[
        validation["satellite_name"]
        != held_out
    ].copy()

    test_subset = test.loc[
        test["satellite_name"]
        == held_out
    ].copy()

    print()
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

    y_validation = (
        validation_subset[TARGET]
        .astype(int)
        .to_numpy()
    )

    y_test = (
        test_subset[TARGET]
        .astype(int)
        .to_numpy()
    )

    print()
    print(
        f"Training positive rate: "
        f"{y_train.mean():.4f}"
    )

    print(
        f"Validation positive rate: "
        f"{y_validation.mean():.4f}"
    )

    print(
        f"Test positive rate: "
        f"{y_test.mean():.4f}"
    )

    # --------------------------------------------------------
    # Run each ablation model
    # --------------------------------------------------------

    for model_name, features in MODELS.items():

        print()
        print(
            "-" * 70
        )

        if model_name == "horizon_only":

            display_name = (
                "MODEL A — HORIZON ONLY"
            )

        elif model_name == "horizon_age":

            display_name = (
                "MODEL B — HORIZON + TLE AGE"
            )

        else:

            display_name = (
                "MODEL C — FULL ORBITAL/TLE"
            )

        print(display_name)

        print(
            "-" * 70
        )

        print(
            "Features:"
        )

        for feature in features:

            print(
                f"  - {feature}"
            )

        # ----------------------------------------------------
        # Feature matrices
        # ----------------------------------------------------

        X_train = (
            train_subset[features]
            .astype(float)
        )

        X_validation = (
            validation_subset[features]
            .astype(float)
        )

        X_test = (
            test_subset[features]
            .astype(float)
        )

        # ----------------------------------------------------
        # Create model
        # ----------------------------------------------------

        model, scale_pos_weight = (
            create_xgboost_model(
                y_train
            )
        )

        print()
        print(
            f"Scale pos weight: "
            f"{scale_pos_weight:.4f}"
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train,
        )

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        train_probabilities = (
            model.predict_proba(
                X_train
            )[:, 1]
        )

        validation_probabilities = (
            model.predict_proba(
                X_validation
            )[:, 1]
        )

        test_probabilities = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        # ----------------------------------------------------
        # Threshold:
        #
        # TRAINING ONLY.
        #
        # Validation and test are never used to choose
        # the classification threshold.
        # ----------------------------------------------------

        threshold, train_f1 = (
            find_best_threshold(
                y_train,
                train_probabilities,
            )
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        train_metrics = calculate_metrics(
            y_train,
            train_probabilities,
            threshold,
        )

        validation_metrics = calculate_metrics(
            y_validation,
            validation_probabilities,
            threshold,
        )

        test_metrics = calculate_metrics(
            y_test,
            test_probabilities,
            threshold,
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        print()
        print(
            f"Training threshold: "
            f"{threshold:.4f}"
        )

        print(
            f"Training F1:        "
            f"{train_f1:.4f}"
        )

        print()
        print("TEST RESULTS")

        print(
            f"  Accuracy:  "
            f"{test_metrics['accuracy']:.4f}"
        )

        print(
            f"  Precision: "
            f"{test_metrics['precision']:.4f}"
        )

        print(
            f"  Recall:    "
            f"{test_metrics['recall']:.4f}"
        )

        print(
            f"  F1:        "
            f"{test_metrics['f1']:.4f}"
        )

        if np.isnan(
            test_metrics["roc_auc"]
        ):

            print(
                "  ROC-AUC:   undefined"
            )

        else:

            print(
                f"  ROC-AUC:   "
                f"{test_metrics['roc_auc']:.4f}"
            )

        if np.isnan(
            test_metrics["pr_auc"]
        ):

            print(
                "  PR-AUC:    undefined"
            )

        else:

            print(
                f"  PR-AUC:    "
                f"{test_metrics['pr_auc']:.4f}"
            )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        results.append(
            {
                "held_out_satellite": held_out,

                "model": model_name,

                "n_features": len(
                    features
                ),

                "train_samples": len(
                    train_subset
                ),

                "validation_samples": len(
                    validation_subset
                ),

                "test_samples": len(
                    test_subset
                ),

                "train_positive_rate": (
                    y_train.mean()
                ),

                "validation_positive_rate": (
                    y_validation.mean()
                ),

                "test_positive_rate": (
                    y_test.mean()
                ),

                "scale_pos_weight": (
                    scale_pos_weight
                ),

                "threshold": threshold,

                "train_f1": (
                    train_metrics["f1"]
                ),

                "validation_f1": (
                    validation_metrics["f1"]
                ),

                "test_accuracy": (
                    test_metrics["accuracy"]
                ),

                "test_precision": (
                    test_metrics["precision"]
                ),

                "test_recall": (
                    test_metrics["recall"]
                ),

                "test_f1": (
                    test_metrics["f1"]
                ),

                "test_roc_auc": (
                    test_metrics["roc_auc"]
                ),

                "test_pr_auc": (
                    test_metrics["pr_auc"]
                ),
            }
        )


# ============================================================
# Results DataFrame
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# Save results
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# Display summary
# ============================================================

print()
print()
print("=" * 70)
print("ABLATION SUMMARY")
print("=" * 70)

summary_columns = [
    "held_out_satellite",
    "model",
    "test_f1",
    "test_roc_auc",
    "test_pr_auc",
]

print()

print(
    results_df[
        summary_columns
    ].to_string(index=False)
)


# ============================================================
# Mean metrics by model
# ============================================================

print()
print("=" * 70)
print("MEAN TEST PERFORMANCE")
print("=" * 70)

mean_results = (
    results_df
    .groupby("model", as_index=False)
    .agg(
        mean_f1=(
            "test_f1",
            "mean",
        ),
        mean_roc_auc=(
            "test_roc_auc",
            "mean",
        ),
        mean_pr_auc=(
            "test_pr_auc",
            "mean",
        ),
    )
)


print()

print(
    mean_results.to_string(
        index=False
    )
)


# ============================================================
# Pairwise gains relative to horizon-only
# ============================================================

print()
print("=" * 70)
print("GAINS RELATIVE TO HORIZON-ONLY")
print("=" * 70)

pivot_roc = (
    results_df
    .pivot(
        index="held_out_satellite",
        columns="model",
        values="test_roc_auc",
    )
)

pivot_pr = (
    results_df
    .pivot(
        index="held_out_satellite",
        columns="model",
        values="test_pr_auc",
    )
)

pivot_f1 = (
    results_df
    .pivot(
        index="held_out_satellite",
        columns="model",
        values="test_f1",
    )
)

gain_rows = []

for satellite in SATELLITES:

    horizon_roc = pivot_roc.loc[
        satellite,
        "horizon_only",
    ]

    age_roc = pivot_roc.loc[
        satellite,
        "horizon_age",
    ]

    full_roc = pivot_roc.loc[
        satellite,
        "full_orbital",
    ]

    horizon_pr = pivot_pr.loc[
        satellite,
        "horizon_only",
    ]

    age_pr = pivot_pr.loc[
        satellite,
        "horizon_age",
    ]

    full_pr = pivot_pr.loc[
        satellite,
        "full_orbital",
    ]

    horizon_f1 = pivot_f1.loc[
        satellite,
        "horizon_only",
    ]

    age_f1 = pivot_f1.loc[
        satellite,
        "horizon_age",
    ]

    full_f1 = pivot_f1.loc[
        satellite,
        "full_orbital",
    ]

    gain_rows.append(
        {
            "held_out_satellite": satellite,

            "age_vs_horizon_roc_auc": (
                age_roc - horizon_roc
            ),

            "full_vs_horizon_roc_auc": (
                full_roc - horizon_roc
            ),

            "age_vs_horizon_pr_auc": (
                age_pr - horizon_pr
            ),

            "full_vs_horizon_pr_auc": (
                full_pr - horizon_pr
            ),

            "age_vs_horizon_f1": (
                age_f1 - horizon_f1
            ),

            "full_vs_horizon_f1": (
                full_f1 - horizon_f1
            ),
        }
    )


gain_df = pd.DataFrame(
    gain_rows
)

print()

print(
    gain_df.to_string(
        index=False
    )
)


# ============================================================
# Overall gains
# ============================================================

print()
print("=" * 70)
print("OVERALL ABLATION FINDINGS")
print("=" * 70)

print()

print(
    "Mean ROC-AUC:"
)

for _, row in mean_results.iterrows():

    print(
        f"  {row['model']}: "
        f"{row['mean_roc_auc']:.4f}"
    )

print()

print(
    "Mean PR-AUC:"
)

for _, row in mean_results.iterrows():

    print(
        f"  {row['model']}: "
        f"{row['mean_pr_auc']:.4f}"
    )

print()

print(
    "Mean F1:"
)

for _, row in mean_results.iterrows():

    print(
        f"  {row['model']}: "
        f"{row['mean_f1']:.4f}"
    )


# ============================================================
# Count satellites where full model improves
# ============================================================

full_vs_horizon_roc = (
    gain_df[
        "full_vs_horizon_roc_auc"
    ]
)

full_vs_horizon_pr = (
    gain_df[
        "full_vs_horizon_pr_auc"
    ]
)
full_vs_horizon_f1 = (
    gain_df[
        "full_vs_horizon_f1"
    ]
)

print()
print(
    "Full model vs horizon-only:"
)

print(
    f"  ROC-AUC improvement: "
    f"{(full_vs_horizon_roc > 0).sum()}"
    f"/{len(gain_df)} satellites"
)

print(
    f"  PR-AUC improvement:  "
    f"{(full_vs_horizon_pr > 0).sum()}"
    f"/{len(gain_df)} satellites"
)

print(
    f"  F1 improvement:      "
    f"{(full_vs_horizon_f1 > 0).sum()}"
    f"/{len(gain_df)} satellites"
)


# ============================================================
# Final
# ============================================================

print()
print("=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)

print()
print("Saved:")
print(
    f"  {OUTPUT_FILE}"
)