from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
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

HELD_OUT_SATELLITES = [
    "GPS BIIR-2",
    "Hubble Space Telescope",
    "ISS",
    "NOAA 19",
    "Sentinel-1A",
    "TDRS-5",
]


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


# ============================================================
# Controlled hyperparameter search
# ============================================================
#
# We deliberately keep this small.
#
# The goal is NOT to brute-force thousands of combinations.
# The goal is to determine whether reasonable regularization
# and model-capacity changes improve cross-satellite
# generalization.
#
# ============================================================

PARAMETER_GRID = [
    {
        "name": "baseline",
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },
    {
        "name": "shallow_regularized",
        "n_estimators": 400,
        "max_depth": 3,
        "learning_rate": 0.05,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
    },
    {
        "name": "medium_regularized",
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
    },
    {
        "name": "strong_regularization",
        "n_estimators": 500,
        "max_depth": 3,
        "learning_rate": 0.03,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 5.0,
    },
    {
        "name": "low_depth",
        "n_estimators": 500,
        "max_depth": 2,
        "learning_rate": 0.05,
        "min_child_weight": 3,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
    },
    {
        "name": "more_capacity",
        "n_estimators": 400,
        "max_depth": 5,
        "learning_rate": 0.05,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
    },
]


# ============================================================
# Data
# ============================================================

def load_data():

    train = pd.read_csv(TRAIN_FILE)
    validation = pd.read_csv(VAL_FILE)
    test = pd.read_csv(TEST_FILE)

    print(f"Train samples:      {len(train)}")
    print(f"Validation samples: {len(validation)}")
    print(f"Test samples:       {len(test)}")

    return train, validation, test


# ============================================================
# Threshold selection
# ============================================================

def select_threshold(y_true, probabilities):

    thresholds = np.linspace(0.01, 0.99, 197)

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

    return best_threshold, best_f1


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    result = {
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

    if len(np.unique(y_true)) == 2:

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


# ============================================================
# XGBoost model
# ============================================================

def create_model(parameters, scale_pos_weight):

    model = xgb.XGBClassifier(
        n_estimators=parameters["n_estimators"],
        max_depth=parameters["max_depth"],
        learning_rate=parameters["learning_rate"],
        min_child_weight=parameters["min_child_weight"],
        subsample=parameters["subsample"],
        colsample_bytree=parameters["colsample_bytree"],
        reg_alpha=parameters["reg_alpha"],
        reg_lambda=parameters["reg_lambda"],

        objective="binary:logistic",
        eval_metric="logloss",

        scale_pos_weight=scale_pos_weight,

        random_state=42,
        n_jobs=-1,
    )

    return model


# ============================================================
# Tune one held-out satellite
# ============================================================

def tune_for_satellite(
    train_df,
    validation_df,
    held_out_satellite,
):

    # --------------------------------------------------------
    # Remove held-out satellite completely
    # --------------------------------------------------------

    train_subset = train_df[
        train_df["satellite_name"] != held_out_satellite
    ].copy()

    validation_subset = validation_df[
        validation_df["satellite_name"] != held_out_satellite
    ].copy()

    X_train = train_subset[FEATURES]
    y_train = train_subset[TARGET].astype(int)

    X_validation = validation_subset[FEATURES]
    y_validation = validation_subset[TARGET].astype(int)

    positive = int(y_train.sum())
    negative = int(len(y_train) - positive)

    scale_pos_weight = (
        negative / positive
        if positive > 0
        else 1.0
    )

    print()
    print("=" * 72)
    print(f"HELD-OUT SATELLITE: {held_out_satellite}")
    print("=" * 72)

    print(
        f"Training samples:   {len(train_subset)}"
    )

    print(
        f"Validation samples: {len(validation_subset)}"
    )

    print(
        f"Training positives:  {positive}"
    )

    print(
        f"Training positive rate: "
        f"{y_train.mean():.4f}"
    )

    print(
        f"scale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    tuning_results = []

    # --------------------------------------------------------
    # Evaluate each parameter configuration
    # --------------------------------------------------------

    for index, parameters in enumerate(
        PARAMETER_GRID,
        start=1,
    ):

        print()
        print(
            f"[{index}/{len(PARAMETER_GRID)}] "
            f"Testing configuration: "
            f"{parameters['name']}"
        )

        model = create_model(
            parameters,
            scale_pos_weight,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[
                (X_validation, y_validation)
            ],
            verbose=False,
        )

        train_prob = model.predict_proba(
            X_train
        )[:, 1]

        validation_prob = model.predict_proba(
            X_validation
        )[:, 1]

        threshold, train_f1 = (
            select_threshold(
                y_train,
                train_prob,
            )
        )

        validation_metrics = calculate_metrics(
            y_validation,
            validation_prob,
            threshold,
        )

        row = {
            "held_out_satellite": held_out_satellite,
            "configuration": parameters["name"],

            "n_estimators": parameters["n_estimators"],
            "max_depth": parameters["max_depth"],
            "learning_rate": parameters["learning_rate"],
            "min_child_weight": parameters["min_child_weight"],
            "subsample": parameters["subsample"],
            "colsample_bytree": parameters["colsample_bytree"],
            "reg_alpha": parameters["reg_alpha"],
            "reg_lambda": parameters["reg_lambda"],

            "training_threshold": threshold,
            "training_f1": train_f1,

            "validation_f1": validation_metrics["f1"],
            "validation_roc_auc": validation_metrics["roc_auc"],
            "validation_pr_auc": validation_metrics["pr_auc"],
            "validation_brier": validation_metrics["brier"],
            "validation_logloss": validation_metrics["logloss"],
        }

        tuning_results.append(row)

        print(
            f"  Validation F1: "
            f"{validation_metrics['f1']:.4f}"
        )

        if np.isnan(validation_metrics["roc_auc"]):
            print("  Validation ROC-AUC: NaN")
        else:
            print(
                f"  Validation ROC-AUC: "
                f"{validation_metrics['roc_auc']:.4f}"
            )

        if np.isnan(validation_metrics["pr_auc"]):
            print("  Validation PR-AUC: NaN")
        else:
            print(
                f"  Validation PR-AUC: "
                f"{validation_metrics['pr_auc']:.4f}"
            )

        print(
            f"  Validation Brier: "
            f"{validation_metrics['brier']:.4f}"
        )

        print(
            f"  Validation LogLoss: "
            f"{validation_metrics['logloss']:.4f}"
        )

    tuning_df = pd.DataFrame(tuning_results)

    # --------------------------------------------------------
    # Select best configuration
    #
    # Primary criterion:
    # validation PR-AUC
    #
    # PR-AUC is more informative than accuracy under the
    # imbalanced reliability target.
    # --------------------------------------------------------

    if tuning_df["validation_pr_auc"].notna().any():

        best_row = (
            tuning_df
            .dropna(subset=["validation_pr_auc"])
            .sort_values(
                [
                    "validation_pr_auc",
                    "validation_brier",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .iloc[0]
        )

    else:

        best_row = (
            tuning_df
            .sort_values(
                [
                    "validation_brier",
                ],
                ascending=[
                    True,
                ],
            )
            .iloc[0]
        )

    print()
    print(
        f"BEST CONFIGURATION: "
        f"{best_row['configuration']}"
    )

    print(
        f"Validation PR-AUC: "
        f"{best_row['validation_pr_auc']:.4f}"
        if not np.isnan(
            best_row["validation_pr_auc"]
        )
        else "Validation PR-AUC: NaN"
    )

    print(
        f"Validation ROC-AUC: "
        f"{best_row['validation_roc_auc']:.4f}"
        if not np.isnan(
            best_row["validation_roc_auc"]
        )
        else "Validation ROC-AUC: NaN"
    )

    print(
        f"Validation Brier: "
        f"{best_row['validation_brier']:.4f}"
    )

    print(
        f"Validation LogLoss: "
        f"{best_row['validation_logloss']:.4f}"
    )

    return tuning_df, best_row


# ============================================================
# Final evaluation
# ============================================================

def evaluate_best_model(
    train_df,
    validation_df,
    test_df,
    held_out_satellite,
    best_row,
):

    train_subset = train_df[
        train_df["satellite_name"] != held_out_satellite
    ].copy()

    validation_subset = validation_df[
        validation_df["satellite_name"] != held_out_satellite
    ].copy()

    test_subset = test_df[
        test_df["satellite_name"] == held_out_satellite
    ].copy()

    X_train = train_subset[FEATURES]
    y_train = train_subset[TARGET].astype(int)

    X_validation = validation_subset[FEATURES]
    y_validation = validation_subset[TARGET].astype(int)

    X_test = test_subset[FEATURES]
    y_test = test_subset[TARGET].astype(int)

    positive = int(y_train.sum())
    negative = int(len(y_train) - positive)

    scale_pos_weight = (
        negative / positive
        if positive > 0
        else 1.0
    )

    parameters = {
        "n_estimators": int(
            best_row["n_estimators"]
        ),
        "max_depth": int(
            best_row["max_depth"]
        ),
        "learning_rate": float(
            best_row["learning_rate"]
        ),
        "min_child_weight": int(
            best_row["min_child_weight"]
        ),
        "subsample": float(
            best_row["subsample"]
        ),
        "colsample_bytree": float(
            best_row["colsample_bytree"]
        ),
        "reg_alpha": float(
            best_row["reg_alpha"]
        ),
        "reg_lambda": float(
            best_row["reg_lambda"]
        ),
    }

    model = create_model(
        parameters,
        scale_pos_weight,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_validation, y_validation)
        ],
        verbose=False,
    )

    train_prob = model.predict_proba(
        X_train
    )[:, 1]

    validation_prob = model.predict_proba(
        X_validation
    )[:, 1]

    test_prob = model.predict_proba(
        X_test
    )[:, 1]

    # Threshold comes from training only.
    threshold, train_f1 = (
        select_threshold(
            y_train,
            train_prob,
        )
    )

    validation_metrics = calculate_metrics(
        y_validation,
        validation_prob,
        threshold,
    )

    test_metrics = calculate_metrics(
        y_test,
        test_prob,
        threshold,
    )

    print()
    print(
        f"FINAL TEST — {held_out_satellite}"
    )

    print(
        f"  Test samples: "
        f"{len(test_subset)}"
    )

    print(
        f"  Test positive rate: "
        f"{y_test.mean():.4f}"
    )

    print(
        f"  Threshold: "
        f"{threshold:.4f}"
    )

    print(
        f"  Test F1: "
        f"{test_metrics['f1']:.4f}"
    )

    if np.isnan(test_metrics["roc_auc"]):
        print("  Test ROC-AUC: NaN")
    else:
        print(
            f"  Test ROC-AUC: "
            f"{test_metrics['roc_auc']:.4f}"
        )

    if np.isnan(test_metrics["pr_auc"]):
        print("  Test PR-AUC: NaN")
    else:
        print(
            f"  Test PR-AUC: "
            f"{test_metrics['pr_auc']:.4f}"
        )

    print(
        f"  Test Brier: "
        f"{test_metrics['brier']:.4f}"
    )

    print(
        f"  Test LogLoss: "
        f"{test_metrics['logloss']:.4f}"
    )

    return {
        "held_out_satellite": held_out_satellite,

        "configuration": best_row[
            "configuration"
        ],

        "n_estimators": parameters[
            "n_estimators"
        ],

        "max_depth": parameters[
            "max_depth"
        ],

        "learning_rate": parameters[
            "learning_rate"
        ],

        "min_child_weight": parameters[
            "min_child_weight"
        ],

        "subsample": parameters[
            "subsample"
        ],

        "colsample_bytree": parameters[
            "colsample_bytree"
        ],

        "reg_alpha": parameters[
            "reg_alpha"
        ],

        "reg_lambda": parameters[
            "reg_lambda"
        ],

        "training_threshold": threshold,
        "training_f1": train_f1,

        "validation_f1": validation_metrics[
            "f1"
        ],

        "validation_roc_auc": validation_metrics[
            "roc_auc"
        ],

        "validation_pr_auc": validation_metrics[
            "pr_auc"
        ],

        "validation_brier": validation_metrics[
            "brier"
        ],

        "validation_logloss": validation_metrics[
            "logloss"
        ],

        "test_samples": len(test_subset),

        "test_positive_rate": y_test.mean(),

        "test_accuracy": test_metrics[
            "accuracy"
        ],

        "test_precision": test_metrics[
            "precision"
        ],

        "test_recall": test_metrics[
            "recall"
        ],

        "test_f1": test_metrics[
            "f1"
        ],

        "test_roc_auc": test_metrics[
            "roc_auc"
        ],

        "test_pr_auc": test_metrics[
            "pr_auc"
        ],

        "test_brier": test_metrics[
            "brier"
        ],

        "test_logloss": test_metrics[
            "logloss"
        ],
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 72)
    print("O-RAP — CONTROLLED XGBOOST TUNING")
    print("=" * 72)

    train_df, validation_df, test_df = (
        load_data()
    )

    all_tuning_results = []
    all_test_results = []

    for satellite in HELD_OUT_SATELLITES:

        tuning_df, best_row = tune_for_satellite(
            train_df,
            validation_df,
            satellite,
        )

        all_tuning_results.append(
            tuning_df
        )

        test_result = evaluate_best_model(
            train_df,
            validation_df,
            test_df,
            satellite,
            best_row,
        )

        all_test_results.append(
            test_result
        )

    # --------------------------------------------------------
    # Save tuning results
    # --------------------------------------------------------

    tuning_results = pd.concat(
        all_tuning_results,
        ignore_index=True,
    )

    tuning_file = (
        RESULTS_DIR
        / "xgboost_tuning_validation_results.csv"
    )

    tuning_results.to_csv(
        tuning_file,
        index=False,
    )

    # --------------------------------------------------------
    # Save final test results
    # --------------------------------------------------------

    test_results = pd.DataFrame(
        all_test_results
    )

    test_file = (
        RESULTS_DIR
        / "xgboost_tuned_leave_one_satellite_out.csv"
    )

    test_results.to_csv(
        test_file,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("TUNED XGBOOST — FINAL SUMMARY")
    print("=" * 72)

    summary_columns = [
        "held_out_satellite",
        "configuration",
        "test_f1",
        "test_roc_auc",
        "test_pr_auc",
        "test_brier",
        "test_logloss",
    ]

    print(
        test_results[
            summary_columns
        ].to_string(index=False)
    )

    print()
    print("=" * 72)
    print("MEAN PERFORMANCE")
    print("=" * 72)

    mean_results = (
        test_results[
            [
                "test_f1",
                "test_roc_auc",
                "test_pr_auc",
                "test_brier",
                "test_logloss",
            ]
        ]
        .mean()
    )

    print(
        f"Mean Test F1:      "
        f"{mean_results['test_f1']:.4f}"
    )

    print(
        f"Mean Test ROC-AUC: "
        f"{mean_results['test_roc_auc']:.4f}"
    )

    print(
        f"Mean Test PR-AUC:  "
        f"{mean_results['test_pr_auc']:.4f}"
    )

    print(
        f"Mean Test Brier:   "
        f"{mean_results['test_brier']:.4f}"
    )

    print(
        f"Mean Test LogLoss: "
        f"{mean_results['test_logloss']:.4f}"
    )

    print()
    print("Saved:")
    print(tuning_file)
    print(test_file)

    print()
    print("✓ XGBoost tuning completed successfully.")


if __name__ == "__main__":
    main()