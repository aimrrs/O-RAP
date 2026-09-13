
import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# O-RAP Baseline Experiment
# ============================================================

SPLIT_DIR = Path(
    "data/processed/splits"
)

RESULT_DIR = Path(
    "results/tables"
)


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(
        np.sum(
            (y_true == 1)
            & (y_pred == 1)
        )
    )

    tn = int(
        np.sum(
            (y_true == 0)
            & (y_pred == 0)
        )
    )

    fp = int(
        np.sum(
            (y_true == 0)
            & (y_pred == 1)
        )
    )

    fn = int(
        np.sum(
            (y_true == 1)
            & (y_pred == 0)
        )
    )

    total = len(y_true)

    accuracy = (
        (tp + tn) / total
        if total > 0
        else 0.0
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
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# ============================================================
# Threshold selection
# ============================================================

def choose_threshold(
    train,
    score_column,
):

    y = train[
        "unreliable_1km"
    ].to_numpy()

    scores = train[
        score_column
    ].to_numpy()

    # Candidate thresholds come only from the training set.
    #
    # This prevents validation/test information from influencing
    # the baseline decision rule.

    candidates = np.unique(
        scores[
            np.isfinite(scores)
        ]
    )

    best_threshold = None
    best_f1 = -1.0

    for threshold in candidates:

        predictions = (
            scores >= threshold
        ).astype(int)

        metrics = calculate_metrics(
            y,
            predictions,
        )

        if metrics["f1"] > best_f1:

            best_f1 = metrics["f1"]
            best_threshold = threshold

    return (
        best_threshold,
        best_f1,
    )


# ============================================================
# Evaluate threshold rule
# ============================================================

def evaluate_threshold(
    dataset,
    score_column,
    threshold,
):

    y_true = dataset[
        "unreliable_1km"
    ].to_numpy()

    scores = dataset[
        score_column
    ].to_numpy()

    y_pred = (
        scores >= threshold
    ).astype(int)

    metrics = calculate_metrics(
        y_true,
        y_pred,
    )

    return metrics


# ============================================================
# Horizon-only baseline
# ============================================================

def build_horizon_score(
    dataset,
):

    # A larger horizon is expected to correspond to larger
    # propagation uncertainty.
    #
    # The raw horizon itself is therefore used as the score.

    return dataset[
        "horizon_hours"
    ].astype(float)


# ============================================================
# TLE-age-only baseline
# ============================================================

def build_age_score(
    dataset,
):

    # Older TLEs are expected to be less reliable.

    return dataset[
        "source_tle_age_hours"
    ].astype(float)


# ============================================================
# Combined horizon + TLE-age baseline
# ============================================================

def build_combined_score(
    train,
    validation,
    test,
):

    # --------------------------------------------------------
    # Standardize using TRAIN statistics only.
    # --------------------------------------------------------

    horizon_mean = train[
        "horizon_hours"
    ].mean()

    horizon_std = train[
        "horizon_hours"
    ].std()

    age_mean = train[
        "source_tle_age_hours"
    ].mean()

    age_std = train[
        "source_tle_age_hours"
    ].std()

    if horizon_std == 0:
        horizon_std = 1.0

    if age_std == 0:
        age_std = 1.0

    def score(dataset):

        horizon_z = (
            dataset["horizon_hours"]
            - horizon_mean
        ) / horizon_std

        age_z = (
            dataset[
                "source_tle_age_hours"
            ]
            - age_mean
        ) / age_std

        # Equal-weight combination.
        return (
            horizon_z
            + age_z
        )

    return (
        score(train),
        score(validation),
        score(test),
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 90)
    print("O-RAP BASELINE EXPERIMENT")
    print("=" * 90)

    # --------------------------------------------------------
    # 1. Load temporal splits
    # --------------------------------------------------------

    train = pd.read_csv(
        SPLIT_DIR
        / "train_temporal.csv"
    )

    validation = pd.read_csv(
        SPLIT_DIR
        / "validation_temporal.csv"
    )

    test = pd.read_csv(
        SPLIT_DIR
        / "test_temporal.csv"
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

    # --------------------------------------------------------
    # 2. Construct baseline scores
    # --------------------------------------------------------

    train["horizon_score"] = (
        build_horizon_score(train)
    )

    validation["horizon_score"] = (
        build_horizon_score(validation)
    )

    test["horizon_score"] = (
        build_horizon_score(test)
    )

    train["age_score"] = (
        build_age_score(train)
    )

    validation["age_score"] = (
        build_age_score(validation)
    )

    test["age_score"] = (
        build_age_score(test)
    )

    (
        train["combined_score"],
        validation["combined_score"],
        test["combined_score"],
    ) = build_combined_score(
        train,
        validation,
        test,
    )

    # --------------------------------------------------------
    # 3. Define experiments
    # --------------------------------------------------------

    experiments = [
        (
            "Horizon-only",
            "horizon_score",
        ),
        (
            "TLE-age-only",
            "age_score",
        ),
        (
            "Horizon + TLE-age",
            "combined_score",
        ),
    ]

    results = []

    # --------------------------------------------------------
    # 4. Train-only threshold selection
    # --------------------------------------------------------

    for experiment_name, score_column in experiments:

        print()
        print("-" * 90)
        print(
            f"Baseline: {experiment_name}"
        )
        print("-" * 90)

        threshold, train_f1 = (
            choose_threshold(
                train,
                score_column,
            )
        )

        print(
            f"Training threshold: "
            f"{threshold:.6f}"
        )

        print(
            f"Training F1: "
            f"{train_f1:.4f}"
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        validation_metrics = (
            evaluate_threshold(
                validation,
                score_column,
                threshold,
            )
        )

        # ----------------------------------------------------
        # Test
        # ----------------------------------------------------

        test_metrics = (
            evaluate_threshold(
                test,
                score_column,
                threshold,
            )
        )

        print()
        print("Validation:")

        print(
            f"  Accuracy:  "
            f"{validation_metrics['accuracy']:.4f}"
        )

        print(
            f"  Precision: "
            f"{validation_metrics['precision']:.4f}"
        )

        print(
            f"  Recall:    "
            f"{validation_metrics['recall']:.4f}"
        )

        print(
            f"  F1:        "
            f"{validation_metrics['f1']:.4f}"
        )

        print()
        print("Test:")

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

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append({

            "baseline":
                experiment_name,

            "threshold":
                threshold,

            "train_f1":
                train_f1,

            "validation_accuracy":
                validation_metrics[
                    "accuracy"
                ],

            "validation_precision":
                validation_metrics[
                    "precision"
                ],

            "validation_recall":
                validation_metrics[
                    "recall"
                ],

            "validation_f1":
                validation_metrics[
                    "f1"
                ],

            "test_accuracy":
                test_metrics[
                    "accuracy"
                ],

            "test_precision":
                test_metrics[
                    "precision"
                ],

            "test_recall":
                test_metrics[
                    "recall"
                ],

            "test_f1":
                test_metrics[
                    "f1"
                ],

            "test_tp":
                test_metrics["tp"],

            "test_tn":
                test_metrics["tn"],

            "test_fp":
                test_metrics["fp"],

            "test_fn":
                test_metrics["fn"],
        })

    # --------------------------------------------------------
    # 5. Save results
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        RESULT_DIR
        / "baseline_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False,
    )

    # --------------------------------------------------------
    # 6. Final summary
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("BASELINE EXPERIMENT COMPLETE")
    print("=" * 90)

    print()

    print(
        results_df[
            [
                "baseline",
                "validation_f1",
                "test_f1",
                "test_precision",
                "test_recall",
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

    print()
    print(
        f"Results saved to: "
        f"{output_file}"
    )

    print()
    print("=" * 90)


if __name__ == "__main__":
    main()
