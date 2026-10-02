from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results" / "tables"

horizon_file = RESULTS_DIR / "horizon_unseen_satellite_results.csv"
xgb_file = RESULTS_DIR / "final_calibrated_tuned_xgboost_results.csv"

horizon = pd.read_csv(horizon_file)
xgb = pd.read_csv(xgb_file)

comparison = horizon.merge(
    xgb,
    on="held_out_satellite",
    suffixes=("_horizon", "_xgb"),
)

comparison["roc_auc_gain_xgb"] = (
    comparison["calibrated_roc_auc"]
    - comparison["test_roc_auc"]
)

comparison["pr_auc_gain_xgb"] = (
    comparison["calibrated_pr_auc"]
    - comparison["test_pr_auc"]
)

comparison["f1_gain_xgb"] = (
    comparison["calibrated_f1"]
    - comparison["test_f1"]
)

print("=" * 70)
print("O-RAP - FINAL XGBOOST vs HORIZON-ONLY LOSO COMPARISON")
print("=" * 70)

print("\nPer-satellite comparison:")
print(
    comparison[
        [
            "held_out_satellite",
            "test_samples",
            "test_positive_rate_horizon",
            "test_roc_auc",
            "calibrated_roc_auc",
            "roc_auc_gain_xgb",
            "test_pr_auc",
            "calibrated_pr_auc",
            "pr_auc_gain_xgb",
            "test_f1",
            "calibrated_f1",
            "f1_gain_xgb",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

valid = comparison.dropna(
    subset=[
        "test_roc_auc",
        "calibrated_roc_auc",
        "test_pr_auc",
        "calibrated_pr_auc",
    ]
)

print("\n" + "-" * 70)
print("MEAN COMPARISON")
print("-" * 70)

print(
    f"Horizon-only mean ROC-AUC : "
    f"{valid['test_roc_auc'].mean():.4f}"
)

print(
    f"XGBoost mean ROC-AUC      : "
    f"{valid['calibrated_roc_auc'].mean():.4f}"
)

print(
    f"Mean ROC-AUC gain         : "
    f"{valid['roc_auc_gain_xgb'].mean():+.4f}"
)

print(
    f"\nHorizon-only mean PR-AUC  : "
    f"{valid['test_pr_auc'].mean():.4f}"
)

print(
    f"XGBoost mean PR-AUC       : "
    f"{valid['calibrated_pr_auc'].mean():.4f}"
)

print(
    f"Mean PR-AUC gain          : "
    f"{valid['pr_auc_gain_xgb'].mean():+.4f}"
)

print(
    f"\nHorizon-only mean F1      : "
    f"{comparison['test_f1'].mean():.4f}"
)

print(
    f"XGBoost mean F1           : "
    f"{comparison['calibrated_f1'].mean():.4f}"
)

print(
    f"Mean F1 gain              : "
    f"{comparison['f1_gain_xgb'].mean():+.4f}"
)

print("\nInterpretation:")
print(
    "ROC-AUC and PR-AUC are the primary apples-to-apples "
    "ranking comparisons."
)
print(
    "F1 is comparable because both operating thresholds are "
    "selected without using held-out test labels."
)
print(
    "Horizon-only Brier/LogLoss are NOT treated as calibrated "
    "probability metrics because its normalized horizon score "
    "is explicitly not a calibrated probability."
)

output_file = (
    RESULTS_DIR
    / "final_xgboost_vs_horizon_loso_comparison.csv"
)

comparison.to_csv(
    output_file,
    index=False,
)

print(f"\nSaved:")
print(f"  {output_file}")

