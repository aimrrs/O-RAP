from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score

out = Path(sys.argv[1])
fold = pd.read_csv(out / "nested_loso_by_satellite.csv")
pred = pd.read_csv(out / "nested_loso_test_predictions.csv")
oof = pd.read_csv(out / "nested_loso_oof_calibration_predictions.csv")
methods = list(fold.model.unique())
summaries = []
for model in methods:
    g = fold.loc[fold.model.eq(model)]
    for subset, d in [("all_defined_folds", g),
                      ("exclude_GPS_one_positive", g.loc[g.outer_satellite.ne("GPS BIIR-2")]),
                      ("positive_event_folds", g.loc[g.test_positives.gt(0)])]:
        row = {"model": model, "subset": subset, "folds": len(d),
               "two_class_folds": int(d.roc_auc.notna().sum())}
        for col in ["roc_auc", "average_precision", "brier", "log_loss", "f1", "precision", "recall"]:
            row[col + "_mean"] = d[col].mean()
        row["ap_lift_mean"] = (d.loc[d.average_precision.notna() & d.test_prevalence.gt(0), "average_precision"] /
                               d.loc[d.average_precision.notna() & d.test_prevalence.gt(0), "test_prevalence"]).mean()
        summaries.append(row)
pd.DataFrame(summaries).to_csv(out / "nested_loso_final_macro_sensitivity.csv", index=False)

base = fold.loc[fold.model.eq("nested_xgboost_oof_platt")].set_index("outer_satellite")
paired = []
for comparator in ["horizon_only", "age_plus_horizon", "logistic_regression_raw", "training_prevalence"]:
    other = fold.loc[fold.model.eq(comparator)].set_index("outer_satellite")
    for sat in base.index:
        paired.append({"comparison": f"xgb_oof_platt_minus_{comparator}", "satellite": sat,
                       "test_positives": int(base.loc[sat, "test_positives"]),
                       **{f"delta_{m}": base.loc[sat, m] - other.loc[sat, m]
                          for m in ["roc_auc", "average_precision", "brier", "log_loss", "f1"]}})
pd.DataFrame(paired).to_csv(out / "nested_loso_final_paired_deltas.csv", index=False)

params = []
for outer, g in oof.groupby("outer_satellite"):
    for name, col in [("xgboost", "xgb_margin"), ("logistic_regression", "lr_margin")]:
        p = g[col].to_numpy(dtype=float)
        y = g.y_true.to_numpy(dtype=int)
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(p.reshape(-1, 1), y)
        params.append({"outer_satellite": outer, "model": name,
                       "calibration_slope": float(model.coef_[0, 0]),
                       "calibration_intercept": float(model.intercept_[0]),
                       "oof_rows": len(g), "oof_positive_rate": float(y.mean())})
pd.DataFrame(params).to_csv(out / "nested_loso_final_calibration_parameters.csv", index=False)

fig, ax = plt.subplots(figsize=(6.3, 4.7))
y = pred.y_true.to_numpy(dtype=int)
for col, label, style, marker in [("xgb_raw", "XGBoost raw", "-", "o"),
                                  ("xgb_oof_platt", "XGBoost OOF-Platt", "--", "s")]:
    probability = pred[col].to_numpy(dtype=float)
    bins = pd.qcut(probability, q=8, labels=False, duplicates="drop")
    grouped = pd.DataFrame({"bin": bins, "p": probability, "y": y}).groupby("bin", observed=True)
    mean = grouped.p.mean().to_numpy()
    frac = grouped.y.mean().to_numpy()
    counts = grouped.size().to_numpy()
    ax.plot(mean, frac, linestyle=style, marker=marker, linewidth=1.7, markersize=5, label=label)
ax.plot([0, 1], [0, 1], "k:", linewidth=1.2, label="Ideal")
ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability",
       ylabel="Observed event fraction", title="Pooled outer-test calibration")
ax.grid(alpha=.2); ax.legend(frameon=False); fig.tight_layout()
fig.savefig(out / "nested_loso_final_reliability.png", dpi=300)
plt.close(fig)
print(pd.DataFrame(summaries).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("Calibration parameters")
print(pd.DataFrame(params).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
