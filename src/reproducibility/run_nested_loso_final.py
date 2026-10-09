from pathlib import Path
import hashlib
import json
import platform
import sys

import numpy as np
import pandas as pd
import matplotlib
import scipy
import sklearn
import sgp4
import xgboost
import xgboost as xgb
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, log_loss, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path(sys.argv[1])  # processed input directory
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
FEATURES = ["horizon_hours", "source_tle_age_hours", "mean_motion",
            "eccentricity", "inclination", "ra_of_asc_node",
            "arg_of_pericenter", "mean_anomaly", "bstar",
            "mean_motion_dot", "mean_motion_ddot", "semimajor_axis",
            "period", "apoapsis", "periapsis"]
TARGET = "unreliable_1km"
CONFIGS = {
    "baseline": dict(n_estimators=300, max_depth=4, learning_rate=.05,
                     min_child_weight=1, subsample=.8, colsample_bytree=.8,
                     reg_alpha=0., reg_lambda=1.),
    "shallow_regularized": dict(n_estimators=400, max_depth=3, learning_rate=.05,
                                 min_child_weight=3, subsample=.8, colsample_bytree=.8,
                                 reg_alpha=.1, reg_lambda=2.),
    "medium_regularized": dict(n_estimators=400, max_depth=4, learning_rate=.05,
                                min_child_weight=3, subsample=.8, colsample_bytree=.8,
                                reg_alpha=.1, reg_lambda=2.),
    "strong_regularization": dict(n_estimators=500, max_depth=3, learning_rate=.03,
                                  min_child_weight=5, subsample=.8, colsample_bytree=.8,
                                  reg_alpha=.5, reg_lambda=5.),
    "low_depth": dict(n_estimators=500, max_depth=2, learning_rate=.05,
                      min_child_weight=3, subsample=.9, colsample_bytree=.9,
                      reg_alpha=.1, reg_lambda=2.),
    "more_capacity": dict(n_estimators=400, max_depth=5, learning_rate=.05,
                          min_child_weight=3, subsample=.8, colsample_bytree=.8,
                          reg_alpha=.1, reg_lambda=2.)}
C_GRID = [.01, .1, 1., 10., 100.]

# Source creation time defines the original temporal split. Purge any
# train/validation row whose reference TLE is not available by that split's
# cutoff. Test labels may use later references for outcome ascertainment.
labels = pd.read_csv(DATA / "orap_multisatellite_labels_2025.csv")
features = pd.read_csv(DATA / "orap_features_2025.csv")
for col in ["source_creation", "source_epoch", "reference_creation"]:
    labels[col] = pd.to_datetime(labels[col], format="mixed", utc=True)
ambiguous = (labels.norad_id.eq(25544)
             & labels.source_creation.eq(pd.Timestamp("2025-01-25T19:06:24Z"))
             & labels.source_epoch.eq(pd.Timestamp("2025-01-25T04:55:56.509824Z")))
labels = labels.loc[~ambiguous].reset_index(drop=True)
assert len(labels) == len(features)
for col in ["norad_id", "satellite_name", "horizon_hours"]:
    assert labels[col].equals(features[col]), f"Feature/label order mismatch: {col}"
assert (labels.source_tle_age_hours - features.source_tle_age_hours).abs().max() < 1e-7
data = features.copy()
data["source_creation"] = labels.source_creation
data["source_epoch"] = labels.source_epoch
data["reference_creation"] = labels.reference_creation
data["reference_delay_hours"] = labels.reference_delay_hours
data["satellite_name"] = data.satellite_name.astype(str)
data[TARGET] = data[TARGET].astype(int)
created = data.source_creation
train_cut = pd.Timestamp("2025-09-01", tz="UTC")
test_cut = pd.Timestamp("2025-11-01", tz="UTC")
end_cut = pd.Timestamp("2026-01-01", tz="UTC")
train = data.loc[created.lt(train_cut) & data.reference_creation.lt(train_cut)].copy()
val = data.loc[created.ge(train_cut) & created.lt(test_cut)
               & data.reference_creation.lt(test_cut)].copy()
test = data.loc[created.ge(test_cut) & created.lt(end_cut)].copy()
SATS = sorted(data.loc[created.lt(end_cut), "satellite_name"].unique())
assert len(train) < 12657 and len(val) < 2011 and len(test) == 2595

def xgb_fit(d, cfg):
    y = d[TARGET].to_numpy()
    pos = int(y.sum())
    if pos == 0:
        raise ValueError("Training subset has no positives")
    model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                              random_state=42, n_jobs=-1,
                              scale_pos_weight=(len(y)-pos)/pos, **cfg)
    model.fit(d[FEATURES], y)
    return model

def lr_fit(d, c):
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(C=c, solver="lbfgs", max_iter=5000,
                                             random_state=42))
    model.fit(d[FEATURES], d[TARGET].to_numpy())
    return model

def predict(model, d):
    return model.predict_proba(d[FEATURES])[:, 1]

def platt_fit(y, margin):
    z = np.asarray(margin, dtype=float).reshape(-1, 1)
    return LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(z, y)

def platt_predict(cal, margin):
    z = np.asarray(margin, dtype=float).reshape(-1, 1)
    return cal.predict_proba(z)[:, 1]

def choose_threshold(y, p):
    best = (.5, -1.)
    for threshold in np.linspace(.01, .99, 197):
        score = f1_score(y, np.asarray(p) >= threshold, zero_division=0)
        if score > best[1]:
            best = (float(threshold), float(score))
    return best

def choose_rank_threshold(y, score):
    values = np.unique(np.asarray(score, dtype=float))
    values = np.r_[values, np.nextafter(values[-1], np.inf)]
    best = (float(values[0]), -1.)
    for threshold in values:
        f1 = f1_score(y, score >= threshold, zero_division=0)
        if f1 > best[1]:
            best = (float(threshold), float(f1))
    return best

def metric_row(y, p, threshold, probabilistic=True):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = p >= threshold
    out = {"roc_auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else np.nan,
           "average_precision": float(average_precision_score(y, p)) if np.unique(y).size == 2 else np.nan,
           "f1": float(f1_score(y, pred, zero_division=0)),
           "precision": float(precision_score(y, pred, zero_division=0)),
           "recall": float(recall_score(y, pred, zero_division=0)),
           "accuracy": float(accuracy_score(y, pred)),
           "predicted_positive_rate": float(np.mean(pred))}
    if probabilistic:
        q = np.clip(p, 1e-7, 1-1e-7)
        out["brier"] = float(brier_score_loss(y, q))
        out["log_loss"] = float(log_loss(y, q, labels=[0, 1]))
    else:
        out["brier"] = np.nan
        out["log_loss"] = np.nan
    return out

def nested_select(outer):
    dev = [s for s in SATS if s != outer]
    xrows, lrows = [], []
    for name, cfg in CONFIGS.items():
        ys, ps = [], []
        for inner in dev:
            tr = train.loc[~train.satellite_name.isin([outer, inner])]
            va = val.loc[val.satellite_name.eq(inner)]
            ys.extend(va[TARGET].tolist())
            ps.extend(predict(xgb_fit(tr, cfg), va).tolist())
        xrows.append({"candidate": name, "inner_ap": average_precision_score(ys, ps),
                      "inner_brier": brier_score_loss(ys, ps)})
    for c in C_GRID:
        ys, ps = [], []
        for inner in dev:
            tr = train.loc[~train.satellite_name.isin([outer, inner])]
            va = val.loc[val.satellite_name.eq(inner)]
            ys.extend(va[TARGET].tolist())
            ps.extend(predict(lr_fit(tr, c), va).tolist())
        lrows.append({"candidate_C": c, "inner_ap": average_precision_score(ys, ps),
                      "inner_brier": brier_score_loss(ys, ps)})
    xb = sorted(xrows, key=lambda z: (-z["inner_ap"], z["inner_brier"]))[0]["candidate"]
    lc = sorted(lrows, key=lambda z: (-z["inner_ap"], z["inner_brier"]))[0]["candidate_C"]
    return xb, lc, xrows, lrows

rows, selections, prediction_frames, oof_frames = [], [], [], []
for outer in SATS:
    print("Outer satellite", outer, flush=True)
    xb, lc, xsel, lsel = nested_select(outer)
    selections += ([dict(outer=outer, model="xgboost", **r) for r in xsel]
                   + [dict(outer=outer, model="logistic_regression", **r) for r in lsel])
    dev = [s for s in SATS if s != outer]
    # Cross-satellite out-of-fold scores mimic the unseen-satellite condition
    # for calibration and threshold fitting.
    oy, ox, ol, oxp, olp, oh, oa = [], [], [], [], [], [], []
    for inner in dev:
        tr = train.loc[~train.satellite_name.isin([outer, inner])]
        va = val.loc[val.satellite_name.eq(inner)]
        xmodel = xgb_fit(tr, CONFIGS[xb])
        lmodel = lr_fit(tr, float(lc))
        px_margin = xmodel.predict(va[FEATURES], output_margin=True)
        pl_margin = lmodel.decision_function(va[FEATURES])
        px = expit(px_margin); pl = expit(pl_margin)
        oy.extend(va[TARGET].tolist()); ox.extend(px_margin.tolist()); ol.extend(pl_margin.tolist())
        oxp.extend(px.tolist()); olp.extend(pl.tolist())
        oh.extend(va.horizon_hours.astype(float).tolist())
        oa.extend((va.horizon_hours + va.source_tle_age_hours).astype(float).tolist())
        oof_frames.append(pd.DataFrame({"outer_satellite": outer, "inner_satellite": inner,
                                        "y_true": va[TARGET].to_numpy(), "xgb_raw_prob": px,
                                        "xgb_margin": px_margin, "lr_raw_prob": pl,
                                        "lr_margin": pl_margin, "horizon_score": va.horizon_hours,
                                        "age_plus_horizon": va.horizon_hours + va.source_tle_age_hours}))
    oy = np.asarray(oy, dtype=int); ox = np.asarray(ox); ol = np.asarray(ol)
    oxp = np.asarray(oxp); olp = np.asarray(olp)
    xcal = platt_fit(oy, ox); lcal = platt_fit(oy, ol)
    oxcal = platt_predict(xcal, ox); olcal = platt_predict(lcal, ol)
    xraw_t, _ = choose_threshold(oy, oxp); xcal_t, _ = choose_threshold(oy, oxcal)
    lraw_t, _ = choose_threshold(oy, olp); lcal_t, _ = choose_threshold(oy, olcal)
    horizon_t, _ = choose_rank_threshold(oy, np.asarray(oh))
    span_t, _ = choose_rank_threshold(oy, np.asarray(oa))

    tr = train.loc[train.satellite_name.ne(outer)]
    va_outer = val.loc[val.satellite_name.ne(outer)]
    te = test.loc[test.satellite_name.eq(outer)]
    assert len(te) and outer not in set(tr.satellite_name) and outer not in set(va_outer.satellite_name)
    xm = xgb_fit(tr, CONFIGS[xb]); lm = lr_fit(tr, float(lc))
    xmargin = xm.predict(te[FEATURES], output_margin=True); xraw = expit(xmargin)
    xprob = platt_predict(xcal, xmargin)
    lmargin = lm.decision_function(te[FEATURES]); lraw = expit(lmargin)
    lprob = platt_predict(lcal, lmargin)
    ytest = te[TARGET].to_numpy()
    horizon = te.horizon_hours.astype(float).to_numpy()
    age_span = (te.horizon_hours + te.source_tle_age_hours).astype(float).to_numpy()
    prevalence = float(tr[TARGET].mean())
    methods = [
        ("nested_xgboost_raw", xraw, xraw_t, True),
        ("nested_xgboost_oof_platt", xprob, xcal_t, True),
        ("logistic_regression_raw", lraw, lraw_t, True),
        ("logistic_regression_oof_platt", lprob, lcal_t, True),
        ("horizon_only", horizon, horizon_t, False),
        ("age_plus_horizon", age_span, span_t, False),
        ("training_prevalence", np.full(len(te), prevalence), .5, True)]
    for name, p, threshold, probabilistic in methods:
        m = metric_row(ytest, p, threshold, probabilistic)
        rows.append({"outer_satellite": outer, "n_train": len(tr), "n_validation": len(va_outer),
                     "n_oof_calibration": len(oy), "n_test": len(te),
                     "test_positives": int(ytest.sum()), "test_prevalence": float(ytest.mean()),
                     "xgb_configuration": xb, "logistic_C": float(lc), "model": name,
                     "selected_threshold": threshold, "training_prevalence": prevalence, **m})
    prediction_frames.append(pd.DataFrame({
        "outer_satellite": outer, "source_creation": te.source_creation.astype(str).to_numpy(),
        "source_epoch": te.source_epoch.astype(str).to_numpy(),
        "horizon_hours": te.horizon_hours.to_numpy(), "y_true": ytest,
        "xgb_raw": xraw, "xgb_margin": xmargin, "xgb_oof_platt": xprob,
        "lr_raw": lraw, "lr_margin": lmargin, "lr_oof_platt": lprob, "horizon_score": horizon,
        "age_plus_horizon": age_span, "training_prevalence": prevalence}))

fold = pd.DataFrame(rows)
fold.to_csv(OUT / "nested_loso_by_satellite.csv", index=False)
pd.DataFrame(selections).to_csv(OUT / "nested_loso_inner_selection.csv", index=False)
pd.concat(prediction_frames, ignore_index=True).to_csv(OUT / "nested_loso_test_predictions.csv", index=False)
pd.concat(oof_frames, ignore_index=True).to_csv(OUT / "nested_loso_oof_calibration_predictions.csv", index=False)
summary = []
for method, group in fold.groupby("model", sort=False):
    rec = {"model": method, "outer_folds": len(group),
           "two_class_folds": int(group.roc_auc.notna().sum())}
    for col in ["roc_auc", "average_precision", "brier", "log_loss", "f1", "precision", "recall"]:
        rec[col + "_macro_mean"] = float(group[col].mean(skipna=True))
        rec[col + "_macro_sd"] = float(group[col].std(skipna=True, ddof=1)) if group[col].notna().sum() > 1 else np.nan
    group_valid = group.loc[group.average_precision.notna() & group.test_prevalence.gt(0)]
    rec["ap_lift_macro_mean"] = float((group_valid.average_precision / group_valid.test_prevalence).mean())
    rec["positive_fold_f1_mean"] = float(group.loc[group.test_positives.gt(0), "f1"].mean())
    summary.append(rec)
pd.DataFrame(summary).to_csv(OUT / "nested_loso_summary.csv", index=False)

purge = {"train_rows_original": 12657, "train_rows_after_reference_cutoff_purge": len(train),
         "train_rows_purged": 12657-len(train), "validation_rows_original": 2011,
         "validation_rows_after_reference_cutoff_purge": len(val),
         "validation_rows_purged": 2011-len(val), "test_rows": len(test),
         "ambiguous_labels_excluded": int(ambiguous.sum()),
         "train_reference_cutoff_utc": train_cut.isoformat(),
         "validation_reference_cutoff_utc": test_cut.isoformat()}
meta = {"protocol": "Nested outer satellite holdout; inner leave-one-satellite-out selection. Before model fitting, training rows whose reference_creation reaches 2025-09-01 UTC and validation rows whose reference_creation reaches 2025-11-01 UTC are purged. For each outer satellite, inner OOF predictions from the other five satellites fit Platt calibration and select F1 thresholds. Final models train on purged Jan-Aug rows from five development satellites; outer Nov-Dec labels are used only for final scoring.",
        "calibration": "Platt scaling is fitted on cross-satellite out-of-fold decision margins (XGBoost raw margins and LR decision_function); this avoids clipping small probabilities before calibration. The OOF base models are trained on four development satellites, while the final outer model is trained on five.",
        "baseline": "Regularized logistic regression on all 15 features; median imputation and standardization fitted within each training fold; C selected by pooled inner AP with Brier tie-break.",
        "ranking_baselines": "Raw requested horizon and total propagation span (TLE age + horizon); thresholds selected from cross-satellite development OOF validation predictions.",
        "probability_baseline": "Constant training-prevalence predictor; prevalence is calculated only from outer development training data.",
        "threshold_rule": "197 probability thresholds from 0.01 through 0.99; F1 maximized on cross-satellite OOF development predictions.",
        "purge": purge, "satellites": SATS, "xgb_candidates": CONFIGS,
        "logistic_C_grid": C_GRID, "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__,
        "scipy": scipy.__version__, "sgp4": sgp4.__version__,
        "matplotlib": matplotlib.__version__, "seed": 42,
        "input_sha256": {name: hashlib.sha256((DATA/name).read_bytes()).hexdigest()
                         for name in ["orap_multisatellite_labels_2025.csv", "orap_features_2025.csv"]}}
(OUT / "nested_loso_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("Nested LOSO summary\n", pd.DataFrame(summary).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("Purge metadata", purge)
