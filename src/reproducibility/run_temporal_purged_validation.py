from pathlib import Path
import hashlib, json, platform, sys
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.metrics import (accuracy_score, average_precision_score,
    brier_score_loss, f1_score, log_loss, precision_score, recall_score,
    roc_auc_score)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

root = Path(sys.argv[1])
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
labels = pd.read_csv(root / "orap_multisatellite_labels_2025.csv")
features = pd.read_csv(root / "orap_features_2025.csv")
for c in ["source_creation", "source_epoch", "reference_creation"]:
    labels[c] = pd.to_datetime(labels[c], format="mixed", utc=True)
amb = (labels.norad_id.eq(25544)
       & labels.source_creation.eq(pd.Timestamp("2025-01-25T19:06:24Z"))
       & labels.source_epoch.eq(pd.Timestamp("2025-01-25T04:55:56.509824Z")))
labels = labels.loc[~amb].reset_index(drop=True)
assert len(labels) == len(features)
for c in ["norad_id", "satellite_name", "horizon_hours"]:
    assert labels[c].equals(features[c])
assert (labels.source_tle_age_hours-features.source_tle_age_hours).abs().max() < 1e-7
d = features.copy()
d["source_creation"] = labels.source_creation
d["reference_creation"] = labels.reference_creation
d["unreliable_1km"] = d.unreliable_1km.astype(int)
trc = pd.Timestamp("2025-09-01", tz="UTC"); vac = pd.Timestamp("2025-11-01", tz="UTC")
train = d.loc[d.source_creation.lt(trc) & d.reference_creation.lt(trc)].copy()
val = d.loc[d.source_creation.ge(trc) & d.source_creation.lt(vac) & d.reference_creation.lt(vac)].copy()
test = d.loc[d.source_creation.ge(vac) & d.source_creation.lt(pd.Timestamp("2026-01-01", tz="UTC"))].copy()
features_x = [c for c in features.columns if c not in {"norad_id", "satellite_name", "unreliable_1km"}]
ytr=train.unreliable_1km.to_numpy(); yv=val.unreliable_1km.to_numpy(); yt=test.unreliable_1km.to_numpy()
pos=int(ytr.sum()); model=xgb.XGBClassifier(n_estimators=300,max_depth=4,learning_rate=.05,
    subsample=.8,colsample_bytree=.8,objective="binary:logistic",eval_metric="logloss",
    scale_pos_weight=(len(ytr)-pos)/pos,random_state=42,n_jobs=-1)
model.fit(train[features_x],ytr,eval_set=[(val[features_x],yv)],verbose=False)
pv=model.predict_proba(val[features_x])[:,1]; pt=model.predict_proba(test[features_x])[:,1]

def prob_threshold(y,p):
    best=(.5,-1.)
    for t in np.linspace(.01,.99,197):
        score=f1_score(y,p>=t,zero_division=0)
        if score>best[1]: best=(float(t),float(score))
    return best
def rank_threshold(y,s):
    candidates=np.unique(np.asarray(s,dtype=float)); candidates=np.r_[candidates,np.nextafter(candidates[-1],np.inf)]
    best=(float(candidates[0]),-1.)
    for t in candidates:
        score=f1_score(y,np.asarray(s)>=t,zero_division=0)
        if score>best[1]: best=(float(t),float(score))
    return best
def macro_within_satellite(y, score, sat):
    rows=[]
    frame=pd.DataFrame({"y":np.asarray(y,dtype=int),"score":np.asarray(score,dtype=float),"sat":np.asarray(sat)})
    for _, g in frame.groupby("sat"):
        if g.y.nunique()==2:
            rows.append((roc_auc_score(g.y,g.score),average_precision_score(g.y,g.score)))
    if not rows:
        return np.nan,np.nan,0
    a=np.asarray(rows,dtype=float)
    return float(a[:,0].mean()),float(a[:,1].mean()),len(rows)

# Select LR regularization on forward-chaining folds within Jan-Aug.
train_sorted=train.sort_values("source_creation").copy()
times=np.sort(train_sorted.source_creation.unique())
time_cv=TimeSeriesSplit(n_splits=5)
lr_candidates=[]
for c in [.01,.1,1.,10.,100.]:
    fold_ap=[]
    for ti,vi in time_cv.split(times):
        tr_times=times[ti]; va_times=times[vi]
        tr_fold=train_sorted.loc[train_sorted.source_creation.isin(tr_times)]
        va_fold=train_sorted.loc[train_sorted.source_creation.isin(va_times)]
        lr_cv=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
                            LogisticRegression(C=c,solver="lbfgs",max_iter=5000,random_state=42))
        lr_cv.fit(tr_fold[features_x],tr_fold.unreliable_1km)
        fold_ap.append(average_precision_score(va_fold.unreliable_1km,
                                               lr_cv.predict_proba(va_fold[features_x])[:,1]))
    lr_candidates.append((float(np.mean(fold_ap)),c))
lr_cv_ap,lr_c=max(lr_candidates,key=lambda z:(z[0],-z[1]))
lr_model=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
                        LogisticRegression(C=lr_c,solver="lbfgs",max_iter=5000,random_state=42))
lr_model.fit(train[features_x],ytr)
plr=lr_model.predict_proba(val[features_x])[:,1]
tlr,vlr=prob_threshold(yv,plr)
plrt=lr_model.predict_proba(test[features_x])[:,1]
train_prevalence=float(np.mean(ytr))
methods=[("xgboost_raw",pv,pt,prob_threshold(yv,pv),True),
         ("horizon_only",val.horizon_hours.to_numpy(float),test.horizon_hours.to_numpy(float),rank_threshold(yv,val.horizon_hours.to_numpy(float)),False),
         ("age_plus_horizon",(val.source_tle_age_hours+val.horizon_hours).to_numpy(float),(test.source_tle_age_hours+test.horizon_hours).to_numpy(float),rank_threshold(yv,(val.source_tle_age_hours+val.horizon_hours).to_numpy(float)),False),
         ("logistic_regression_raw",plr,plrt,(tlr,vlr),True),
         ("training_prevalence",np.full(len(val),train_prevalence),np.full(len(test),train_prevalence),(0.5,0.0),True),
         ("all_positive_rule",np.ones(len(val)),np.ones(len(test)),(0.0,0.0),False)]
rows=[]
for name,_,score,(threshold,valf1),prob in methods:
    pred=score>=threshold
    within_auc,within_ap,within_n=macro_within_satellite(yt,score,test.satellite_name)
    rows.append({"model":name,"threshold":threshold,"validation_f1":valf1,"n_train":len(train),"n_validation":len(val),"n_test":len(test),"test_positives":int(yt.sum()),
      "accuracy":accuracy_score(yt,pred),"precision":precision_score(yt,pred,zero_division=0),"recall":recall_score(yt,pred,zero_division=0),"f1":f1_score(yt,pred,zero_division=0),
      "roc_auc":roc_auc_score(yt,score),"average_precision":average_precision_score(yt,score),
      "within_satellite_folds":within_n,"within_satellite_roc_auc":within_auc,"within_satellite_average_precision":within_ap,
      "brier":brier_score_loss(yt,score) if prob else np.nan,"log_loss":log_loss(yt,np.clip(score,1e-7,1-1e-7),labels=[0,1]) if prob else np.nan,
      "test_positive_rate":float(yt.mean()),"predicted_positive_rate":float(pred.mean()),
      "selected_C":lr_c if name=="logistic_regression_raw" else np.nan,"training_cv_ap":lr_cv_ap if name=="logistic_regression_raw" else np.nan})
pd.DataFrame(rows).to_csv(out/"chronological_purged_validation_threshold_metrics.csv",index=False)
pd.DataFrame({"source_creation":test.source_creation.astype(str),"satellite_name":test.satellite_name,"y_true":yt,"xgb_probability":pt}).to_csv(out/"chronological_test_predictions.csv",index=False)
meta={"protocol":"Chronological Jan-Aug training, Sep-Oct validation, Nov-Dec test; train rows with reference_creation on/after Sep 1 purged; validation rows with reference_creation on/after Nov 1 purged. XGBoost configuration is fixed to 300 estimators, depth 4, learning rate 0.05. LR C is selected by forward-chaining AP on five Jan-Aug training folds. Probability thresholds are chosen on validation F1; horizon and total-span thresholds are chosen on validation F1. Test probabilities are raw and uncalibrated. Within-satellite ranking metrics are macro means over test satellites with both classes.",
      "rows":{"train":len(train),"validation":len(val),"test":len(test),"train_purged":12657-len(train),"validation_purged":2011-len(val)},
      "features":features_x,"threshold_grid":[.01,.99,197],"lr_C_grid":[.01,.1,1,10,100],
      "lr_forward_folds":5,"lr_training_cv_ap":lr_cv_ap,"seed":42,
      "python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,
      "scikit_learn":sklearn.__version__,"xgboost":xgb.__version__,
      "input_sha256":{name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                      for name in ["orap_multisatellite_labels_2025.csv","orap_features_2025.csv"]}}
(out/"chronological_purged_validation_threshold_metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
print(pd.DataFrame(rows).to_string(index=False,float_format=lambda x:f"{x:.4f}"));print(meta["rows"])
