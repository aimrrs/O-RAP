from pathlib import Path
import sys
import pandas as pd

fold = pd.read_csv(Path(sys.argv[1]) / "nested_loso_by_satellite.csv")
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
base = fold.loc[fold.model.eq("training_prevalence")].set_index("outer_satellite")
counts = fold.loc[fold.model.eq("nested_xgboost_raw"), ["outer_satellite", "n_test", "test_positives"]].set_index("outer_satellite")
rows=[]
for sat, row in counts.iterrows():
    n=int(row.n_test); pos=int(row.test_positives)
    f1=2*pos/(n+pos) if pos else 0.0
    rows.append({"outer_satellite":sat,"n_test":n,"test_positives":pos,
                 "threshold_rule":"predict positive for every sample",
                 "predicted_positive_rate":1.0,"precision":pos/n,
                 "recall":1.0 if pos else 0.0,"f1":f1})
allpos=pd.DataFrame(rows)
allpos.to_csv(out/"nested_loso_flag_all_by_satellite.csv",index=False)
models=["nested_xgboost_raw","nested_xgboost_oof_platt",
        "logistic_regression_raw","logistic_regression_oof_platt"]
skill=[]
for model in models:
    g=fold.loc[fold.model.eq(model)].set_index("outer_satellite")
    for sat,r in g.iterrows():
        null=float(base.loc[sat,"brier"])
        score=float(r.brier)
        skill.append({"outer_satellite":sat,"model":model,
                      "model_brier":score,"training_prevalence_null_brier":null,
                      "brier_skill_score":1-score/null})
skill=pd.DataFrame(skill)
skill.to_csv(out/"nested_loso_brier_skill_by_satellite.csv",index=False)
skill.groupby("model",as_index=False).brier_skill_score.mean().to_csv(
    out/"nested_loso_brier_skill_summary.csv",index=False)
print("Flag-all macro F1:",allpos.f1.mean())
print(skill.groupby("model").brier_skill_score.mean().to_string())
