"""Recalculates headline figures from the raw and scored files without using common.py. Run from the project root."""
import json, pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score as auc
m = json.load(open("outputs/metrics.json")); p = pd.read_csv("outputs/tables/policy_comparison.csv").set_index("policy")
d = pd.concat([pd.read_parquet("data/raw/train_fe.parquet", columns=["target", "term_num", "issue_date"]), pd.read_parquet("data/raw/test_fe.parquet", columns=["target", "term_num", "issue_date"])])
d["t"] = pd.to_datetime(d.issue_date); s = d[(d.term_num == 36) & (d.t >= "2013-01-01") & (d.t < "2016-01-01")]
t = pd.read_parquet("data/interim/test_scored.parquet")
i = (36 * t.installment - t.loan_amnt).clip(lower=0) * .75; v = np.where(t.target == 1, -.6 * t.loan_amnt, i); k = (t["pd"] <= m["strategy"]["cutoff"]).values
r = [("all rows", len(d), m["data"]["rows"]), ("study rows", len(s), m["data"]["study_rows"]), ("test defaults", 100 * t.target.mean(), p.loc["Approve all", "default_rate"]),
     ("grade AUC", auc(t.target, t.sub_grade_num), m["model"]["metrics"][0]["test_auc"]), ("score AUC", auc(t.target, t["pd"]), m["model"]["metrics"][3]["test_auc"]),
     ("approve all value", v.sum() / 1e6, p.loc["Approve all", "value_m"]), ("score value", v[k].sum() / 1e6, p.loc["Application score cutoff", "value_m"]), ("score approval", 100 * k.mean(), p.loc["Application score cutoff", "approval_rate"])]
for n, a, b in r:
    print(f"{n:18s} recalculated={a:,.4f} reported={b:,.4f} diff={abs(a-b):.6f}")
    assert abs(a - b) < 0.01, f"{n} does not match"
print("All checks passed")
