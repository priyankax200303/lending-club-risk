"""Builds and executes the analysis notebooks. Run from the project root."""
import nbformat as nbf
from nbclient import NotebookClient
def nb(cells, path):
    n = nbf.v4.new_notebook(); n.cells = [nbf.v4.new_markdown_cell(c[3:]) if c.startswith("MD:") else nbf.v4.new_code_cell(c) for c in cells]
    n.metadata.kernelspec = {"name": "python3", "display_name": "Python 3", "language": "python"}
    NotebookClient(n, timeout=1500, kernel_name="python3").execute(cwd="notebooks"); nbf.write(n, path)
setup = '''import warnings; warnings.filterwarnings("ignore")
import sys, json, os
sys.path.insert(0, "../scripts")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from common import *
pd.set_option("display.width", 150); pd.set_option("display.max_columns", 30)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e6e6e3", "font.size": 10})
ACC, MUTE, INK, BAD = "#20808D", "#9a9a94", "#2b2b2b", "#C8452F"
os.makedirs("../outputs/charts", exist_ok=True); os.makedirs("../outputs/tables", exist_ok=True)
def save_metrics(k, v):
    p = "../outputs/metrics.json"; m = json.load(open(p)) if os.path.exists(p) else {}
    m[k] = v; json.dump(m, open(p, "w"), indent=2, default=float)
'''
data = ["MD:# Data checks and study design\n\n**Goal.** Confirm what the data can support before any modelling.\n\n**Design choices.** Only fields known at application are used. Only 36-month loans issued from 2013 to 2015 are used, because later loans had not all reached maturity when the data ended.",
setup,
'''d = load(); print(d.shape, "defaults", int(d.target.sum()), "rate", round(d.target.mean(), 4), "from", d.issue_month.min().date(), "to", d.issue_month.max().date())
print("duplicate rows:", d.drop(columns=["issue_month", "grade"]).duplicated().sum())''',
"MD:## Maturity bias\n\nThe file contains resolved loans only. A 36-month loan issued in 2016 could only appear by the end of 2018 if it had already defaulted or been paid off early, so recent default rates are inflated.",
'''q = d[d.term_num == 36].groupby(d.issue_month.dt.to_period("Q")).target.agg(loans="size", default_rate="mean").reset_index()
q["issue_month"] = q.issue_month.astype(str); q = q[q.issue_month >= "2013Q1"]; q.to_csv("../outputs/tables/vintage_default_rates.csv", index=False); display(q.round(3))
fig, ax = plt.subplots(figsize=(7.4, 2.8)); ax.bar(q.issue_month, 100 * q.default_rate, color=[ACC if v < "2016Q1" else MUTE for v in q.issue_month]); ax.tick_params(axis="x", rotation=60)
ax.set_ylabel("Default rate (%)"); ax.set_title("Quarters after 2015 are not comparable: unfinished loans are missing", loc="left", fontsize=11); plt.tight_layout(); plt.savefig("../outputs/charts/vintage_default_rates.png"); plt.show()''',
"MD:## Leakage screen\n\nA field that uses information after approval usually predicts default too well. No single application field exceeds 0.60 here, which fits application-time data. FICO scores below 0.5 only because higher FICO means lower risk.",
'''from sklearn.metrics import roc_auc_score as auc
s = prepare(d); rows = []
for c in RAW_NUM + ["int_rate", "sub_grade_num"]:
    v = s[c].fillna(s[c].median()); a = auc(s.target, v); rows.append(dict(field=c, auc=a, strength=max(a, 1 - a)))
screen = pd.DataFrame(rows).sort_values("strength", ascending=False); screen.to_csv("../outputs/tables/leakage_screen.csv", index=False); display(screen.head(8).round(3))
print("Strongest single field:", screen.iloc[0].field, round(screen.iloc[0].strength, 3))''',
"MD:## Study split\n\nThe test period is later than the training period, so the score is judged on loans it could not have seen. The validation period sets calibration and cutoffs, and the test period is used once.",
'''sp = s.groupby("split").agg(loans=("target", "size"), defaults=("target", "sum"), default_rate=("target", "mean"), avg_rate=("int_rate", "mean"), first=("issue_month", "min"), last=("issue_month", "max")).reindex(["train", "validation", "test"]).reset_index()
sp.to_csv("../outputs/tables/study_split.csv", index=False); display(sp.round(4))
print(s[RAW_NUM].isna().mean().sort_values(ascending=False).head(6).round(4))
save_metrics("data", dict(rows=len(d), defaults=int(d.target.sum()), study_rows=len(s), strongest_field=screen.iloc[0].field, strongest_auc=screen.iloc[0].strength, splits=sp.drop(columns=["first", "last"]).to_dict("records")))'''
]
model = ["MD:# Default risk model\n\n**Question.** Can application data rank default risk better than Lending Club's own sub-grade?\n\n**What is left out.** The interest rate and grade come from Lending Club's own risk model, so they are held out as the benchmark. Monthly installment is also removed: on a 36-month loan, installment and loan amount together reveal the rate exactly. An early version kept it and looked better than it really was.\n\n**Rule.** The champion model is chosen on the validation period. The test period is only used to report the result.",
setup,
'''from sklearn.metrics import roc_auc_score as auc, brier_score_loss as brier
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
s = prepare(load()); tr, va, te = [s[s.split == k] for k in ("train", "validation", "test")]; print(len(tr), len(va), len(te))
num = [c for c in FEATURES if c != "purpose"]
# Proof of the leak: rate implied by payment and principal
x = s.sample(5000, random_state=1); r = x.int_rate / 1200; imp = x.loan_amnt * r / (1 - (1 + r) ** -36); print("Correlation of formula payment with actual installment:", round(np.corrcoef(imp, x.installment)[0, 1], 6))''',
"MD:## Logistic regression\n\nThis is the interpretable baseline. Skewed balances and incomes are logged, and missing values use the training median.",
'''def logp(x): return np.log1p(np.clip(x, 0, None))
logcols = ["annual_inc", "tot_cur_bal", "total_rev_hi_lim", "avg_cur_bal", "bc_open_to_buy", "tot_hi_cred_lim", "total_bc_limit", "loan_amnt", "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mths_since_recent_bc"]
other = [c for c in num if c not in logcols]
pre = make_column_transformer(
 (make_pipeline(FunctionTransformer(logp, feature_names_out="one-to-one"), SimpleImputer(strategy="median"), StandardScaler()), logcols),
 (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), other),
 (OneHotEncoder(handle_unknown="ignore"), ["purpose"]))
lr = make_pipeline(pre, LogisticRegression(max_iter=500, C=1.0)); lr.fit(tr[FEATURES], tr.target)
print("Validation AUC:", round(auc(va.target, lr.predict_proba(va[FEATURES])[:, 1]), 4))''',
"MD:## Gradient boosting\n\nA small grid is compared on the validation period, with early stopping setting the number of trees.",
'''grid = []; models = {}
for leaves, child, lam in [(7, 300, 5), (15, 300, 5), (31, 300, 5), (7, 1000, 20), (15, 1000, 20)]:
    m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=leaves, min_child_samples=child, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=lam, random_state=42, verbose=-1, n_jobs=2)
    m.fit(tr[FEATURES], tr.target, eval_set=[(va[FEATURES], va.target)], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
    a = auc(va.target, m.predict_proba(va[FEATURES])[:, 1]); grid.append(dict(num_leaves=leaves, min_child=child, reg_lambda=lam, trees=m.best_iteration_, val_auc=a)); models[(leaves, child, lam)] = m
gt = pd.DataFrame(grid); gt.to_csv("../outputs/tables/boosting_grid.csv", index=False); display(gt.round(4))
gb = models[max(models, key=lambda k: auc(va.target, models[k].predict_proba(va[FEATURES])[:, 1]))]''',
"MD:## Champion and calibration\n\nThe model with the better validation AUC becomes the champion. Default rates moved between periods, so its scores are recalibrated on the validation period with a one-feature logistic fit (Platt scaling).",
'''val = {"Logistic regression": lr.predict_proba(va[FEATURES])[:, 1], "Gradient boosting": gb.predict_proba(va[FEATURES])[:, 1]}
tst = {"Logistic regression": lr.predict_proba(te[FEATURES])[:, 1], "Gradient boosting": gb.predict_proba(te[FEATURES])[:, 1]}
trn = {"Logistic regression": lr.predict_proba(tr[FEATURES])[:, 1], "Gradient boosting": gb.predict_proba(tr[FEATURES])[:, 1]}
champ = max(val, key=lambda k: auc(va.target, val[k])); print("Champion on validation:", champ, {k: round(auc(va.target, v), 4) for k, v in val.items()})
lg = lambda p: np.log(p / (1 - p)).reshape(-1, 1); cal = LogisticRegression(C=1e6).fit(lg(val[champ]), va.target)
pv, pt = cal.predict_proba(lg(val[champ]))[:, 1], cal.predict_proba(lg(tst[champ]))[:, 1]
def ks(y, x):
    t = pd.DataFrame(dict(x=x, y=y)).groupby("x").y.agg(["sum", "size"]); b = t["sum"].cumsum() / t["sum"].sum(); g = (t["size"] - t["sum"]).cumsum() / (t["size"] - t["sum"]).sum(); return float((b - g).abs().max())
fx = FEATURES + ["int_rate", "sub_grade_num"]; gb2 = lgb.LGBMClassifier(n_estimators=gb.best_iteration_, learning_rate=0.03, num_leaves=gb.num_leaves, min_child_samples=gb.min_child_samples, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=gb.reg_lambda, random_state=42, verbose=-1, n_jobs=2).fit(tr[fx], tr.target)
gb2_val, gb2_tst = gb2.predict_proba(va[fx])[:, 1], gb2.predict_proba(te[fx])[:, 1]
y = te.target.values; gv = te.sub_grade_num.values
rows = [dict(model="Lending Club sub-grade (benchmark)", val_auc=auc(va.target, va.sub_grade_num), test_auc=auc(y, gv), test_ks=ks(y, gv), test_brier=np.nan)]
for k in tst: rows.append(dict(model=k, val_auc=auc(va.target, val[k]), test_auc=auc(y, tst[k]), test_ks=ks(y, tst[k]), test_brier=brier(y, tst[k])))
rows.append(dict(model=champ + ", calibrated (champion)", val_auc=auc(va.target, pv), test_auc=auc(y, pt), test_ks=ks(y, pt), test_brier=brier(y, pt)))
rows.append(dict(model="Combined: application data plus rate and grade", val_auc=auc(va.target, gb2_val), test_auc=auc(y, gb2_tst), test_ks=ks(y, gb2_tst), test_brier=brier(y, gb2_tst)))
tbl = pd.DataFrame(rows); tbl.to_csv("../outputs/tables/model_metrics.csv", index=False); display(tbl.round(4))
print("Train AUC", {k: round(auc(tr.target, v), 4) for k, v in trn.items()})
a2 = auc(y, gb2.predict_proba(te[fx])[:, 1]); print("Test AUC when rate and grade are added:", round(a2, 4))
v2 = gb2.predict_proba(va[fx])[:, 1]; cal2 = LogisticRegression(C=1e6).fit(lg(v2), va.target); pv2, pt2 = cal2.predict_proba(lg(v2))[:, 1], cal2.predict_proba(lg(gb2.predict_proba(te[fx])[:, 1]))[:, 1]
print("Validation AUC with rate and grade:", round(auc(va.target, pv2), 4))
rng = np.random.default_rng(7); dg = []
for _ in range(300):
    i = rng.integers(0, len(y), len(y)); dg.append(auc(y[i], pt2[i]) - auc(y[i], gv[i]))
lo2, hi2 = np.percentile(dg, [2.5, 97.5]); gap2 = auc(y, pt2) - auc(y, gv); print("AUC gap, combined minus grade:", round(gap2, 4), "95% CI", round(lo2, 4), round(hi2, 4))''',
"MD:## Combined model\n\nThe next cell block adds context for the result above. A combined model uses application data plus the rate and grade, as a lender would if it already had its own pricing.",
"MD:## Does it beat the benchmark?\n\nA bootstrap of the test period gives an interval for the AUC difference between the champion and Lending Club's sub-grade.",
'''rng = np.random.default_rng(42); diffs = []
for _ in range(300):
    i = rng.integers(0, len(y), len(y)); diffs.append(auc(y[i], pt[i]) - auc(y[i], gv[i]))
lo, hi = np.percentile(diffs, [2.5, 97.5]); gap = auc(y, pt) - auc(y, gv); print("AUC gap, champion minus grade:", round(gap, 4), "95% CI", round(lo, 4), round(hi, 4))''',
'''q = pd.qcut(pt, 10, labels=False, duplicates="drop"); c = pd.DataFrame(dict(b=q, p=pt, y=y)).groupby("b").agg(loans=("y", "size"), predicted=("p", "mean"), actual=("y", "mean")).reset_index()
c.to_csv("../outputs/tables/calibration.csv", index=False); display(c.round(3)); print("Largest decile gap, points:", round(100 * (c.predicted - c.actual).abs().max(), 2))
fig, ax = plt.subplots(figsize=(4.2, 3.8)); m = max(c.predicted.max(), c.actual.max()) * 105
ax.plot([0, m], [0, m], color=MUTE, lw=1); ax.plot(100 * c.predicted, 100 * c.actual, "o-", color=ACC); ax.set_xlabel("Predicted default rate (%)"); ax.set_ylabel("Actual default rate (%)")
ax.set_title("Predicted vs actual, test period", loc="left", fontsize=11); plt.tight_layout(); plt.savefig("../outputs/charts/calibration.png"); plt.show()''',
"MD:## What drives the score\n\nThe table shows scaled logistic coefficients, so a positive value means higher predicted risk per standard deviation. These describe the model, not cause and effect. Applicant-level contributions could also supply the reasons a lender must give a declined applicant.",
'''names = list(lr[0].get_feature_names_out()); coef = lr[1].coef_[0]
imp = pd.DataFrame(dict(feature=[n.split("__")[-1].replace("purpose_", "purpose: ") for n in names], coefficient=coef)).assign(size=lambda t: t.coefficient.abs()).sort_values("size", ascending=False).drop(columns="size")
imp.to_csv("../outputs/tables/logistic_coefficients.csv", index=False); display(imp.head(12).round(3))
sm = te.sample(30000, random_state=42); sv = gb.booster_.predict(sm[FEATURES], pred_contrib=True)[:, :-1]
shap = pd.DataFrame(dict(feature=FEATURES, mean_abs_shap=np.abs(sv).mean(0))).sort_values("mean_abs_shap", ascending=False); shap.to_csv("../outputs/tables/shap_importance.csv", index=False); display(shap.head(8).round(3))
t = imp[~imp.feature.str.startswith("purpose")].head(10).iloc[::-1]; fig, ax = plt.subplots(figsize=(6.2, 3.2)); ax.barh(t.feature, t.coefficient, color=[BAD if v > 0 else ACC for v in t.coefficient]); ax.axvline(0, color=INK, lw=1); ax.set_xlabel("Log-odds per standard deviation"); ax.set_title("Top score drivers (red raises predicted risk)", loc="left", fontsize=11); plt.tight_layout(); plt.savefig("../outputs/charts/score_drivers.png"); plt.show()''',
"MD:## Realistic timing check\n\nThe main design has a limit. Validation loans from early 2015 only finished in 2016 to 2018, after the test loans were issued, so a real lender could not have calibrated this way in late 2015. This check uses only 36-month loans issued by June 2012, which had reached maturity before July 2015, and scores the same test loans. Those older loans are fewer and often lack newer bureau fields, so it is a rough lower bound.",
'''full = load(); m = full[(full.term_num == 36) & (full.issue_month < "2012-07-01")].copy()
m.loc[(m.dti < 0) | (m.dti > 100), "dti"] = np.nan; m.loc[m.annual_inc <= 0, "annual_inc"] = np.nan
m["inq_missing"] = ((m.mths_since_recent_inq >= 999) | m.mths_since_recent_inq.isna()).astype(int); m.loc[m.mths_since_recent_inq >= 999, "mths_since_recent_inq"] = np.nan
m["bal_missing"] = m.tot_cur_bal.isna().astype(int); m["purpose"] = pd.Categorical(m.purpose, categories=te.purpose.cat.categories)
gm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=300, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=20, random_state=42, verbose=-1, n_jobs=2).fit(m[FEATURES], m.target)
pm = gm.predict_proba(te[FEATURES])[:, 1]; am = auc(y, pm); print("Matured-only training loans:", len(m), "| test AUC:", round(am, 4), "| mean predicted", round(pm.mean(), 3), "vs actual", round(y.mean(), 3))
tm = te.assign(pm=pm); k = tm.pm <= np.quantile(pm, 0.8223); vm = value_per_loan(tm[k], tm[k].target, 0.60, 0.75).sum() / 1e6; print("Value at same 82.2% approval, $M:", round(vm, 2))
pcal = gb.predict_proba(te[FEATURES])[:, 1]; print("Champion without later calibration: mean predicted", round(pcal.mean(), 3))
save_metrics("timing", dict(matured_rows=len(m), matured_auc=am, matured_value_m=vm, matured_mean_pd=float(pm.mean()), uncalibrated_mean_pd=float(pcal.mean())))''',
"MD:## Stability\n\nThe population stability index compares score distributions between training and test. Values below 0.1 are usually read as stable.",
'''def psi(a, b, n=10):
    e = np.unique(np.quantile(a, np.linspace(0, 1, n + 1))); e[0], e[-1] = -np.inf, np.inf
    x = np.histogram(a, e)[0] / len(a); z = np.histogram(b, e)[0] / len(b); return float(np.sum((z - x) * np.log((z + 1e-6) / (x + 1e-6))))
ps = psi(trn[champ], tst[champ]); print("Score PSI, train to test:", round(ps, 3))
cols = ["issue_month", "target", "loan_amnt", "installment", "int_rate", "sub_grade_num", "grade", "fico_avg"]
out = te[cols].copy(); out["pd"] = pt; out["pd_plus"] = pt2; va_o = va[cols].copy(); va_o["pd"] = pv; va_o["pd_plus"] = pv2
os.makedirs("../data/interim", exist_ok=True); out.to_parquet("../data/interim/test_scored.parquet"); va_o.to_parquet("../data/interim/validation_scored.parquet")
save_metrics("model", dict(champion=champ, metrics=tbl.to_dict("records"), train_auc={k: auc(tr.target, v) for k, v in trn.items()}, auc_with_rate=a2, combined_gap=float(gap2), combined_lo=lo2, combined_hi=hi2, gap=float(gap), gap_lo=lo, gap_hi=hi, psi=ps, cal_max_gap_pts=float(100 * (c.predicted - c.actual).abs().max())))'''
]
strat = ["MD:# Approval strategy\n\n**Question.** Which loans should a lender approve?\n\n**Value model.** A repaid loan earns scheduled interest scaled for prepayment. A defaulted loan loses a share of the amount lent. The data has no payment or recovery fields, so both are assumptions, tested below.\n\n**Rule.** Policies are chosen on the validation period and scored once on the test period.",
setup,
'''va = pd.read_parquet("../data/interim/validation_scored.parquet"); te = pd.read_parquet("../data/interim/test_scored.parquet")
LGD, INT = 0.60, 0.75
def real(x, lgd=LGD, i=INT): return value_per_loan(x, x.target, lgd, i)
def exp(x, col="pd", lgd=LGD, i=INT): return value_per_loan(x, x[col], lgd, i)
def summary(x, keep, name, lgd=LGD, i=INT):
    k = x[keep]; v = real(k, lgd, i).sum()
    return dict(policy=name, approval_rate=100 * keep.mean(), default_rate=100 * k.target.mean(), avg_rate=(k.int_rate * k.loan_amnt).sum() / k.loan_amnt.sum(), lent_m=k.loan_amnt.sum() / 1e6, credit_loss_m=lgd * k.loan_amnt[k.target == 1].sum() / 1e6, value_m=v / 1e6, return_pct=100 * v / k.loan_amnt.sum())
print("Approve all, test:", summary(te, pd.Series(True, index=te.index), "all"))''',
"MD:## Cutoff curve\n\nFor each maximum predicted default rate, what happens to volume, defaults and value?",
'''rows = []
for c in np.arange(0.04, 0.42, 0.01):
    for n, x in (("validation", va), ("test", te)):
        rows.append(dict(sample=n, cutoff=c, **summary(x, x["pd"] <= c, f"{c:.2f}")))
cv = pd.DataFrame(rows).drop(columns="policy"); cv.to_csv("../outputs/tables/cutoff_curve.csv", index=False)
vc = cv[cv["sample"] == "validation"]; best = float(vc.loc[vc.value_m.idxmax(), "cutoff"]); print("Best validation cutoff:", round(best, 2)); display(cv[(cv["sample"] == "test") & cv.cutoff.round(2).isin([.08, .12, .16, .2, .24, .28, .32, .36])].round(2))
t = cv[cv["sample"] == "test"]; fig, ax = plt.subplots(figsize=(6.6, 3)); ax.plot(t.approval_rate, t.value_m, color=ACC, lw=2); ax.scatter([t.loc[(t.cutoff - best).abs().idxmin(), "approval_rate"]], [t.loc[(t.cutoff - best).abs().idxmin(), "value_m"]], color=BAD, zorder=3)
ax.set_xlabel("Approval rate (%)"); ax.set_ylabel("Portfolio value ($M, assumed)"); ax.set_title("Value rises, then flattens and falls as approvals widen", loc="left", fontsize=11); plt.tight_layout(); plt.savefig("../outputs/charts/cutoff_curve.png"); plt.show()''',
"MD:## Policy comparison on the test period\n\nThe grade policy approves the safest sub-grades up to the same approval rate. The expected value rules approve a loan when expected value is positive, which counts its interest as well as its risk. The combined rows use the model that also sees the rate and grade. The recommended policy is the one with the best validation value, not the best test value.",
'''def same_rate_grade(x, r):
    o = x.sort_values(["sub_grade_num", "int_rate"]); k = pd.Series(False, index=x.index); k.loc[o.index[: int(r * len(x))]] = True; return k
def best_cut(col):
    return max(np.arange(0.04, 0.42, 0.01), key=lambda c: real(va[va[col] <= c]).sum())
cp = best_cut("pd_plus"); print("Combined model cutoff:", round(cp, 2))
def policies(x):
    m = x["pd"] <= best
    return {"Approve all": pd.Series(True, index=x.index), "Grade policy, same approval as score": same_rate_grade(x, m.mean()), "Application score cutoff": m, "Application score, expected value rule": exp(x) > 0, "Combined score cutoff": x["pd_plus"] <= cp, "Combined score, expected value rule": exp(x, "pd_plus") > 0}
pv = policies(va); pt = policies(te)
tb = pd.DataFrame([summary(te, k, n) for n, k in pt.items()]); tb["validation_value_m"] = [real(va[k]).sum() / 1e6 for k in pv.values()]; tb["value_vs_all_m"] = tb.value_m - tb.value_m.iloc[0]; tb["loss_vs_all_pct"] = 100 * (tb.credit_loss_m / tb.credit_loss_m.iloc[0] - 1)
tb.to_csv("../outputs/tables/policy_comparison.csv", index=False); display(tb.round(2))
rec = tb.iloc[tb.validation_value_m.values.argmax()].policy; print("Recommended on validation:", rec)
rng = np.random.default_rng(42); a = real(te).values
def boot(k1, k2):
    b, c = pt[k1].values, pt[k2].values; d = []
    for _ in range(1000):
        i = rng.integers(0, len(a), len(a)); d.append((a[i] * b[i]).sum() - (a[i] * c[i]).sum())
    return ((a * b).sum() - (a * c).sum()) / 1e6, *(np.percentile(d, [2.5, 97.5]) / 1e6)
b1 = boot("Application score cutoff", "Grade policy, same approval as score"); b2 = boot(rec, "Approve all"); print("Score minus grade:", np.round(b1, 2), "| Recommended minus approve all:", np.round(b2, 2))''',
"MD:## Why does a tied AUC produce more value?\n\nThe score ignores the interest rate, so at the same approval rate it keeps more higher-rate loans than the grade policy does. The table splits the value gap into interest earned and credit losses. Higher-rate loans may also carry risks this simple value model misses, such as a higher chance of early default, so this gap should not be read as better risk selection.",
'''def parts(k):
    x = te[k]; return dict(interest_m=(value_per_loan(x, 0 * x.target, LGD, INT).sum() - value_per_loan(x[x.target == 1], 0 * x.target[x.target == 1], LGD, INT).sum()) / 1e6, credit_loss_m=LGD * x.loan_amnt[x.target == 1].sum() / 1e6)
pa, pb = parts(pt["Application score cutoff"]), parts(pt["Grade policy, same approval as score"]); dcmp = pd.DataFrame([dict(item=k, score=pa[k], grade=pb[k], difference=pa[k] - pb[k]) for k in pa]); dcmp.to_csv("../outputs/tables/value_gap_decomposition.csv", index=False); display(dcmp.round(2))''',
"MD:## Sensitivity\n\nThe assumptions drive the money figures. The first table re-tunes the cutoff for each scenario, which shows the best a lender could do if it knew the true assumptions. The second keeps the recommended cutoffs fixed, which is the honest test of a single rule.",
'''g = []
for lgd in (0.45, 0.60, 0.75):
    for i in (0.60, 0.75, 0.90):
        cols = {}
        for n, col in (("score", "pd"), ("combined", "pd_plus")):
            c = max(np.arange(0.04, 0.42, 0.01), key=lambda z: real(va[va[col] <= z], lgd, i).sum()); cols[n] = (c, te[col] <= c)
        ev = exp(te, "pd_plus", lgd, i) > 0; kp = same_rate_grade(te, cols["score"][1].mean())
        g.append(dict(lgd=lgd, interest_share=i, score_cutoff=cols["score"][0], approval_rate=100 * cols["score"][1].mean(), all_m=real(te, lgd, i).sum() / 1e6, grade_m=real(te[kp], lgd, i).sum() / 1e6, score_m=real(te[cols["score"][1]], lgd, i).sum() / 1e6, combined_m=real(te[cols["combined"][1]], lgd, i).sum() / 1e6, combined_ev_m=real(te[ev], lgd, i).sum() / 1e6))
sg = pd.DataFrame(g); sg["best_policy"] = sg[["all_m", "grade_m", "score_m", "combined_m", "combined_ev_m"]].idxmax(axis=1); sg.to_csv("../outputs/tables/sensitivity_retuned.csv", index=False); display(sg.round(2))
fx = []
for lgd in (0.45, 0.55, 0.60, 0.75):
    for i in (0.60, 0.75, 0.90):
        fx.append(dict(lgd=lgd, interest_share=i, all_m=real(te, lgd, i).sum() / 1e6, score_fixed_m=real(te[te["pd"] <= best], lgd, i).sum() / 1e6, combined_fixed_m=real(te[te["pd_plus"] <= cp], lgd, i).sum() / 1e6))
ff = pd.DataFrame(fx); ff["score_gain_m"] = ff.score_fixed_m - ff.all_m; ff["combined_gain_m"] = ff.combined_fixed_m - ff.all_m; ff.to_csv("../outputs/tables/sensitivity_fixed_cutoff.csv", index=False); display(ff.round(2))
print("Fixed score cutoff beats approve all:", int((ff.score_gain_m > 0).sum()), "of", len(ff), "| fixed combined:", int((ff.combined_gain_m > 0).sum()), "of", len(ff))''',
"MD:## Is the pricing right?\n\nDoes the rate charged cover risk in each grade? This is realised value by Lending Club grade on the test period, using the base assumptions.",
'''gr = te.assign(v=real(te)).groupby("grade").agg(loans=("target", "size"), default_rate=("target", "mean"), predicted=("pd", "mean"), predicted_combined=("pd_plus", "mean"), avg_rate=("int_rate", "mean"), lent=("loan_amnt", "sum"), value=("v", "sum")).reset_index(); gr["return_pct"] = 100 * gr.value / gr.lent; gr["default_rate"] *= 100; gr["predicted"] *= 100; gr["predicted_combined"] *= 100
for lgd, i in ((0.45, 0.75), (0.55, 0.75), (0.60, 0.90), (0.75, 0.75)):
    gr[f"ret_lgd{int(lgd*100)}_int{int(i*100)}"] = te.assign(v=real(te, lgd, i)).groupby("grade").v.sum().values / te.groupby("grade").loan_amnt.sum().values * 100
gr.to_csv("../outputs/tables/grade_economics.csv", index=False); display(gr.round(2))
save_metrics("strategy", dict(lgd=LGD, interest_share=INT, cutoff=best, combined_cutoff=cp, recommended=rec, policies=tb.to_dict("records"), score_minus_grade=list(b1), rec_minus_all=list(b2), best_counts=sg.best_policy.value_counts().to_dict(), fixed_score_wins=int((ff.score_gain_m > 0).sum()), fixed_combined_wins=int((ff.combined_gain_m > 0).sum()), fixed_n=len(ff)))'''
]
nb(data, "notebooks/01_data_checks.ipynb"); nb(model, "notebooks/02_default_model.ipynb"); nb(strat, "notebooks/03_approval_strategy.ipynb")
