"""Shared data preparation. Keeps application-time fields only and defines the out-of-time split."""
import numpy as np, pandas as pd
RAW_NUM = ["loan_amnt", "annual_inc", "dti", "tot_cur_bal", "total_rev_hi_lim", "avg_cur_bal", "bc_open_to_buy",
           "bc_util", "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mths_since_recent_bc",
           "mths_since_recent_inq", "tot_hi_cred_lim", "total_bc_limit", "fico_avg"]
# Excluded on purpose: int_rate and sub_grade (Lending Club's own risk view, used as the benchmark),
# installment and installment_to_income (with loan amount and term they reveal the interest rate exactly),
# zip_region (a geographic field that can act as a proxy for protected groups), and the source's fitted features.
FEATURES = RAW_NUM + ["loan_to_income", "inq_missing", "bal_missing", "purpose"]

def load():
    d = pd.concat([pd.read_parquet("../data/raw/train_fe.parquet"), pd.read_parquet("../data/raw/test_fe.parquet")], ignore_index=True)
    d["issue_month"] = pd.to_datetime(d["issue_date"])
    d["grade"] = ((d.sub_grade_num - 1) // 5).map(dict(enumerate("ABCDEFG")))
    return d

def prepare(d):
    """36-month loans issued 2013 to 2015. Later 36-month loans had not all matured when the data ended."""
    s = d[(d.term_num == 36) & (d.issue_month >= "2013-01-01") & (d.issue_month < "2016-01-01")].copy()
    s.loc[(s.dti < 0) | (s.dti > 100), "dti"] = np.nan
    s.loc[s.annual_inc <= 0, "annual_inc"] = np.nan
    s["inq_missing"] = (s.mths_since_recent_inq >= 999).astype(int) | s.mths_since_recent_inq.isna().astype(int)
    s.loc[s.mths_since_recent_inq >= 999, "mths_since_recent_inq"] = np.nan
    s["bal_missing"] = s.tot_cur_bal.isna().astype(int)
    s["purpose"] = s.purpose.astype("category")
    s["split"] = np.select([s.issue_month < "2015-01-01", s.issue_month < "2015-07-01"], ["train", "validation"], "test")
    return s

def value_per_loan(s, default, lgd, interest_share):
    """Realised or expected value. Interest is scheduled interest scaled for prepayment. Both inputs are assumptions."""
    interest = (36 * s.installment - s.loan_amnt).clip(lower=0) * interest_share
    return (1 - default) * interest - default * lgd * s.loan_amnt
