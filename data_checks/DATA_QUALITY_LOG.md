# Data quality log

Details are in `notebooks/01_data_checks.ipynb`.

| # | Issue found | Size | Treatment |
|---|---|---|---|
| 1 | The source is a third-party processed copy, not Lending Club's raw file | All 1,345,310 rows | Disclosed. Its fitted winsorised and log features are ignored. `fico_avg`, `loan_to_income` and `sub_grade_num` are derived by the source and are used |
| 2 | Resolved loans only, so recent vintages are biased | Monthly 36-month default rate climbs from March 2016, and reaches 22.4% for 2016 Q2 | Study uses 36-month loans issued 2013 to 2015 |
| 3 | Monthly installment reveals the interest rate | Inverting the payment formula recovers the rate within about 0.02 points | Removed installment and installment-to-income from the score |
| 4 | Rate and sub-grade are Lending Club's own risk view | All rows | Excluded from the score and used as the benchmark |
| 5 | ZIP region can act as a proxy for protected groups | 943 regions | Excluded |
| 6 | Invalid debt-to-income values, below 0 or above 100 | 5 study rows | Set to missing |
| 7 | Annual income of zero | 2 study rows | Set to missing |
| 8 | Inquiry recency uses 999 for "none" | About 10.6% of the study is missing after cleaning | Set to missing and flagged |
| 9 | No payment, recovery or prepayment fields | All rows | Loss and interest shares are assumptions with sensitivity tests |
| 10 | Approved loans only | Rejected applicants are absent | Disclosed. Results apply only to the approved population |
| 11 | Unresolved loans are absent from 2015 too | Unknown count, likely small for 36-month loans issued by December 2015 | Disclosed |
| 12 | Validation outcomes finished after test loans were issued | All validation loans | Disclosed, with a matured-loans check in the model notebook |
