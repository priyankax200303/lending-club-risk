# Lending Club default risk and approval strategy

**Verdict:** a default model on application data ties Lending Club's own sub-grade (AUC 0.682 vs 0.681), so it is useful but not better on its own. Combining the two adds real signal (AUC 0.700). A cutoff on the combined score cuts credit losses by 31% on 85% of funded loans, or about 20% per dollar lent. The size of the gain depends on loss and prepayment assumptions, so the next step is a pilot using real repayment data.

**Scope:** 546,018 36-month loans issued from 2013 to 2015. Built in Python with logistic regression and gradient boosting.

> Dollar value figures depend on assumed loss and prepayment rates, because the data has no repayment fields. They show direction and scale, and are not realised profit.

## Results

### Model, test period (July to December 2015)
| Model | Test AUC | KS |
|---|---|---|
| Lending Club sub-grade (benchmark) | 0.681 | 0.267 |
| Logistic regression | 0.674 | 0.251 |
| Gradient boosting, application data | 0.682 | 0.262 |
| Combined: application data plus rate and grade | 0.700 | 0.290 |

- Application score minus grade: +0.0002 AUC (95% interval -0.003 to +0.003), which is a tie.
- Combined minus grade: +0.018 AUC (95% interval 0.016 to 0.020).
- Calibration: the largest gap across score deciles is 1.2 points, after recalibrating on early 2015 outcomes.
- Stability: score PSI from training to test is 0.002.
- Weak spot: the application score underestimates default in grades D to G, for example 21.6% predicted vs 26.4% actual in D. The combined model is closer, at 24.8%.

### Approval policy, test period
Base assumptions: 60% of the amount lent is lost on default, and 75% of scheduled interest is earned on repaid loans.

| Policy | Funded loans kept | Default rate | Credit loss | Value |
|---|---|---|---|---|
| Approve all | 100% | 14.7% | $178.3M | $50.8M |
| Grade policy, 82.2% kept | 82.2% | 12.0% | $119.7M | $57.0M |
| Application score, predicted default up to 22% | 82.2% | 11.9% | $119.6M | $65.7M |
| Combined score, predicted default up to 24% | 85.3% | 11.9% | $122.9M | $65.8M |
| Combined score, expected value rule | 79.4% | 11.6% | $112.1M | $66.5M |

- The 31% to 33% fall in credit losses includes less lending. Loss per dollar lent falls from 8.6% to 6.9%.
- Compared with approving everything, the application score cutoff adds $14.9M (95% interval $13.3M to $16.4M). That interval covers sampling noise only.
- With fixed cutoffs, both scores beat approve-all in 11 of 12 assumption scenarios. At a 45% loss with 90% of interest earned, the score cutoff loses $8.5M, and at 45% and 75% it is about break-even.
- The score beats the grade policy by $8.7M, but $8.6M of that is interest mix and credit losses are almost identical. This is not better risk selection.
- Value was chosen on the validation period, where the three best rules fell within $0.6M. That is too close to separate them. The combined score is preferred because it ranks better on validation (0.692 vs 0.675) and is closer to actual default in lower grades.

### Realistic timing check
Early-2015 outcomes were not known when late-2015 loans were issued. A model trained only on 44,063 loans that had matured by mid-2015 scored AUC 0.652, with value of $59.5M at the same 82.2% kept. Those older loans lack some newer bureau fields, so this is a rough lower bound, but real-world results would sit below the main figures.

### Pricing by grade
Return is the 36-month cumulative figure on amount lent, under the base assumptions. It excludes funding costs, operating costs, fees and discounting.

| Grade | Default rate | Average rate | Return | Return at 45% loss |
|---|---|---|---|---|
| A | 5.1% | 6.8% | 4.7% | 5.5% |
| B | 11.8% | 9.9% | 3.5% | 5.3% |
| C | 19.8% | 13.2% | 1.0% | 4.0% |
| D | 26.4% | 16.6% | -1.1% | 3.0% |
| E | 33.5% | 19.0% | -4.8% | 0.3% |
| F | 43.1% | 23.3% | -10.5% | -3.8% |
| G | 53.3% | 27.3% | -22.3% | -13.2% |

Grades F and G are negative in every case tested. Grade D turns positive at a 55% loss, so its result depends on the assumption. After funding and operating costs, more grades would fall below an acceptable return.

## Design decisions
- **Out-of-time split:** train on 2013 to 2014, tune and calibrate on January to June 2015, and report on July to December 2015.
- **Maturity:** 36-month loans after 2015 are excluded. Default rates start rising in March 2016, because unfinished loans are missing.
- **Leakage:** installment is removed, because with loan amount and term it reveals the interest rate. An early run that kept it appeared to beat the grade.
- **Test period use:** early runs did see test results, and that exposed the leak and changed the design. The final results are not a pure first look. Validation also served early stopping, calibration and cutoff choice.
- **Fairness:** ZIP region is excluded. No protected attributes are in the data, so a fair-lending test was not possible.

## Limits
- The data is a third-party processed copy of Lending Club's resolved loans.
- It covers funded loans only, so performance on rejected applicants is unknown.
- All results come from one period, so they may not hold through a downturn.
- Default is treated as a fixed loss share, with no timing. Early and late defaults lose different amounts.
- The score has not been tested for fair-lending compliance and is not ready for real decisions.

## Repository
| Path | Contents |
|---|---|
| `notebooks/` | Data checks, default model and approval strategy |
| `scripts/` | Preparation, rebuild and reconciliation code |
| `data_checks/` | Data quality log |
| `outputs/` | Tables, charts and metrics |
| `MEMO.md` | One-page recommendation |

## Run it
```bash
pip install -r requirements.txt
bash scripts/get_data.sh
bash scripts/run_all.sh
```

## Sources
- [Processed Lending Club copy](https://github.com/YonatanGafri/lending-club-mlops)
