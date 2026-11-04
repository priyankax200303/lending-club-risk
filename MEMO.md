# Recommendation: pilot a combined score, and measure real losses first

**Decision needed:** how to set approval rules and pricing for 36-month personal loans.

**Recommendation:** combine the existing grade with application data, then pilot a cutoff at 24% predicted default. Measure actual loss and prepayment rates before setting the final cutoff, because the gain depends on them.

## What the data shows
- **A cutoff works under most assumptions.** Keeping 85.3% of funded loans lowers the default rate from 14.7% to 11.9%. Credit losses fall 31%, from $178.3M to $122.9M. Lending falls 14%, so loss per dollar lent falls about 20%, from 8.6% to 6.9%.
- **The new score alone does not beat the grade.** Test AUC is 0.682 vs 0.681. A grade cutoff at the same 82.2% kept gives nearly the same credit loss.
- **Together they are stronger.** Combining them lifts AUC to 0.700, a gain of 0.018 (95% interval 0.016 to 0.020).
- **Lower grades look underpriced.** Grades D to G are 13.4% of lending and lose $6.3M under base assumptions, before funding and operating costs. F and G lose money in every case tested. D turns positive at a 55% loss.

## Value
Base assumptions: 60% loss on default, and 75% of scheduled interest earned on repaid loans. The data has no repayment fields, so these need replacing with real figures.

| Policy | Funded loans kept | Default rate | Credit loss | Value |
|---|---|---|---|---|
| Approve all | 100% | 14.7% | $178.3M | $50.8M |
| Grade policy | 82.2% | 12.0% | $119.7M | $57.0M |
| Application score cutoff | 82.2% | 11.9% | $119.6M | $65.7M |
| Combined score cutoff | 85.3% | 11.9% | $122.9M | $65.8M |

- **Fixed cutoffs:** the combined cutoff beat approve-all in 11 of 12 scenarios. It lost $7.0M at a 45% loss with 90% of interest earned.
- **Timing:** using only outcomes known at the time, AUC falls to 0.652 and value to $59.5M, so expect less than the table shows.
- **Grade comparison:** the score's $8.7M gain over the grade policy is not a risk win. $8.6M of it comes from keeping higher-rate loans, with credit losses unchanged.

## Ranked actions
1. **Measure actual loss and prepayment rates.** These decide whether a cutoff adds value and where it belongs.
2. **Add funding and operating costs.** Current returns exclude them, so true margins are lower.
3. **Pilot the combined cutoff.** Keep a small random group of otherwise declined applicants, to measure what approved-only data cannot show.
4. **Review pricing for grades D to G.** Test higher rates or lower limits, since a price rise can also change who accepts.
5. **Run fair-lending checks before any live use.** The data has no protected attributes, so this analysis could not test for bias.

## What would change this view
- Real losses near 45% with high interest retained, where a 24% cutoff stops paying.
- A downturn period where the score ranks risk less well.
- Pilot results showing declined applicants repay better than predicted.
