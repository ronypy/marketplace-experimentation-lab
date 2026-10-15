# Marketplace Experimentation Lab

Analyzed a 64,000-customer randomized promotion test with SRM checks, Holm-corrected inference, and power analysis. It is divided into 2 parts. 

**Part A** analyzes a real randomized email test on 64,000 customers: which promotion to ship, how much it's worth, and how CUPED/CUPAC would have let us run it with fewer people.

**Part B** uses one month of real NYC Uber trip data to show why a normal rider-level A/B test gives a biased answer when both groups share the same driver supply, and how a switchback design fixes it.

Numbers below come from the notebooks in `notebooks/`. The analysis plan for Part A (metric, comparisons, correction method, outlier rule) was written before looking at any result, in `A1` of that notebook.

## TL;DR

- CUPAC (ML-based variance reduction) cut variance 2.3% on visit rate, but almost nothing on spend, because spend is ~99% zeros and no pre-period feature predicts it well.
- A/A test on 1,000 splits of the control group: false positive rate came out to 4.9-5.1%, right where it should be.
- On real Uber data, a rider-level A/B test overstated a 10% demand lift as +10.0%, when the true effect (after accounting for limited driver supply) was +5.1%. A switchback design landed within 0.1 points of the truth.
- Ignoring clustering in a switchback wait-time test gives a 52-68% false positive rate instead of 5%, depending on block length.

---

## Part A: Promotion A/B test, power, and CUPED

A marketing team sent two different promo emails to randomly selected customers. Did either one work, which should we ship, and could we have run this with fewer people?

### Sanity checks

- **SRM:** arm sizes are Mens 21,307 / Womens 21,387 / Control 21,306. Chi-square p = 0.90. No mismatch.
- **Covariate balance:** standardized mean difference on recency, history, mens, womens, newbie is under 0.01 for both arms vs. control (threshold is 0.05). Randomization looks clean.
- **Expected values check:** control visit rate is 10.62% (expected ~10.6%), control spend is $0.65 (expected ~$0.65). Mens email beats Womens on visit rate, as expected.

### Treatment effects

Holm correction applied within each metric, across the two comparisons.

| Metric | Arm | Diff | 95% CI | Relative lift | p (raw) | p (Holm) |
|---|---|---|---|---|---|---|
| Spend (uncapped) | Mens | 0.77 | [0.49, 1.05] | +117.9% | 1.2e-07 | 2.3e-07 |
| Spend (uncapped) | Womens | 0.42 | [0.17, 0.68] | +65.0% | 1.1e-03 | 1.1e-03 |
| Spend (capped at $243.66, p99.9) | Mens | 0.66 | [0.43, 0.88] | +113.5% | 8.5e-09 | 1.7e-08 |
| Spend (capped) | Womens | 0.38 | [0.18, 0.59] | +65.9% | 3.0e-04 | 3.0e-04 |
| Visit | Mens | 0.077 | [0.070, 0.083] | +72.1% | <1e-16 | <1e-16 |
| Visit | Womens | 0.045 | [0.039, 0.052] | +42.6% | <1e-16 | <1e-16 |
| Conversion | Mens | 0.0068 | [0.0050, 0.0086] | +118.8% | 1.5e-13 | 2.9e-13 |
| Conversion | Womens | 0.0031 | [0.0015, 0.0047] | +54.3% | 1.6e-04 | 1.6e-04 |

Every comparison holds up after Holm correction, capped or not. The outlier cap pulls spend down a bit (Mens: 117.9% to 113.5%) but doesn't change the call. Figure 1: `reports/figures/fig1_forest_plot.png`.

### Power

Before the test: detecting a 10% relative lift in spend at 80% power needs **494,680 customers per arm**. We had 21,306 in control. After the test, the achieved MDE was **48.2% relative** ($0.31 absolute). The real effects (118% and 65%) were large enough to clear that bar, but the test was underpowered for anything smaller. Figure 2: `reports/figures/fig2_power_curve.png`.

### CUPED and CUPAC

| Metric | Method | Effect (Mens) | Variance reduction |
|---|---|---|---|
| Spend | raw | 0.770 | 0% |
| Spend | CUPED (history) | 0.767 | 0.05% |
| Spend | CUPAC (all pre-period features) | 0.768 | 0.02% |
| Visit | raw | 0.077 | 0% |
| Visit | CUPED (history) | 0.076 | 0.42% |
| Visit | CUPAC (all pre-period features) | 0.076 | 2.27% |

Neither method does much here. Correlation between spend and history is only 0.02, because spend is zero for most customers (only buyers spend anything), so a pre-period covariate can't explain much of its variance. CUPAC does a bit better on visit (2.27% vs. 0.42% for plain CUPED) since it uses all pre-period features, not just one. This is the real result: sometimes CUPED just doesn't help, and that's worth knowing before you plan a test around it.

### A/A validation

1,000 random splits of the control group, raw and CUPED test on each:

- False positive rate: raw 4.90%, CUPED 5.10%. Both land near the nominal 5%, so CUPED isn't inflating false positives.
- Average SE: raw 0.1588, CUPED 0.1588. Basically identical, matching the near-zero variance reduction above.

Figure 3: `reports/figures/fig3_aa_pvalues.png`.

### Decision

- **Ship Mens email.** It beats Womens on every metric, and both beat control with Holm-corrected p < 0.002.
- **Expected impact:** $76,983 per 100,000 customers [$48,515, $105,451], uncapped, over the 2-week window. $65,920 [$43,480, $88,350] if you use the capped number instead.
- **Caveats:** only a 2-week window, so we're not separating a first-send novelty effect from steady-state lift, and we have no read on long-run retention.
- **Next test:** a longer observation window, and either a stronger covariate for spend (a rolling average of recent spend, say) or a sample size sized to the real 48% MDE we actually got.

---

## Part B: Switchback designs on real NYC Uber data

Say Uber runs a rider promo that makes treated riders request 10% more trips. In a normal rider-level A/B test, treated and control riders pull from the same pool of drivers, so the test overstates the real effect. This uses one month of real NYC Uber trips to measure that bias and show a switchback design fixes it, then uses real wait-time data to pick a block length.

### B1: zone x hour cells

Filtered to Uber trips with a valid wait time (0-60 min) in the 20 busiest Manhattan zones: **2,783,307 trips**, mean wait **4.58 min**, median wait **3.70 min**, across 14,880 zone-hour cells.

Figure 4 (`reports/figures/fig4_demand_heatmap.png`) shows the usual pattern: weekday rush hours and weekend nights are the demand peaks.

### B2: interference simulation (L = 10% demand lift, 300 draws per design)

Demand is the real trip count. Supply cap is the 80th percentile of demand per zone and hour-of-week. Completed trips = min(demand, supply). Treated riders request 10% more trips than they would have.

| Design | Mean estimate | SD | Bias vs. ground truth |
|---|---|---|---|
| Ground truth | 5.09% | - | - |
| Rider-level A/B | 9.99% | 0.13% | +4.90 pts |
| Switchback (2h blocks) | 5.20% | 0.94% | +0.11 pts |

The rider-level A/B lands right on the injected 10% lift no matter how tight supply is, because it's measuring requests, not what the market can actually serve. The switchback lands within 0.11 points of the real 5.09% effect, but its spread is about 7x wider. That's the trade: the switchback trades some variance for getting rid of the bias. Figure 5: `reports/figures/fig5_bias_variance.png`.

**Harm to control riders:** in an all-control world, 98.36% of requests get matched. In the rider-level A/B, control riders only see a **95.68%** match rate, a 2.68-point drop caused just by sharing supply with the treated group. Control riders' outcomes depend on what treatment did, which breaks the no-interference assumption (SUTVA) that the test design assumes holds.

**Sensitivity check** (150 draws per row):

| Supply percentile | Ground truth | Rider-level A/B mean | Bias |
|---|---|---|---|
| 70th (tightest) | 3.99% | 10.00% | +6.01 pts |
| 80th | 5.09% | 9.99% | +4.90 pts |
| 90th (loosest) | 5.86% | 10.01% | +4.14 pts |

The A/B estimate barely moves. The ground truth does, and so does the bias: tighter supply means more interference, so the gap between what the A/B test says and reality gets wider.

### B3: picking a switchback block length

A/A test (no injected effect) on the real wait series. 500 random assignments per block length, estimated with `wait ~ T + C(PULocationID) + C(how)`.

| Block length | Empirical SE | Naive SE | Empirical / naive | MDE | False positive rate (naive SE) |
|---|---|---|---|---|---|
| 1h | 0.0115 min | 0.0038 min | 3.0x | 1.94s | 51.6% |
| 2h | 0.0149 min | 0.0038 min | 3.9x | 2.51s | 58.6% |
| 4h | 0.0174 min | 0.0038 min | 4.5x | 2.92s | 67.6% |

Wait times inside the same zone-hour block are correlated, so treating every trip as its own independent observation (the naive SE) understates the real SE by 3 to 4.5x. On real data with no effect at all, that naive SE would call a false win 52-68% of the time instead of 5%. Block-level inference isn't optional here. Longer blocks have less carryover but fewer independent units, so the honest MDE grows from 1.94s at 1 hour to 2.92s at 4 hours. Figure 6: `reports/figures/fig6_block_length.png`.

**The trade-off:** longer blocks mean less carryover (driver positions and queues from the last block settle down), but fewer independent units, so the MDE gets worse. Shorter blocks give more power but more carryover contamination. Two common fixes: drop the first 15-30 minutes after each switch as a burn-in, and randomize at the city-time level if zones can affect each other. Two things this analysis doesn't handle: spillovers between neighboring zones, and carryover within a single hour.

---

## Why this matters for two-sided marketplaces

The same mechanism that biases this rider-level test shows up anywhere two sides share a constrained resource: pricing changes that affect how many drivers are online, incentive programs that shift supply across zones, or dispatch algorithm tests where one arm's routing choices change what's available to the other arm. Any time your treatment can change the resource your control group is also competing for, a simple A/B split measures something other than the real effect. Switchbacks, or some other cluster-level randomization, are usually the fix.


## How to reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
jupyter nbconvert --to notebook --execute --inplace notebooks/01_promo_ab_cuped.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_switchback_nyc_uber.ipynb
```

Data (gitignored, download per `plan.md` before running):
- `data/hillstrom.csv`: Hillstrom MineThatData email test (Part A).
- `data/fhvhv_2026-03.parquet`: one month of NYC TLC HVFHV trip records, ~500 MB (Part B).
- `data/taxi_zone_lookup.csv`: TLC zone-to-borough lookup (Part B).

Part B's notebook runs about 1,800 simulated regressions (300 interference draws, plus 500 A/A draws for each of 3 block lengths) and takes roughly 10 minutes.
