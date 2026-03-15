# Replication Notes: The Collaboration Code

## Purpose

This document explains all discrepancies between the submitted manuscript
(Lynch, "The Collaboration Code") and the replication pipeline
(`01_build_dataset.py`, `02_analysis.py`, `03_comparison.py`). It is
intended for reviewers and independent researchers who want to understand
why the corrected results differ from the submitted version.

## Root Cause 1: Task-Level Sample (n = 3,611 → 3,365)

### What happened

The Anthropic Economic Index maps 3,365 unique O\*NET tasks to Claude
conversation data. Of these, 346 tasks also appear in a cluster-level
taxonomy. Because 94 of the 346 tasks map to multiple clusters, a left
join between the task-level base (3,365 rows) and the cluster data
(593 task-cluster rows) produces an integrated dataset of 3,612 rows.

The submitted manuscript ran all task-level regressions (H1, H2) on
the 3,612-row integrated dataset. This means the 94 multi-cluster tasks
are counted 2–7 times each, violating the independence assumption of OLS.

### Why the coefficients change more than 7%

The 247 duplicate rows are not a random 7% subsample. They are the
94 highest-usage tasks in the dataset:

| Statistic              | Unique tasks (n=3,365) | Duplicate rows (n=247) |
|------------------------|------------------------|------------------------|
| Mean pct               | 0.030                  | ~3.66                  |
| Ratio to population    | 1×                     | ~122×                  |
| Sum of pct             | ~100                   | ~900                   |

These extreme values act as leverage points in OLS. A single observation
at 120× the mean exerts disproportionate influence on the slope. Counting
it twice approximately doubles that influence.

The effect is largest for H2a (risk_management_index → adoption) because
the highest-usage tasks (e.g., code debugging) also have the highest
risk_management_index values. The regression on n=3,611 was essentially
fitting a line through ~94 extreme points counted multiple times.

The log-linear H1 model is less affected (β changes 9%) because the
log transform compresses extreme DV values from 6.65 to 1.89, dramatically
reducing their leverage.

### The correct approach

Task-level hypotheses require one row per task. The deduplicated sample
(n = 3,365) is the correct analytical unit. The `01_build_dataset.py`
script uses `drop_duplicates(subset=["task_name"])` to produce this.

### Impact on results

| Hypothesis | Manuscript β | Corrected β | Direction | Significance |
|------------|-------------|-------------|-----------|-------------|
| H1 (log-linear) | 3.100 (22.2×) | 2.814 (16.7×) | Same | Same (p < .001) |
| H1 (augmentation OLS) | 0.113 | 0.060 | Same | Same (p < .001) |
| H2a | 2.484 (R²=.253) | 0.665 (R²=.085) | Same | Same (p < .001) |
| H2b | 0.190 | 0.263 | Same | Same (p < .001) |
| H2c | 0.082 | 0.155 | Same | Same (p < .001) |

All task-level hypotheses maintain the same direction and significance.
The substantive conclusions hold: augmentation predicts adoption (H1),
and risk management positively (not negatively) predicts adoption (H2a),
contradicting PMT.

## Root Cause 2: Cluster-Level Sample (n = 488/508 → 568)

### What happened

The manuscript reports varying cluster-level sample sizes across
hypotheses: n = 508 for H3 and H6, n = 488 for H4. No filter criteria
are documented that produce these exact counts from the 593 total
cluster rows.

The replication uses a single transparent rule: include all cluster rows
with non-null `cluster_thinking_fraction` (the extended thinking DV
required for H3, H4, and H6). This produces n = 568.

### Investigation of n = 488 (H4)

Extensive testing of filter combinations found that restricting to
clusters with `cluster_thinking_fraction > 0.022` and
`cluster_automation_index > 0.15` produces approximately n = 486–488
and closely reproduces the manuscript's coefficients. However, these
thresholds were never documented, and applying them selectively amplifies
the augmentation effect (β strengthens 3.4×, from −0.018 to −0.062).

### Impact on results

| Hypothesis | Manuscript | Corrected (n=568) | Change |
|------------|-----------|-------------------|--------|
| H3 | β=−0.061, p<.001 | β=−0.049, p<.001 | Direction and significance same |
| H4 augmentation | β=−0.058, p=.031 | β=−0.018, p=.458 | **No longer significant** |
| H4 automation | β=0.002, p=.952 | β=0.033, p=.182 | Remains non-significant |
| H6 Model 1 augmentation | B=−.027, p=.011 | B=−.021, p=.035 | Weaker but still significant |
| H6 interaction | B=−.273, p=.479 | B=−.154, p=.660 | Remains non-significant |

**H4 requires reframing**: On the transparent n = 568 sample, neither
automation nor augmentation individually predicts extended thinking usage
when both are entered. This means H4 (STS) is not supported — the
combination of automation and augmentation patterns does not predict
extended thinking as hypothesized.

H3 remains supported. H6 remains not supported (same as manuscript).

## Root Cause 3: Standard vs. HC3 Standard Errors

### What happened

The manuscript tables report standard (homoskedastic) OLS standard
errors. The replication pipeline uses HC3 heteroskedasticity-consistent
standard errors throughout, which is the more conservative choice given
the heavily skewed DV.

### Impact

HC3 SEs are generally similar to standard SEs for these models. The
comparison script (`03_comparison.py`) runs both versions for every model
and reports the standard-SE results alongside HC3 for direct comparison.
No hypothesis changes significance based on SE type alone.

## H5 Mediation: Near-Perfect Replication

The cross-level mediation (H5) uses the cluster-level dataset (n = 568),
which includes both task-level and cluster-level variables. The Baron &
Kenny steps and Sobel test replicate within rounding error of the
manuscript (all values within 0.2%). This is expected because H5 was
likely run on the cluster sample, where duplicate rows do not create
the same leverage problem (each row is a unique task-cluster combination).

## Verification Procedure for Independent Researchers

1. Run `python 01_build_dataset.py` — downloads data from HuggingFace,
   builds all datasets. Compare SHA-256 checksums in `data/build_log.txt`.
2. Run `python 02_analysis.py` — produces all hypothesis results in
   `results/`. Uses HC3 robust SEs.
3. Run `python 03_comparison.py` — produces side-by-side comparison
   of manuscript vs. replication values in `results/comparison_report.txt`.
4. Verify that `data/task_level_dataset.csv` has exactly 3,365 rows,
   `data/cluster_level_dataset.csv` has exactly 593 rows, and
   `data/integrated_dataset.csv` has exactly 3,612 rows.

## Summary of Manuscript Items Requiring Revision

### Tables requiring updated values
- Table 5.1: Correlation matrix (recompute at n=3,365)
- Table 5.2: H1 individual predictors (n, coefficients, SEs)
- Table 5.3: H1 augmentation index (n, coefficients, SEs)
- Table 5.4: H1 log-linear (n, coefficients, R², fold-change)
- Table 5.5: H2a risk management (n, coefficients, R²)
- Table 5.6: H2 sub-models (n, coefficients)
- Table 5.7: H3 cluster thinking (n, coefficients, R²)
- Table 5.8: H4 multiple regression (n, coefficients — augmentation no longer significant)
- Table 5.9: H6 descriptive statistics (n, means, SDs)
- Table 5.10: Cross-level correlations (recompute both levels)
- Table 5.11: Cluster-level correlation matrix (n)
- Table 5.12/5.13: H6 hierarchical regression (n, coefficients)

### Narrative sections requiring revision
- Abstract: "22-fold" → "~17-fold"
- Section 4.1 (Sample sizes): 3,612 → 3,365 for task-level; explain
  deduplication; adopt single cluster rule (n=568)
- H1 results: update coefficient values and fold-change
- H2 results: update β and R² values; temper "substantial" effect language
- H4 results: reframe as not supported; augmentation no longer significant
- Discussion: adjust H4 interpretation
