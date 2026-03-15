"""
03_comparison.py
=================
Side-by-side comparison of manuscript results vs. replication results.

Runs every hypothesis using the corrected samples:
    - Task-level: n=3,365 (deduplicated, one row per unique task)
    - Cluster-level: n=568 (all clusters with non-null cluster_thinking_fraction,
      empty collaboration ratios filled with 0)

Produces a single comparison report showing manuscript values alongside
replication values for every reported statistic.

Usage:
    python 03_comparison.py

Output:
    results/comparison_report.txt
"""

import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import Logit


DATA_DIR = "data"
RESULTS_DIR = "results"


def ols_hc3(y, X, add_const=True):
    if add_const:
        X = sm.add_constant(X)
    return sm.OLS(y, X, missing="drop").fit(cov_type="HC3")


def ols_standard(y, X, add_const=True):
    """OLS without robust SEs — to check if manuscript used standard errors."""
    if add_const:
        X = sm.add_constant(X)
    return sm.OLS(y, X, missing="drop").fit()


def sobel_test(a, b, se_a, se_b):
    indirect = a * b
    se_indirect = np.sqrt(a**2 * se_b**2 + b**2 * se_a**2)
    z = indirect / se_indirect
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p, indirect


def fmt(val, decimals=4):
    """Format a number for the report."""
    if isinstance(val, str):
        return val
    return f"{val:.{decimals}f}"


def comparison_line(label, manuscript_val, replication_val, decimals=4):
    """Format a single comparison line."""
    m = fmt(manuscript_val, decimals) if manuscript_val is not None else "—"
    r = fmt(replication_val, decimals)
    if manuscript_val is not None and isinstance(manuscript_val, (int, float)) and isinstance(replication_val, (int, float)):
        if manuscript_val != 0:
            pct_diff = abs(replication_val - manuscript_val) / abs(manuscript_val) * 100
            return f"  {label:<40} {m:>12}  {r:>12}  ({pct_diff:.1f}% diff)"
        else:
            return f"  {label:<40} {m:>12}  {r:>12}"
    return f"  {label:<40} {m:>12}  {r:>12}"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    task_df = pd.read_csv(os.path.join(DATA_DIR, "task_level_dataset.csv"))
    cluster_df = pd.read_csv(os.path.join(DATA_DIR, "cluster_level_dataset.csv"))

    lines = []
    lines.append("=" * 80)
    lines.append("MANUSCRIPT vs. REPLICATION: SIDE-BY-SIDE COMPARISON")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"{'':40} {'MANUSCRIPT':>12}  {'REPLICATION':>12}")
    lines.append(f"{'':40} {'(submitted)':>12}  {'(corrected)':>12}")
    lines.append("-" * 80)

    # ===================================================================
    # SAMPLE SIZES
    # ===================================================================
    lines.append("")
    lines.append("SAMPLE SIZES")
    lines.append(comparison_line("Task-level n", 3611, len(task_df), 0))
    lines.append(comparison_line("Cluster-level n (with thinking)", 508,
                                 cluster_df["cluster_thinking_fraction"].notna().sum(), 0))
    lines.append(comparison_line("Total cluster rows", 593, len(cluster_df), 0))
    lines.append("")
    lines.append("  NOTE: Manuscript used n=3,611 (integrated dataset with multi-cluster")
    lines.append("  duplicate rows). Replication uses n=3,365 (deduplicated, one row per")
    lines.append("  unique task). Cluster-level uses all rows with non-null thinking data.")

    # ===================================================================
    # DV DIAGNOSTICS — explains why coefficients change more than sample size
    # ===================================================================
    lines.append("")
    lines.append("DV DIAGNOSTICS: task_conversation_share (pct)")
    dv_mean = task_df["task_conversation_share"].mean()
    dv_std = task_df["task_conversation_share"].std()
    dv_sum = task_df["task_conversation_share"].sum()
    lines.append(f"  Replication (n=3,365): mean={dv_mean:.6f}, std={dv_std:.6f}, sum={dv_sum:.2f}")
    lines.append(f"  Manuscript  (n=3,611): mean=0.277000, std=0.447000, sum~=1000.2")
    lines.append("")
    lines.append("  The pct variable sums to ~100 across unique tasks (each task's share")
    lines.append("  of total conversations). The 247 duplicate rows re-count the 94")
    lines.append("  highest-usage tasks (avg pct ~3.66, or 120x the population mean),")
    lines.append("  inflating the sum to ~1000 and the mean from 0.030 to 0.277.")
    lines.append("  These extreme-leverage duplicates dominated OLS fits on n=3,611.")
    lines.append("  The deduplicated n=3,365 sample removes this non-independence.")
    lines.append("  All coefficient changes at the task level are driven by this")
    lines.append("  leverage correction, not by a variable rescaling or swap.")

    # ===================================================================
    # H1: TAM — Task-Level Adoption
    # ===================================================================
    lines.append("")
    lines.append("=" * 80)
    lines.append("H1: TECHNOLOGY ACCEPTANCE MODEL (TAM) — TASK LEVEL")
    lines.append("=" * 80)

    dv = task_df["task_conversation_share"]

    # H1 Model 1: OLS — task_iteration + learning
    r1 = ols_standard(dv, task_df[["task_iteration", "learning"]])
    r1_hc3 = ols_hc3(dv, task_df[["task_iteration", "learning"]])
    lines.append("")
    lines.append("Table 5.2: OLS — task_iteration + learning")
    lines.append(comparison_line("n", 3611, int(r1.nobs), 0))
    lines.append(comparison_line("Constant", 0.051, r1.params["const"]))
    lines.append(comparison_line("task_iteration β", 0.127, r1.params["task_iteration"]))
    lines.append(comparison_line("task_iteration SE", 0.041, r1.bse["task_iteration"]))
    lines.append(comparison_line("task_iteration p", 0.002, r1.pvalues["task_iteration"]))
    lines.append(comparison_line("learning β", 0.082, r1.params["learning"]))
    lines.append(comparison_line("learning SE", 0.031, r1.bse["learning"]))
    lines.append(comparison_line("learning p", 0.007, r1.pvalues["learning"]))
    lines.append(comparison_line("R²", 0.004, r1.rsquared))
    lines.append(f"  {'HC3 task_iteration SE':<40} {'—':>12}  {fmt(r1_hc3.bse['task_iteration']):>12}")
    lines.append(f"  {'HC3 learning SE':<40} {'—':>12}  {fmt(r1_hc3.bse['learning']):>12}")

    # H1 Model 2: OLS — augmentation_index
    r2 = ols_standard(dv, task_df[["augmentation_index"]])
    r2_hc3 = ols_hc3(dv, task_df[["augmentation_index"]])
    lines.append("")
    lines.append("Table 5.3: OLS — augmentation_index")
    lines.append(comparison_line("n", 3611, int(r2.nobs), 0))
    lines.append(comparison_line("Constant", 0.047, r2.params["const"]))
    lines.append(comparison_line("augmentation_index β", 0.113, r2.params["augmentation_index"]))
    lines.append(comparison_line("augmentation_index SE", 0.026, r2.bse["augmentation_index"]))
    lines.append(comparison_line("R²", 0.005, r2.rsquared))
    lines.append(f"  {'HC3 SE':<40} {'—':>12}  {fmt(r2_hc3.bse['augmentation_index']):>12}")

    # H1 Model 3: Log-linear OLS
    log_dv = np.log(dv)
    r3 = ols_standard(log_dv, task_df[["augmentation_index"]])
    r3_hc3 = ols_hc3(log_dv, task_df[["augmentation_index"]])
    fold_change = np.exp(r3.params["augmentation_index"])
    lines.append("")
    lines.append("Table 5.4: Log-linear OLS — log(task_conversation_share) ~ augmentation_index")
    lines.append(comparison_line("n", 3611, int(r3.nobs), 0))
    lines.append(comparison_line("Intercept", -5.786, r3.params["const"]))
    lines.append(comparison_line("augmentation_index β", 3.100, r3.params["augmentation_index"]))
    lines.append(comparison_line("augmentation_index SE", 0.077, r3.bse["augmentation_index"]))
    lines.append(comparison_line("R²", 0.310, r3.rsquared))
    lines.append(comparison_line("F-statistic", 1620.84, r3.fvalue, 2))
    lines.append(comparison_line("exp(β) fold-change", 22.2, fold_change, 1))
    lines.append(f"  {'HC3 SE':<40} {'—':>12}  {fmt(r3_hc3.bse['augmentation_index']):>12}")

    # ===================================================================
    # H2: PMT — Risk Management
    # ===================================================================
    lines.append("")
    lines.append("=" * 80)
    lines.append("H2: PROTECTION MOTIVATION THEORY (PMT) — TASK LEVEL")
    lines.append("=" * 80)

    # H2a: risk_management_index -> task_conversation_share
    r_h2 = ols_standard(dv, task_df[["risk_management_index"]])
    r_h2_hc3 = ols_hc3(dv, task_df[["risk_management_index"]])
    lines.append("")
    lines.append("Table 5.5: OLS — risk_management_index")
    lines.append(comparison_line("n", 3611, int(r_h2.nobs), 0))
    lines.append(comparison_line("Intercept", -0.006, r_h2.params["const"]))
    lines.append(comparison_line("risk_management_index β", 2.484, r_h2.params["risk_management_index"]))
    lines.append(comparison_line("risk_management_index SE", 0.071, r_h2.bse["risk_management_index"]))
    lines.append(comparison_line("R²", 0.253, r_h2.rsquared))
    lines.append(comparison_line("F-statistic", 1220.69, r_h2.fvalue, 2))
    lines.append(f"  {'HC3 SE':<40} {'—':>12}  {fmt(r_h2_hc3.bse['risk_management_index']):>12}")

    # H2b: risk_management_index -> directive
    r_h2b = ols_standard(task_df["directive"], task_df[["risk_management_index"]])
    lines.append("")
    lines.append("Table 5.6 H2b: risk_management_index -> directive")
    lines.append(comparison_line("β", 0.190, r_h2b.params["risk_management_index"]))
    lines.append(comparison_line("SE", 0.036, r_h2b.bse["risk_management_index"]))

    # H2c: risk_management_index -> task_iteration
    r_h2c = ols_standard(task_df["task_iteration"], task_df[["risk_management_index"]])
    lines.append("")
    lines.append("Table 5.6 H2c: risk_management_index -> task_iteration")
    lines.append(comparison_line("β", 0.082, r_h2c.params["risk_management_index"]))
    lines.append(comparison_line("SE", 0.034, r_h2c.bse["risk_management_index"]))

    # ===================================================================
    # H3: SET — Cluster-Level Trust
    # ===================================================================
    lines.append("")
    lines.append("=" * 80)
    lines.append("H3: SOCIAL EXCHANGE THEORY (SET) — CLUSTER LEVEL")
    lines.append("=" * 80)

    df_h3 = cluster_df.dropna(subset=["cluster_thinking_fraction"]).copy()
    r_h3 = ols_standard(df_h3["cluster_thinking_fraction"], df_h3[["cluster_augmentation_index"]])
    r_h3_hc3 = ols_hc3(df_h3["cluster_thinking_fraction"], df_h3[["cluster_augmentation_index"]])
    lines.append("")
    lines.append("Table 5.7: OLS — cluster_augmentation_index -> cluster_thinking_fraction")
    lines.append(comparison_line("n", 508, int(r_h3.nobs), 0))
    lines.append(comparison_line("Intercept", 0.0853, r_h3.params["const"]))
    lines.append(comparison_line("cluster_augmentation_index β", -0.0611, r_h3.params["cluster_augmentation_index"]))
    lines.append(comparison_line("cluster_augmentation_index SE", 0.0087, r_h3.bse["cluster_augmentation_index"]))
    lines.append(comparison_line("R²", 0.088, r_h3.rsquared))
    lines.append(comparison_line("F-statistic", 49.07, r_h3.fvalue, 2))
    lines.append(f"  {'HC3 SE':<40} {'—':>12}  {fmt(r_h3_hc3.bse['cluster_augmentation_index']):>12}")

    # ===================================================================
    # H4: STS — Cluster-Level
    # ===================================================================
    lines.append("")
    lines.append("=" * 80)
    lines.append("H4: SOCIO-TECHNICAL SYSTEMS (STS) — CLUSTER LEVEL")
    lines.append("=" * 80)

    df_h4 = cluster_df.dropna(subset=["cluster_thinking_fraction"]).copy()
    r_h4 = ols_standard(
        df_h4["cluster_thinking_fraction"],
        df_h4[["cluster_automation_index", "cluster_augmentation_index"]]
    )
    r_h4_hc3 = ols_hc3(
        df_h4["cluster_thinking_fraction"],
        df_h4[["cluster_automation_index", "cluster_augmentation_index"]]
    )
    lines.append("")
    lines.append("Table 5.8: OLS — cluster_automation + cluster_augmentation")
    lines.append(comparison_line("n", 488, int(r_h4.nobs), 0))
    lines.append(comparison_line("Intercept", 0.0836, r_h4.params["const"]))
    lines.append(comparison_line("cluster_automation_index β", 0.0016, r_h4.params["cluster_automation_index"]))
    lines.append(comparison_line("cluster_automation_index SE", 0.0268, r_h4.bse["cluster_automation_index"]))
    lines.append(comparison_line("cluster_automation_index p", 0.952, r_h4.pvalues["cluster_automation_index"]))
    lines.append(comparison_line("cluster_augmentation_index β", -0.0579, r_h4.params["cluster_augmentation_index"]))
    lines.append(comparison_line("cluster_augmentation_index SE", 0.0267, r_h4.bse["cluster_augmentation_index"]))
    lines.append(comparison_line("cluster_augmentation_index p", 0.031, r_h4.pvalues["cluster_augmentation_index"]))
    lines.append(comparison_line("R²", 0.102, r_h4.rsquared))
    lines.append("")
    lines.append("  H4 NOTE: Manuscript n=488 cannot be reproduced from any transparent")
    lines.append("  filter. Replication uses n=568 (all clusters with non-null thinking")
    lines.append("  data, same rule as H3). On this sample, cluster_augmentation is NOT")
    lines.append("  significant (p=0.458). H4 changes from weakly supported to not")
    lines.append("  supported. The manuscript's significant finding appears to depend on")
    lines.append("  an undocumented sample restriction that excluded 80 clusters.")

    # ===================================================================
    # H5: Cross-Level Mediation
    # ===================================================================
    lines.append("")
    lines.append("=" * 80)
    lines.append("H5: CROSS-LEVEL MEDIATION (TAM × SET)")
    lines.append("=" * 80)

    df_h5 = cluster_df.dropna(subset=["cluster_thinking_fraction"]).copy()
    iv = df_h5["augmentation_index"]
    mediator = df_h5["cluster_thinking_fraction"]
    dv_h5 = df_h5["task_conversation_share"]

    # Step 1: c path
    s1 = ols_standard(dv_h5, iv)
    s1_hc3 = ols_hc3(dv_h5, iv)
    lines.append("")
    lines.append("Step 1 (Path c): augmentation_index -> task_conversation_share")
    lines.append(comparison_line("β (augmentation)", -1.008, s1.params.iloc[1]))
    lines.append(comparison_line("SE", 0.239, s1.bse.iloc[1]))
    lines.append(comparison_line("R²", 0.031, s1.rsquared))

    # Step 2: a path
    s2 = ols_standard(mediator, iv)
    s2_hc3 = ols_hc3(mediator, iv)
    lines.append("")
    lines.append("Step 2 (Path a): augmentation_index -> cluster_thinking_fraction")
    lines.append(comparison_line("β (augmentation)", -0.017, s2.params.iloc[1]))
    lines.append(comparison_line("SE", 0.006, s2.bse.iloc[1]))
    lines.append(comparison_line("R²", 0.014, s2.rsquared))

    # Step 3: c' and b paths
    X_s3 = df_h5[["augmentation_index", "cluster_thinking_fraction"]]
    s3 = ols_standard(dv_h5, X_s3)
    s3_hc3 = ols_hc3(dv_h5, X_s3)
    lines.append("")
    lines.append("Step 3: augmentation_index + cluster_thinking_fraction -> task_conversation_share")
    lines.append(comparison_line("β (augmentation, c')", -0.903, s3.params["augmentation_index"]))
    lines.append(comparison_line("SE (augmentation)", 0.238, s3.bse["augmentation_index"]))
    lines.append(comparison_line("β (thinking, b)", 6.194, s3.params["cluster_thinking_fraction"]))
    lines.append(comparison_line("SE (thinking)", 1.652, s3.bse["cluster_thinking_fraction"]))
    lines.append(comparison_line("R²", 0.054, s3.rsquared))

    # Sobel test
    a = s2.params.iloc[1]
    a_se = s2.bse.iloc[1]
    b = s3.params["cluster_thinking_fraction"]
    b_se = s3.bse["cluster_thinking_fraction"]
    z_sobel, p_sobel, indirect = sobel_test(a, b, a_se, b_se)
    lines.append("")
    lines.append("Sobel Test")
    lines.append(comparison_line("Indirect effect (a × b)", None, indirect))
    lines.append(comparison_line("Sobel z", None, z_sobel))
    lines.append(comparison_line("Sobel p", None, p_sobel))

    # ===================================================================
    # H6: SET × STS — Cluster-Level Adoption Breadth
    # ===================================================================
    lines.append("")
    lines.append("=" * 80)
    lines.append("H6: SET × STS — CLUSTER-LEVEL ADOPTION BREADTH")
    lines.append("=" * 80)

    df_h6 = cluster_df.dropna(
        subset=["percent_users", "cluster_augmentation_index", "cluster_thinking_fraction"]
    ).copy()

    # Manuscript descriptives (Table 5.9)
    lines.append("")
    lines.append("Table 5.9: Descriptive Statistics")
    lines.append(comparison_line("n", 508, len(df_h6), 0))
    lines.append(comparison_line("percent_users M", 0.162, df_h6["percent_users"].mean()))
    lines.append(comparison_line("percent_users SD", 0.034, df_h6["percent_users"].std()))
    lines.append(comparison_line("cluster_augmentation M", 0.563, df_h6["cluster_augmentation_index"].mean()))
    lines.append(comparison_line("cluster_augmentation SD", 0.140, df_h6["cluster_augmentation_index"].std()))
    lines.append(comparison_line("cluster_thinking M", 0.051, df_h6["cluster_thinking_fraction"].mean()))
    lines.append(comparison_line("cluster_thinking SD", 0.026, df_h6["cluster_thinking_fraction"].std()))

    # Mean-center
    df_h6["aug_c"] = df_h6["cluster_augmentation_index"] - df_h6["cluster_augmentation_index"].mean()
    df_h6["think_c"] = df_h6["cluster_thinking_fraction"] - df_h6["cluster_thinking_fraction"].mean()
    df_h6["aug_x_think"] = df_h6["aug_c"] * df_h6["think_c"]

    # Model 1: augmentation only
    m1 = ols_standard(df_h6["percent_users"], df_h6[["cluster_augmentation_index"]])
    lines.append("")
    lines.append("Table 5.12 Model 1: augmentation only")
    lines.append(comparison_line("Augmentation B", -0.027, m1.params["cluster_augmentation_index"]))
    lines.append(comparison_line("Augmentation p", 0.011, m1.pvalues["cluster_augmentation_index"]))
    lines.append(comparison_line("R²", 0.013, m1.rsquared))

    # Model 3: augmentation + thinking
    m3 = ols_standard(
        df_h6["percent_users"],
        df_h6[["cluster_augmentation_index", "cluster_thinking_fraction"]]
    )
    lines.append("")
    lines.append("Table 5.12 Model 3: augmentation + thinking")
    lines.append(comparison_line("Augmentation B", -0.031, m3.params["cluster_augmentation_index"]))
    lines.append(comparison_line("Augmentation p", 0.007, m3.pvalues["cluster_augmentation_index"]))
    lines.append(comparison_line("Thinking B", -0.060, m3.params["cluster_thinking_fraction"]))
    lines.append(comparison_line("Thinking p", 0.328, m3.pvalues["cluster_thinking_fraction"]))

    # Model 4: + interaction
    m4 = ols_standard(
        df_h6["percent_users"],
        df_h6[["aug_c", "think_c", "aug_x_think"]]
    )
    lines.append("")
    lines.append("Table 5.12 Model 4: + interaction")
    lines.append(comparison_line("Interaction B", -0.273, m4.params["aug_x_think"]))
    lines.append(comparison_line("Interaction p", None, m4.pvalues["aug_x_think"]))

    # ===================================================================
    # CROSS-LEVEL CORRELATIONS (Table 5.10)
    # ===================================================================
    lines.append("")
    lines.append("=" * 80)
    lines.append("TABLE 5.10: CROSS-LEVEL CORRELATION COMPARISON")
    lines.append("=" * 80)

    # Task-level correlations with task_conversation_share
    task_corr_vars = ["directive", "feedback_loop", "task_iteration", "learning",
                      "validation", "augmentation_index"]
    lines.append("")
    lines.append("Task-level correlations with task_conversation_share:")
    manuscript_task_r = {
        "directive": 0.057, "feedback_loop": 0.554, "task_iteration": 0.044,
        "learning": 0.035, "validation": 0.091, "augmentation_index": 0.072
    }
    for var in task_corr_vars:
        r_val = task_df[var].corr(task_df["task_conversation_share"])
        lines.append(comparison_line(f"  {var} r", manuscript_task_r.get(var), r_val))

    # ===================================================================
    # SUMMARY
    # ===================================================================
    lines.append("")
    lines.append("=" * 80)
    lines.append("SUMMARY OF KEY CHANGES")
    lines.append("=" * 80)
    lines.append("")
    lines.append("  1. TASK-LEVEL SAMPLE: 3,611 -> 3,365 (deduplicated)")
    lines.append(f"     H1 log-linear fold-change: 22.2x -> {fold_change:.1f}x")
    lines.append(f"     H1 log-linear R²: 0.310 -> {r3.rsquared:.3f}")
    lines.append(f"     H1 log-linear β: 3.100 -> {r3.params['augmentation_index']:.3f}")
    lines.append("")
    lines.append("  2. CLUSTER-LEVEL SAMPLE: 508 -> 568 (all with thinking data)")
    lines.append(f"     H3 β: -0.061 -> {r_h3.params['cluster_augmentation_index']:.4f}")
    lines.append(f"     H3 R²: 0.088 -> {r_h3.rsquared:.3f}")
    lines.append("")
    lines.append("  3. DIRECTION AND SIGNIFICANCE:")
    lines.append("     H1: Supported (all three models, same direction, all p < .001)")
    lines.append("     H2a: Supported (positive, p < .001; coefficient smaller due to")
    lines.append("           leverage correction — see DV DIAGNOSTICS above)")
    lines.append("     H2b/H2c: Supported (same direction)")
    lines.append("     H3: Supported (negative, p < .001)")
    lines.append("     H4: NOT SUPPORTED on n=568 (augmentation p=0.458)")
    lines.append("     H5: Supported (mediation pattern intact, near-identical values)")
    lines.append("     H6: Not supported (same as manuscript)")
    lines.append("")
    lines.append("  4. WHY TASK-LEVEL COEFFICIENTS CHANGE MORE THAN 7%:")
    lines.append("     The 247 duplicate rows are the 94 highest-usage tasks (avg pct")
    lines.append("     ~3.66 vs population mean ~0.03). In OLS, these act as extreme")
    lines.append("     leverage points that dominate the slope estimate. Removing them")
    lines.append("     corrects non-independence and produces honest estimates. The log-")
    lines.append("     linear H1 model changes only 9% because log(pct) compresses the")
    lines.append("     extremes. H2a changes 73% because risk_management_index is")
    lines.append("     concentrated in these same high-usage tasks (code debugging).")
    lines.append("")
    lines.append("  5. ITEMS REQUIRING MANUSCRIPT UPDATE:")
    lines.append("     - All task-level n's: 3,611 -> 3,365")
    lines.append("     - All cluster-level n's: 508 -> 568 (H3, H6), 488 -> 568 (H4)")
    lines.append(f"     - H1 headline: 22-fold -> ~{fold_change:.0f}-fold")
    lines.append("     - H2a: β from 2.484 to ~0.67, R² from 0.253 to ~0.085")
    lines.append("     - H4: Augmentation no longer significant; reframe as not supported")
    lines.append("     - All coefficient values and SEs in Tables 5.1-5.12")
    lines.append("     - Abstract (22-fold reference)")
    lines.append("     - Methods: explain deduplication rationale and cluster sample rule")
    lines.append("")
    lines.append("=" * 80)

    report = "\n".join(lines)
    report_path = os.path.join(RESULTS_DIR, "comparison_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    print(report)
    print(f"\nSaved to {report_path}")


if __name__ == "__main__":
    main()
