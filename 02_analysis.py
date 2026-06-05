"""
02_analysis.py
===============
Hypothesis testing for Lynch et al. analysis of the Anthropic Economic Index.

Runs H1-H6 using the analytical samples produced by 01_build_dataset.py.
All models use HC3 robust standard errors throughout.

Statistical methods:
    - OLS regression (standard, log-linear, hierarchical)
    - Fractional logit (GLM with binomial family, logit link)
    - Baron & Kenny (1986) mediation with Sobel test
    - Pearson correlation matrices
    - Descriptive statistics

Note: Zero-inflated models were considered but are not applicable here.
    Anthropic's source data only includes tasks with non-zero usage (pct > 0),
    so the structural zeros that zero-inflated models address do not exist in
    this sample. See manuscript H1/H2 scope conditions.

Usage:
    python 02_analysis.py

Requires:
    - data/task_level_dataset.csv   (from 01_build_dataset.py)
    - data/cluster_level_dataset.csv (from 01_build_dataset.py)

Outputs:
    - results/descriptives.txt
    - results/correlations_task.csv
    - results/correlations_cluster.csv
    - results/H1_results.txt
    - results/H2_results.txt
    - results/H3_results.txt
    - results/H4_results.txt
    - results/H5_mediation_results.txt
    - results/H6_results.txt
    - results/robustness_checks.txt
    - results/analysis_log.txt
"""

import os
import logging
import warnings
from datetime import datetime, timezone
from io import StringIO

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import Logit

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = "data"
RESULTS_DIR = "results"

# Suppress convergence warnings during iterative model fitting
warnings.filterwarnings("ignore", category=sm.tools.sm_exceptions.ConvergenceWarning)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def ols_with_hc3(y, X, add_constant=True):
    """
    Run OLS regression with HC3 robust standard errors.

    Parameters
    ----------
    y : array-like
        Dependent variable.
    X : DataFrame or array-like
        Independent variable(s).
    add_constant : bool
        Whether to add a constant term.

    Returns
    -------
    statsmodels RegressionResultsWrapper with HC3 covariance.
    """
    if add_constant:
        X = sm.add_constant(X)
    model = sm.OLS(y, X, missing="drop")
    return model.fit(cov_type="HC3")


def format_ols_summary(result, title=""):
    """Format OLS results into a clean text block."""
    lines = []
    if title:
        lines.append("=" * 70)
        lines.append(title)
        lines.append("=" * 70)
    lines.append(result.summary().as_text())
    lines.append("")
    lines.append(f"R-squared:     {result.rsquared:.6f}")
    lines.append(f"Adj R-squared: {result.rsquared_adj:.6f}")
    lines.append(f"F-statistic:   {result.fvalue:.4f} (p = {result.f_pvalue:.2e})")
    lines.append(f"N observations: {int(result.nobs)}")
    lines.append(f"Covariance:    HC3 (robust)")
    lines.append("")
    return "\n".join(lines)


def sobel_test(a, b, se_a, se_b):
    """
    Sobel test for significance of the indirect effect in mediation.

    Parameters
    ----------
    a : float
        Coefficient of IV -> Mediator path.
    b : float
        Coefficient of Mediator -> DV path (controlling for IV).
    se_a : float
        Standard error of a.
    se_b : float
        Standard error of b.

    Returns
    -------
    tuple: (z_statistic, p_value, indirect_effect)
    """
    indirect = a * b
    se_indirect = np.sqrt(a**2 * se_b**2 + b**2 * se_a**2)
    z = indirect / se_indirect
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p, indirect


# ---------------------------------------------------------------------------
# Step 0: Load data
# ---------------------------------------------------------------------------

def load_data():
    """Load the analytical samples produced by 01_build_dataset.py."""
    task_path = os.path.join(DATA_DIR, "task_level_dataset.csv")
    cluster_path = os.path.join(DATA_DIR, "cluster_level_dataset.csv")

    if not os.path.exists(task_path) or not os.path.exists(cluster_path):
        raise FileNotFoundError(
            "Data files not found. Run 01_build_dataset.py first."
        )

    task_df = pd.read_csv(task_path)
    cluster_df = pd.read_csv(cluster_path)

    logger.info(f"Task-level data: {len(task_df)} rows, {len(task_df.columns)} columns")
    logger.info(f"Cluster-level data: {len(cluster_df)} rows, {len(cluster_df.columns)} columns")

    return task_df, cluster_df


# ---------------------------------------------------------------------------
# Step 1: Descriptive statistics
# ---------------------------------------------------------------------------

def compute_descriptives(task_df, cluster_df):
    """Compute descriptive statistics at both analytical levels."""
    lines = []
    lines.append("=" * 70)
    lines.append("DESCRIPTIVE STATISTICS")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("=" * 70)

    # Task-level variables
    task_vars = [
        "task_conversation_share",
        "directive", "feedback_loop", "task_iteration", "learning", "validation",
        "automation_index", "augmentation_index", "risk_management_index",
        "thinking_fraction",
    ]
    lines.append("\n--- TASK LEVEL (n = {}) ---\n".format(len(task_df)))

    for var in task_vars:
        if var not in task_df.columns:
            continue
        s = task_df[var].dropna()
        lines.append(f"  {var}:")
        lines.append(f"    n = {len(s)}, mean = {s.mean():.6f}, sd = {s.std():.6f}")
        lines.append(f"    min = {s.min():.6f}, max = {s.max():.6f}")
        lines.append(f"    skewness = {s.skew():.4f}, kurtosis = {s.kurtosis():.4f}")
        lines.append("")

    # Cluster-level variables
    cluster_vars = [
        "task_conversation_share",
        "cluster_directive", "cluster_feedback_loop",
        "cluster_task_iteration", "cluster_learning", "cluster_validation",
        "cluster_automation_index", "cluster_augmentation_index",
        "cluster_risk_management_index",
        "cluster_thinking_fraction",
        "percent_users", "percent_records",
    ]
    lines.append("\n--- CLUSTER LEVEL (n = {}) ---\n".format(len(cluster_df)))

    for var in cluster_vars:
        if var not in cluster_df.columns:
            continue
        s = cluster_df[var].dropna()
        lines.append(f"  {var}:")
        lines.append(f"    n = {len(s)}, mean = {s.mean():.6f}, sd = {s.std():.6f}")
        lines.append(f"    min = {s.min():.6f}, max = {s.max():.6f}")
        lines.append(f"    skewness = {s.skew():.4f}, kurtosis = {s.kurtosis():.4f}")
        lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 2: Correlation matrices
# ---------------------------------------------------------------------------

def compute_correlations(task_df, cluster_df):
    """Compute Pearson correlation matrices at both levels."""

    # Task-level correlations
    task_corr_vars = [
        "task_conversation_share",
        "directive", "feedback_loop", "task_iteration", "learning", "validation",
        "automation_index", "augmentation_index", "risk_management_index",
        "thinking_fraction",
    ]
    task_corr_vars = [v for v in task_corr_vars if v in task_df.columns]
    task_corr = task_df[task_corr_vars].corr(method="pearson")

    # Task-level p-values
    n_task = len(task_df)
    task_pvals = pd.DataFrame(
        np.zeros_like(task_corr), index=task_corr.index, columns=task_corr.columns
    )
    for i, col_i in enumerate(task_corr_vars):
        for j, col_j in enumerate(task_corr_vars):
            if i != j:
                mask = task_df[[col_i, col_j]].dropna()
                if len(mask) > 2:
                    r, p = stats.pearsonr(mask[col_i], mask[col_j])
                    task_pvals.loc[col_i, col_j] = p

    # Cluster-level correlations
    cluster_corr_vars = [
        "task_conversation_share",
        "cluster_automation_index", "cluster_augmentation_index",
        "cluster_risk_management_index",
        "cluster_thinking_fraction",
        "percent_users", "percent_records",
    ]
    cluster_corr_vars = [v for v in cluster_corr_vars if v in cluster_df.columns]
    cluster_corr = cluster_df[cluster_corr_vars].corr(method="pearson")

    return task_corr, task_pvals, cluster_corr


# ---------------------------------------------------------------------------
# Step 3: H1 — TAM (Task-level adoption)
# ---------------------------------------------------------------------------

def run_h1(task_df):
    """
    H1: Technology Acceptance Model — task-level adoption.

    DV: task_conversation_share
    IVs: task_iteration, learning, augmentation_index

    Models:
        1a. OLS: task_conversation_share ~ augmentation_index
        1b. OLS: task_conversation_share ~ task_iteration + learning
        1c. Log-linear: log(task_conversation_share) ~ augmentation_index
    """
    lines = []
    lines.append("=" * 70)
    lines.append("H1: TECHNOLOGY ACCEPTANCE MODEL (TAM) — TASK LEVEL")
    lines.append("=" * 70)

    dv = task_df["task_conversation_share"]

    # Model 1a: augmentation_index -> adoption
    result_1a = ols_with_hc3(dv, task_df[["augmentation_index"]])
    lines.append(format_ols_summary(result_1a, "Model 1a: OLS — augmentation_index"))

    # Model 1b: task_iteration + learning -> adoption
    result_1b = ols_with_hc3(dv, task_df[["task_iteration", "learning"]])
    lines.append(format_ols_summary(result_1b, "Model 1b: OLS — task_iteration + learning"))

    # Model 1c: Log-linear specification
    # log(DV) ~ augmentation_index (the 22-fold headline result)
    log_dv = np.log(dv)
    result_1c = ols_with_hc3(log_dv, task_df[["augmentation_index"]])
    lines.append(format_ols_summary(result_1c, "Model 1c: Log-linear — log(task_conversation_share) ~ augmentation_index"))

    # Interpretation of log-linear coefficient
    coef = result_1c.params["augmentation_index"]
    fold_change = np.exp(coef)
    lines.append(f"  Log-linear interpretation:")
    lines.append(f"  Coefficient = {coef:.4f}")
    lines.append(f"  exp(coef) = {fold_change:.2f}-fold change for unit increase in augmentation_index")
    lines.append("")

    return "\n".join(lines), {
        "1a": result_1a, "1b": result_1b, "1c": result_1c
    }


# ---------------------------------------------------------------------------
# Step 4: H2 — PMT (Task-level risk management)
# ---------------------------------------------------------------------------

def run_h2(task_df):
    """
    H2: Protection Motivation Theory (reframed) — risk-aware engagement.

    H2a: risk_management_index -> task_conversation_share (adoption intensity)
    H2b: risk_management_index -> directive (delegation style)
    H2c: risk_management_index -> task_iteration (collaboration style)

    Uses the deduplicated task-level sample (one row per unique O*NET task)
    to ensure independence of observations.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("H2: PROTECTION MOTIVATION THEORY (PMT) — TASK LEVEL")
    lines.append("=" * 70)

    dv = task_df["task_conversation_share"]

    # H2a: risk_management_index -> adoption intensity
    result_2a = ols_with_hc3(dv, task_df[["risk_management_index"]])
    lines.append(format_ols_summary(result_2a, "H2a: OLS — risk_management_index -> task_conversation_share"))

    # H2b: risk_management_index -> directive (delegation style)
    result_2b = ols_with_hc3(task_df["directive"], task_df[["risk_management_index"]])
    lines.append(format_ols_summary(result_2b, "H2b: OLS — risk_management_index -> directive"))

    # H2c: risk_management_index -> task_iteration (collaboration style)
    result_2c = ols_with_hc3(task_df["task_iteration"], task_df[["risk_management_index"]])
    lines.append(format_ols_summary(result_2c, "H2c: OLS — risk_management_index -> task_iteration"))

    return "\n".join(lines), {"2a": result_2a, "2b": result_2b, "2c": result_2c}


# ---------------------------------------------------------------------------
# Step 5: H3 — SET (Cluster-level trust)
# ---------------------------------------------------------------------------

def run_h3(cluster_df):
    """
    H3: Social Exchange Theory — collaborative interaction predicts trust.

    DV: cluster_thinking_fraction
    IV: cluster_augmentation_index
    """
    lines = []
    lines.append("=" * 70)
    lines.append("H3: SOCIAL EXCHANGE THEORY (SET) — CLUSTER LEVEL")
    lines.append("=" * 70)

    # Drop rows where DV is missing
    df = cluster_df.dropna(subset=["cluster_thinking_fraction"]).copy()
    lines.append(f"  Analytical sample: {len(df)} rows with non-null cluster_thinking_fraction")
    lines.append("")

    dv = df["cluster_thinking_fraction"]

    # Model 3: cluster_augmentation_index -> cluster_thinking_fraction
    result_3 = ols_with_hc3(dv, df[["cluster_augmentation_index"]])
    lines.append(format_ols_summary(result_3, "Model 3: OLS — cluster_augmentation_index"))

    return "\n".join(lines), {"3": result_3}


# ---------------------------------------------------------------------------
# Step 6: H4 — STS (Cluster-level socio-technical balance)
# ---------------------------------------------------------------------------

def run_h4(cluster_df):
    """
    H4: Socio-Technical Systems Theory — both automation and augmentation
    predict trust emergence.

    DV: cluster_thinking_fraction
    IVs: cluster_automation_index, cluster_augmentation_index
    """
    lines = []
    lines.append("=" * 70)
    lines.append("H4: SOCIO-TECHNICAL SYSTEMS THEORY (STS) — CLUSTER LEVEL")
    lines.append("=" * 70)

    df = cluster_df.dropna(subset=["cluster_thinking_fraction"]).copy()
    lines.append(f"  Analytical sample: {len(df)} rows with non-null cluster_thinking_fraction")
    lines.append("")

    dv = df["cluster_thinking_fraction"]

    # Model 4: both indices -> cluster_thinking_fraction
    result_4 = ols_with_hc3(
        dv, df[["cluster_automation_index", "cluster_augmentation_index"]]
    )
    lines.append(format_ols_summary(
        result_4,
        "Model 4: OLS — cluster_automation_index + cluster_augmentation_index"
    ))

    return "\n".join(lines), {"4": result_4}


# ---------------------------------------------------------------------------
# Step 7: H5 — Cross-level mediation (TAM × SET)
# ---------------------------------------------------------------------------

def run_h5(cluster_df):
    """
    H5: Cross-level mediation — augmentation -> cluster_thinking_fraction -> adoption.

    Baron & Kenny (1986) four-step procedure:
        Step 1: IV -> DV (total effect, path c)
        Step 2: IV -> Mediator (path a)
        Step 3: IV + Mediator -> DV (path c' and path b)
        Step 4: Sobel test for significance of indirect effect

    IV: augmentation_index (task-level, available in cluster_df)
    Mediator: cluster_thinking_fraction
    DV: task_conversation_share
    """
    lines = []
    lines.append("=" * 70)
    lines.append("H5: CROSS-LEVEL MEDIATION (TAM × SET)")
    lines.append("Baron & Kenny (1986) Four-Step Procedure with Sobel Test")
    lines.append("=" * 70)

    # Use rows where mediator is available
    df = cluster_df.dropna(subset=["cluster_thinking_fraction"]).copy()
    lines.append(f"  Analytical sample: {len(df)} rows with non-null cluster_thinking_fraction")
    lines.append("")

    iv = df["augmentation_index"]
    mediator = df["cluster_thinking_fraction"]
    dv = df["task_conversation_share"]

    # Step 1: Total effect (c path): IV -> DV
    step1 = ols_with_hc3(dv, iv)
    lines.append(format_ols_summary(step1, "Step 1 (Path c): augmentation_index -> task_conversation_share"))

    c = step1.params.iloc[1]  # coefficient for IV (index 0 is constant)
    c_p = step1.pvalues.iloc[1]
    lines.append(f"  Path c (total effect): {c:.6f} (p = {c_p:.2e})")
    lines.append("")

    # Step 2: IV -> Mediator (a path)
    step2 = ols_with_hc3(mediator, iv)
    lines.append(format_ols_summary(step2, "Step 2 (Path a): augmentation_index -> cluster_thinking_fraction"))

    a = step2.params.iloc[1]
    a_se = step2.bse.iloc[1]
    a_p = step2.pvalues.iloc[1]
    lines.append(f"  Path a: {a:.6f} (SE = {a_se:.6f}, p = {a_p:.2e})")
    lines.append("")

    # Step 3: IV + Mediator -> DV (c' and b paths)
    X_step3 = df[["augmentation_index", "cluster_thinking_fraction"]]
    step3 = ols_with_hc3(dv, X_step3)
    lines.append(format_ols_summary(
        step3,
        "Step 3 (Paths c' and b): augmentation_index + cluster_thinking_fraction -> task_conversation_share"
    ))

    c_prime = step3.params["augmentation_index"]
    c_prime_p = step3.pvalues["augmentation_index"]
    b = step3.params["cluster_thinking_fraction"]
    b_se = step3.bse["cluster_thinking_fraction"]
    b_p = step3.pvalues["cluster_thinking_fraction"]

    lines.append(f"  Path c' (direct effect): {c_prime:.6f} (p = {c_prime_p:.2e})")
    lines.append(f"  Path b: {b:.6f} (SE = {b_se:.6f}, p = {b_p:.2e})")
    lines.append("")

    # Step 4: Sobel test
    z, p, indirect = sobel_test(a, b, a_se, b_se)
    lines.append("--- Sobel Test ---")
    lines.append(f"  Indirect effect (a × b): {indirect:.6f}")
    lines.append(f"  Sobel z-statistic: {z:.4f}")
    lines.append(f"  Sobel p-value: {p:.2e}")
    lines.append("")

    # Mediation summary
    if c != 0:
        proportion_mediated = indirect / c
        lines.append(f"  Proportion mediated: {proportion_mediated:.4f} ({proportion_mediated*100:.1f}%)")
    lines.append("")

    # Baron & Kenny criteria check
    lines.append("--- Baron & Kenny Criteria ---")
    lines.append(f"  1. IV -> DV significant (c path)?       {'YES' if c_p < 0.05 else 'NO'} (p = {c_p:.2e})")
    lines.append(f"  2. IV -> Mediator significant (a path)?  {'YES' if a_p < 0.05 else 'NO'} (p = {a_p:.2e})")
    lines.append(f"  3. Mediator -> DV significant (b path)?  {'YES' if b_p < 0.05 else 'NO'} (p = {b_p:.2e})")
    lines.append(f"  4. Direct effect reduced (c' < c)?       {'YES' if abs(c_prime) < abs(c) else 'NO'} (c={c:.6f}, c'={c_prime:.6f})")
    if c_prime_p >= 0.05 and b_p < 0.05:
        lines.append(f"  Conclusion: FULL MEDIATION (c' not significant, b significant)")
    elif abs(c_prime) < abs(c) and b_p < 0.05:
        lines.append(f"  Conclusion: PARTIAL MEDIATION (c' reduced but still significant)")
    else:
        lines.append(f"  Conclusion: MEDIATION NOT ESTABLISHED by Baron & Kenny criteria")
    lines.append("")

    return "\n".join(lines), {
        "step1": step1, "step2": step2, "step3": step3,
        "sobel_z": z, "sobel_p": p, "indirect": indirect,
    }


# ---------------------------------------------------------------------------
# Step 8: H6 — SET × STS (Cluster-level adoption breadth)
# ---------------------------------------------------------------------------

def run_h6(cluster_df):
    """
    H6: Social Exchange Theory × Socio-Technical Systems Theory.

    DV: percent_users (adoption breadth)
    IVs: cluster_augmentation_index, cluster_thinking_fraction
    Moderation: interaction term (mean-centered)

    Hierarchical OLS:
        Model 6a: cluster_augmentation_index only
        Model 6b: + cluster_thinking_fraction
        Model 6c: + interaction term (mean-centered)
    """
    lines = []
    lines.append("=" * 70)
    lines.append("H6: SET × STS — CLUSTER-LEVEL ADOPTION BREADTH")
    lines.append("Hierarchical OLS with Mean-Centered Interaction")
    lines.append("=" * 70)

    df = cluster_df.dropna(
        subset=["percent_users", "cluster_augmentation_index", "cluster_thinking_fraction"]
    ).copy()
    lines.append(f"  Analytical sample: {len(df)} rows with complete data")
    lines.append("")

    dv = df["percent_users"]

    # Mean-center predictors for interaction term
    df["aug_centered"] = (
        df["cluster_augmentation_index"] - df["cluster_augmentation_index"].mean()
    )
    df["think_centered"] = (
        df["cluster_thinking_fraction"] - df["cluster_thinking_fraction"].mean()
    )
    df["aug_x_think"] = df["aug_centered"] * df["think_centered"]

    # Model 6a: main effect of augmentation only
    result_6a = ols_with_hc3(dv, df[["cluster_augmentation_index"]])
    lines.append(format_ols_summary(result_6a, "Model 6a: cluster_augmentation_index only"))

    # Model 6b: add thinking fraction
    result_6b = ols_with_hc3(
        dv, df[["cluster_augmentation_index", "cluster_thinking_fraction"]]
    )
    lines.append(format_ols_summary(result_6b, "Model 6b: + cluster_thinking_fraction"))

    # Model 6c: add interaction (mean-centered)
    result_6c = ols_with_hc3(
        dv, df[["aug_centered", "think_centered", "aug_x_think"]]
    )
    lines.append(format_ols_summary(
        result_6c,
        "Model 6c: + interaction (mean-centered augmentation × thinking)"
    ))

    # R-squared change
    lines.append("--- Hierarchical Model Comparison ---")
    lines.append(f"  Model 6a R²: {result_6a.rsquared:.6f}")
    lines.append(f"  Model 6b R²: {result_6b.rsquared:.6f} (ΔR² = {result_6b.rsquared - result_6a.rsquared:.6f})")
    lines.append(f"  Model 6c R²: {result_6c.rsquared:.6f} (ΔR² = {result_6c.rsquared - result_6b.rsquared:.6f})")
    lines.append("")

    return "\n".join(lines), {
        "6a": result_6a, "6b": result_6b, "6c": result_6c,
    }


# ---------------------------------------------------------------------------
# Step 9: Robustness checks
# ---------------------------------------------------------------------------

def run_robustness_checks(task_df, cluster_df):
    """
    Supplementary robustness checks:
        - Fractional logit for H1/H2 (bounded DV)
        - Sensitivity checks for H3 (controls, influential observations)
    """
    lines = []
    lines.append("=" * 70)
    lines.append("ROBUSTNESS CHECKS")
    lines.append("=" * 70)

    # ----- Fractional logit for H1 -----
    lines.append("\n--- Fractional Logit: H1 (augmentation_index -> task_conversation_share) ---\n")

    # task_conversation_share is each task's percentage share of ALL
    # conversations (sums to ~100 across tasks). Dividing by 100 yields the
    # true proportion in (0,1) — the natural scale for fractional logit
    # (Papke & Wooldridge, 1996). Complete cases only: GLM cannot accept NaN.
    frac_df = task_df.dropna(
        subset=["task_conversation_share", "augmentation_index",
                "risk_management_index"]
    ).copy()
    dv_frac = frac_df["task_conversation_share"] / 100.0

    X_frac = sm.add_constant(frac_df[["augmentation_index"]])
    try:
        frac_logit = GLM(
            dv_frac, X_frac,
            family=Binomial(link=Logit())
        ).fit(cov_type="HC3")
        lines.append(frac_logit.summary().as_text())
    except Exception as e:
        lines.append(f"  Fractional logit failed: {e}")
    lines.append("")

    # ----- Fractional logit for H2 -----
    lines.append("\n--- Fractional Logit: H2 (risk_management_index -> task_conversation_share) ---\n")
    X_frac2 = sm.add_constant(frac_df[["risk_management_index"]])
    try:
        frac_logit2 = GLM(
            dv_frac, X_frac2,
            family=Binomial(link=Logit())
        ).fit(cov_type="HC3")
        lines.append(frac_logit2.summary().as_text())
    except Exception as e:
        lines.append(f"  Fractional logit failed: {e}")
    lines.append("")

    # ----- H3 sensitivity: controls for user breadth and activity volume -----
    lines.append("\n--- H3 Sensitivity: Adding Controls ---\n")

    df_h3 = cluster_df.dropna(subset=["cluster_thinking_fraction"]).copy()

    # Control 1: Add percent_users (user breadth)
    if "percent_users" in df_h3.columns:
        df_ctrl = df_h3.dropna(subset=["percent_users"])
        dv_h3 = df_ctrl["cluster_thinking_fraction"]
        X_ctrl1 = df_ctrl[["cluster_augmentation_index", "percent_users"]]
        result_ctrl1 = ols_with_hc3(dv_h3, X_ctrl1)
        lines.append(format_ols_summary(
            result_ctrl1,
            "H3 + control: percent_users"
        ))

    # Control 2: Add percent_records (activity volume)
    if "percent_records" in df_h3.columns:
        df_ctrl = df_h3.dropna(subset=["percent_records"])
        dv_h3 = df_ctrl["cluster_thinking_fraction"]
        X_ctrl2 = df_ctrl[["cluster_augmentation_index", "percent_records"]]
        result_ctrl2 = ols_with_hc3(dv_h3, X_ctrl2)
        lines.append(format_ols_summary(
            result_ctrl2,
            "H3 + control: percent_records"
        ))

    # Control 3: Exclude high-influence observations (Cook's D > 4/n)
    lines.append("\n--- H3 Sensitivity: Excluding High-Influence Observations ---\n")
    # Work on the complete-case subset so Cook's D indices align with rows
    h3_vars = ["cluster_thinking_fraction", "cluster_augmentation_index"]
    df_h3_complete = df_h3.dropna(subset=h3_vars).copy()
    dv_h3_full = df_h3_complete["cluster_thinking_fraction"]
    X_h3_full = sm.add_constant(df_h3_complete[["cluster_augmentation_index"]])
    model_h3_full = sm.OLS(dv_h3_full, X_h3_full).fit()
    influence = model_h3_full.get_influence()
    cooks_d = influence.cooks_distance[0]
    threshold = 4 / len(df_h3_complete)
    high_influence_mask = cooks_d > threshold
    n_excluded = high_influence_mask.sum()
    lines.append(f"  Cook's D threshold: {threshold:.6f}")
    lines.append(f"  High-influence observations excluded: {n_excluded}")

    df_h3_trimmed = df_h3_complete[~high_influence_mask]
    if len(df_h3_trimmed) > 10:
        result_trimmed = ols_with_hc3(
            df_h3_trimmed["cluster_thinking_fraction"],
            df_h3_trimmed[["cluster_augmentation_index"]]
        )
        lines.append(format_ols_summary(
            result_trimmed,
            f"H3 excluding {n_excluded} high-influence observations"
        ))
    lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("Starting analysis...")

    # Load data
    task_df, cluster_df = load_data()

    # Create output directory
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Step 1: Descriptive statistics
    logger.info("Computing descriptive statistics...")
    desc_text = compute_descriptives(task_df, cluster_df)
    with open(os.path.join(RESULTS_DIR, "descriptives.txt"), "w") as f:
        f.write(desc_text)
    logger.info("  -> results/descriptives.txt")

    # Step 2: Correlations
    logger.info("Computing correlation matrices...")
    task_corr, task_pvals, cluster_corr = compute_correlations(task_df, cluster_df)
    task_corr.to_csv(os.path.join(RESULTS_DIR, "correlations_task.csv"))
    cluster_corr.to_csv(os.path.join(RESULTS_DIR, "correlations_cluster.csv"))
    logger.info("  -> results/correlations_task.csv, correlations_cluster.csv")

    # Step 3-8: Hypothesis tests
    all_results = {}

    logger.info("Running H1 (TAM — task-level adoption)...")
    h1_text, h1_models = run_h1(task_df)
    with open(os.path.join(RESULTS_DIR, "H1_results.txt"), "w") as f:
        f.write(h1_text)
    all_results["H1"] = h1_models
    logger.info("  -> results/H1_results.txt")

    logger.info("Running H2 (PMT — risk management)...")
    h2_text, h2_models = run_h2(task_df)
    with open(os.path.join(RESULTS_DIR, "H2_results.txt"), "w") as f:
        f.write(h2_text)
    all_results["H2"] = h2_models
    logger.info("  -> results/H2_results.txt")

    logger.info("Running H3 (SET — cluster-level trust)...")
    h3_text, h3_models = run_h3(cluster_df)
    with open(os.path.join(RESULTS_DIR, "H3_results.txt"), "w") as f:
        f.write(h3_text)
    all_results["H3"] = h3_models
    logger.info("  -> results/H3_results.txt")

    logger.info("Running H4 (STS — socio-technical balance)...")
    h4_text, h4_models = run_h4(cluster_df)
    with open(os.path.join(RESULTS_DIR, "H4_results.txt"), "w") as f:
        f.write(h4_text)
    all_results["H4"] = h4_models
    logger.info("  -> results/H4_results.txt")

    logger.info("Running H5 (cross-level mediation)...")
    h5_text, h5_models = run_h5(cluster_df)
    with open(os.path.join(RESULTS_DIR, "H5_mediation_results.txt"), "w") as f:
        f.write(h5_text)
    all_results["H5"] = h5_models
    logger.info("  -> results/H5_mediation_results.txt")

    logger.info("Running H6 (SET × STS — adoption breadth)...")
    h6_text, h6_models = run_h6(cluster_df)
    with open(os.path.join(RESULTS_DIR, "H6_results.txt"), "w") as f:
        f.write(h6_text)
    all_results["H6"] = h6_models
    logger.info("  -> results/H6_results.txt")

    # Step 9: Robustness checks
    logger.info("Running robustness checks...")
    robust_text = run_robustness_checks(task_df, cluster_df)
    with open(os.path.join(RESULTS_DIR, "robustness_checks.txt"), "w") as f:
        f.write(robust_text)
    logger.info("  -> results/robustness_checks.txt")

    # Analysis log
    log_lines = [
        "=" * 70,
        "ANALYSIS LOG",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "=" * 70,
        "",
        "MODELS RUN",
        f"  H1 (TAM):  3 models (OLS, OLS components, log-linear)",
        f"  H2 (PMT):  1 model (OLS)",
        f"  H3 (SET):  1 model (OLS)",
        f"  H4 (STS):  1 model (OLS)",
        f"  H5 (Mediation): 3-step Baron & Kenny + Sobel test",
        f"  H6 (SET×STS): 3 models (hierarchical OLS with interaction)",
        "",
        "ROBUSTNESS CHECKS",
        f"  Fractional logit: H1, H2",
        f"  H3 sensitivity: controls (percent_users, percent_records), Cook's D trimming",
        "",
        "ALL MODELS USE HC3 ROBUST STANDARD ERRORS",
        "",
        "=" * 70,
    ]
    with open(os.path.join(RESULTS_DIR, "analysis_log.txt"), "w") as f:
        f.write("\n".join(log_lines))

    logger.info("Analysis complete. All results saved to results/")


if __name__ == "__main__":
    main()
