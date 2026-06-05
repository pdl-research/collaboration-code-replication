"""
03_reviewer_models.py — Batch A of reviewer-requested extensions.

Addresses:
  - Reviewer 1, point 4: model comparison suite for the highly skewed DV,
    with goodness-of-fit statistics to justify model choice.
  - Reviewer 2: robustness checks excluding high-share tasks
    (task_conversation_share > 1.0) for H1 and H2a. Note: the manuscript's
    "62 tasks" figure was an artifact of the duplicated integrated dataset
    (63 rows); the deduplicated sample contains 8 such unique tasks
    (7 in the complete-case sample).
  - Repo bug fix: fractional logit previously failed on NaN rows.

Run AFTER 01_build_dataset.py. Outputs to results/.
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial, Gamma, Poisson
from statsmodels.genmod.families.links import Logit, Log

DATA_DIR = "data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

RNG_SEED = 42


def fit_ols(y, X):
    return sm.OLS(y, sm.add_constant(X)).fit(cov_type="HC3")


def fit_glm(y, X, family):
    return GLM(y, sm.add_constant(X), family=family).fit(cov_type="HC3")


def coef_line(res, name):
    return (f"    {name}: coef={res.params[name]:.4f}, SE={res.bse[name]:.4f}, "
            f"p={res.pvalues[name]:.3e}")


def main():
    lines = []
    task = pd.read_csv(os.path.join(DATA_DIR, "task_level_dataset.csv"))

    # ----- Complete-case sample (fixes the NaN failure) -----
    cols = ["task_conversation_share", "augmentation_index",
            "risk_management_index", "task_iteration", "learning"]
    df = task.dropna(subset=cols).copy()
    lines.append("=" * 70)
    lines.append("BATCH A: REVIEWER-REQUESTED MODEL EXTENSIONS (TASK LEVEL)")
    lines.append("=" * 70)
    lines.append(f"Complete-case sample: n={len(df)} of {len(task)} task rows")
    lines.append("")

    y = df["task_conversation_share"]
    # DV as a true proportion of all conversations: share/100 in (0,1)
    y_prop = y / 100.0
    assert (y_prop > 0).all() and (y_prop < 1).all(), "DV/100 not in (0,1)"
    lines.append(f"DV check: task_conversation_share sums to {y.sum():.2f} "
                 f"(expected ~100); /100 lies in (0,1): OK")
    lines.append("")

    # =====================================================================
    # PART 1: MODEL COMPARISON SUITE (Reviewer 1.4)
    # One IV (augmentation_index) across five estimators.
    # =====================================================================
    lines.append("=" * 70)
    lines.append("PART 1: MODEL COMPARISON FOR SKEWED DV (augmentation_index)")
    lines.append("=" * 70)
    X = df[["augmentation_index"]]

    comparison = []

    # 1. OLS on raw DV
    m_ols = fit_ols(y, X)
    comparison.append(("OLS (raw DV)", m_ols.aic, m_ols.bic,
                       m_ols.rsquared, np.sqrt(np.mean(m_ols.resid ** 2))))
    lines.append("\n[1] OLS on raw DV")
    lines.append(coef_line(m_ols, "augmentation_index"))

    # 2. Log-linear OLS
    m_log = fit_ols(np.log(y), X)
    # RMSE on original scale via Duan smearing for fair comparison
    smear = np.mean(np.exp(m_log.resid))
    yhat_log = np.exp(m_log.fittedvalues) * smear
    comparison.append(("Log-linear OLS", m_log.aic, m_log.bic,
                       m_log.rsquared, np.sqrt(np.mean((y - yhat_log) ** 2))))
    lines.append("\n[2] Log-linear OLS (log DV)")
    lines.append(coef_line(m_log, "augmentation_index"))
    lines.append(f"    exp(coef) = {np.exp(m_log.params['augmentation_index']):.2f}; "
                 f"Duan smearing factor = {smear:.4f}")
    lines.append("    NOTE: log-scale AIC/BIC/R2 not directly comparable to raw-scale models;")
    lines.append("    RMSE column is back-transformed to the raw scale for comparability.")

    # 3. Fractional logit (Papke-Wooldridge QMLE) on DV/100
    m_frac = fit_glm(y_prop, X, Binomial(link=Logit()))
    pr2_frac = 1 - m_frac.deviance / m_frac.null_deviance
    yhat_frac = m_frac.fittedvalues * 100.0
    comparison.append(("Fractional logit (DV/100)", m_frac.aic, m_frac.bic_llf,
                       pr2_frac, np.sqrt(np.mean((y - yhat_frac) ** 2))))
    lines.append("\n[3] Fractional logit, GLM Binomial(logit), QMLE, HC3")
    lines.append(coef_line(m_frac, "augmentation_index"))
    lines.append(f"    Pseudo-R2 (deviance) = {pr2_frac:.4f}")

    # 4. GLM Gamma with log link (positive continuous, skewed)
    m_gamma = fit_glm(y, X, Gamma(link=Log()))
    pr2_gamma = 1 - m_gamma.deviance / m_gamma.null_deviance
    comparison.append(("GLM Gamma (log link)", m_gamma.aic, m_gamma.bic_llf,
                       pr2_gamma, np.sqrt(np.mean((y - m_gamma.fittedvalues) ** 2))))
    lines.append("\n[4] GLM Gamma(log link), HC3")
    lines.append(coef_line(m_gamma, "augmentation_index"))
    lines.append(f"    exp(coef) = {np.exp(m_gamma.params['augmentation_index']):.2f}")
    lines.append(f"    Pseudo-R2 (deviance) = {pr2_gamma:.4f}")

    # 5. Poisson QMLE with log link (consistent for nonneg continuous DV w/ robust SE)
    m_pois = fit_glm(y, X, Poisson(link=Log()))
    pr2_pois = 1 - m_pois.deviance / m_pois.null_deviance
    comparison.append(("Poisson QMLE (log link)", m_pois.aic, m_pois.bic_llf,
                       pr2_pois, np.sqrt(np.mean((y - m_pois.fittedvalues) ** 2))))
    lines.append("\n[5] Poisson QMLE (log link), HC3 — Santos Silva & Tenreyro (2006) style")
    lines.append(coef_line(m_pois, "augmentation_index"))
    lines.append(f"    exp(coef) = {np.exp(m_pois.params['augmentation_index']):.2f}")
    lines.append(f"    Pseudo-R2 (deviance) = {pr2_pois:.4f}")

    # Comparison table
    lines.append("\n--- GOODNESS-OF-FIT COMPARISON ---")
    lines.append(f"{'Model':<28} {'AIC':>12} {'BIC':>12} {'(Pseudo-)R2':>12} {'RMSE(raw)':>10}")
    for name, aic, bic, r2, rmse in comparison:
        lines.append(f"{name:<28} {aic:>12.1f} {bic:>12.1f} {r2:>12.4f} {rmse:>10.4f}")
    lines.append("")

    # Repeat headline estimators for H2a
    lines.append("=" * 70)
    lines.append("PART 1b: SAME SUITE FOR H2a (risk_management_index)")
    lines.append("=" * 70)
    X2 = df[["risk_management_index"]]
    for label, fam, dv in [("Fractional logit", Binomial(link=Logit()), y_prop),
                           ("GLM Gamma(log)", Gamma(link=Log()), y),
                           ("Poisson QMLE(log)", Poisson(link=Log()), y)]:
        m = fit_glm(dv, X2, fam)
        pr2 = 1 - m.deviance / m.null_deviance
        lines.append(f"\n[{label}]")
        lines.append(coef_line(m, "risk_management_index"))
        lines.append(f"    exp(coef) = {np.exp(m.params['risk_management_index']):.2f}; "
                     f"Pseudo-R2 = {pr2:.4f}")
    m2_log = fit_ols(np.log(y), X2)
    lines.append("\n[Log-linear OLS]")
    lines.append(coef_line(m2_log, "risk_management_index"))
    lines.append(f"    exp(coef) = {np.exp(m2_log.params['risk_management_index']):.2f}, "
                 f"R2 = {m2_log.rsquared:.4f}")
    lines.append("")

    # =====================================================================
    # PART 2: OUTLIER ROBUSTNESS — exclude tasks with share > 1.0 (R2)
    # =====================================================================
    lines.append("=" * 70)
    lines.append("PART 2: ROBUSTNESS EXCLUDING HIGH-SHARE TASKS (share > 1.0)")
    lines.append("=" * 70)
    high = df["task_conversation_share"] > 1.0
    df_trim = df[~high]
    lines.append(f"Excluded: {high.sum()} tasks (expected 7 in complete-case "
                 f"sample; manuscript's '62' was a duplicate-dataset artifact); "
                 f"retained n={len(df_trim)}")
    yt = df_trim["task_conversation_share"]

    # H1 log-linear
    m = fit_ols(np.log(yt), df_trim[["augmentation_index"]])
    lines.append("\n[H1 log-linear, trimmed]")
    lines.append(coef_line(m, "augmentation_index"))
    lines.append(f"    exp(coef) = {np.exp(m.params['augmentation_index']):.2f}, "
                 f"R2 = {m.rsquared:.4f}")

    # H1 OLS components
    m = fit_ols(yt, df_trim[["task_iteration", "learning"]])
    lines.append("\n[H1 OLS components, trimmed]")
    lines.append(coef_line(m, "task_iteration"))
    lines.append(coef_line(m, "learning"))
    lines.append(f"    R2 = {m.rsquared:.4f}")

    # H2a OLS + log-linear
    m = fit_ols(yt, df_trim[["risk_management_index"]])
    lines.append("\n[H2a OLS, trimmed]")
    lines.append(coef_line(m, "risk_management_index"))
    lines.append(f"    R2 = {m.rsquared:.4f}")

    m = fit_ols(np.log(yt), df_trim[["risk_management_index"]])
    lines.append("\n[H2a log-linear, trimmed]")
    lines.append(coef_line(m, "risk_management_index"))
    lines.append(f"    exp(coef) = {np.exp(m.params['risk_management_index']):.2f}, "
                 f"R2 = {m.rsquared:.4f}")

    # Fractional logit on trimmed sample (H1)
    m = fit_glm(yt / 100.0, df_trim[["augmentation_index"]], Binomial(link=Logit()))
    lines.append("\n[H1 fractional logit, trimmed]")
    lines.append(coef_line(m, "augmentation_index"))
    lines.append("")

    out = os.path.join(RESULTS_DIR, "batchA_reviewer_models.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"Written: {out}")
    print("\n".join(lines[-60:]))


if __name__ == "__main__":
    main()
