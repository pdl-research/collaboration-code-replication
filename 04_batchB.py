"""
04_batchB.py — Batch B of reviewer-requested extensions.

Addresses:
  - Reviewer 2: non-independence of 593 cluster rows (346 unique tasks):
    cluster-robust SEs grouped by task_name for H3, H4, H6 (sensitivity:
    grouped by intermediate cluster, cluster_name_1).
  - Reviewer 2: bootstrapped CIs for the H5 indirect effect (replaces
    Baron-Kenny/Sobel). IID and task-cluster bootstrap, 5,000 reps, seed 42.
  - Reviewer 1.2: H3 sensitivity with complexity/novelty proxies
    (technical-domain dummy, feedback-loop intensity, volume, breadth).

Run AFTER 01_build_dataset.py. Outputs to results/batchB_results.txt.
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_DIR = "data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
SEED = 42
N_BOOT = 5000

TECH_KEYWORDS = ("software", "code", "coding", "debug", "program", "script",
                 "data", "technical", "computer", "algorithm", "web", "app")


def fit(y, X, cov_type="HC3", groups=None):
    Xc = sm.add_constant(X)
    model = sm.OLS(y, Xc)
    if cov_type == "cluster":
        return model.fit(cov_type="cluster", cov_kwds={"groups": groups})
    return model.fit(cov_type="HC3")


def report(res, names, label):
    out = [f"\n[{label}]  n={int(res.nobs)}, R2={res.rsquared:.4f}, cov={res.cov_type}"]
    for nm in names:
        out.append(f"    {nm}: b={res.params[nm]:.4f}, SE={res.bse[nm]:.4f}, "
                   f"p={res.pvalues[nm]:.3e}")
    return out


def main():
    np.random.seed(SEED)
    lines = ["=" * 70, "BATCH B: CLUSTERED SEs, BOOTSTRAP MEDIATION, H3 CONTROLS", "=" * 70]

    df = pd.read_csv(os.path.join(DATA_DIR, "cluster_level_dataset.csv"))
    d = df.dropna(subset=["cluster_thinking_fraction"]).copy()
    lines.append(f"Cluster rows with thinking data: {len(d)} (of {len(df)}); "
                 f"unique tasks: {d['task_name'].nunique()}")

    # =====================================================================
    # PART 1: CLUSTER-ROBUST SEs (groups = task_name)
    # =====================================================================
    lines.append("\n" + "=" * 70)
    lines.append("PART 1: H3/H4/H6 WITH TASK-CLUSTERED STANDARD ERRORS")
    lines.append("=" * 70)

    # --- H3 ---
    h3 = d.dropna(subset=["cluster_augmentation_index"])
    y3, X3 = h3["cluster_thinking_fraction"], h3[["cluster_augmentation_index"]]
    lines += report(fit(y3, X3), ["cluster_augmentation_index"], "H3 baseline HC3")
    lines += report(fit(y3, X3, "cluster", h3["task_name"]),
                    ["cluster_augmentation_index"], "H3 clustered by task_name")
    lines += report(fit(y3, X3, "cluster", h3["cluster_name_1"]),
                    ["cluster_augmentation_index"], "H3 clustered by cluster_name_1 (143 groups)")

    # --- H4 ---
    h4 = d.dropna(subset=["cluster_automation_index", "cluster_augmentation_index"])
    y4 = h4["cluster_thinking_fraction"]
    X4 = h4[["cluster_automation_index", "cluster_augmentation_index"]]
    nm4 = list(X4.columns)
    lines += report(fit(y4, X4), nm4, "H4 baseline HC3")
    lines += report(fit(y4, X4, "cluster", h4["task_name"]), nm4, "H4 clustered by task_name")

    # --- H6 (hierarchical, mean-centered interaction) ---
    h6 = d.dropna(subset=["percent_users", "cluster_augmentation_index",
                          "cluster_thinking_fraction"]).copy()
    h6["aug_c"] = h6["cluster_augmentation_index"] - h6["cluster_augmentation_index"].mean()
    h6["thk_c"] = h6["cluster_thinking_fraction"] - h6["cluster_thinking_fraction"].mean()
    h6["interaction"] = h6["aug_c"] * h6["thk_c"]
    y6 = h6["percent_users"]
    for label, cols in [("H6 Model a", ["aug_c"]),
                        ("H6 Model b", ["aug_c", "thk_c"]),
                        ("H6 Model c", ["aug_c", "thk_c", "interaction"])]:
        lines += report(fit(y6, h6[cols], "cluster", h6["task_name"]), cols,
                        f"{label} clustered by task_name")

    # =====================================================================
    # PART 2: H5 BOOTSTRAPPED MEDIATION (replaces Sobel)
    # =====================================================================
    lines.append("\n" + "=" * 70)
    lines.append(f"PART 2: H5 BOOTSTRAP MEDIATION ({N_BOOT} reps, seed {SEED})")
    lines.append("=" * 70)

    h5 = d.dropna(subset=["augmentation_index", "cluster_thinking_fraction",
                          "task_conversation_share"]).copy()
    lines.append(f"H5 sample: n={len(h5)}, unique tasks: {h5['task_name'].nunique()}")

    def paths(frame):
        a = sm.OLS(frame["cluster_thinking_fraction"],
                   sm.add_constant(frame["augmentation_index"])).fit().params.iloc[1]
        m3 = sm.OLS(frame["task_conversation_share"],
                    sm.add_constant(frame[["augmentation_index",
                                           "cluster_thinking_fraction"]])).fit()
        b = m3.params["cluster_thinking_fraction"]
        c_prime = m3.params["augmentation_index"]
        return a, b, a * b, c_prime

    a0, b0, ind0, cp0 = paths(h5)
    lines.append(f"Point estimates: a={a0:.5f}, b={b0:.4f}, "
                 f"indirect a*b={ind0:.5f}, c'={cp0:.4f}")


    rng = np.random.default_rng(SEED)
    n = len(h5)

    # Numpy-based fast paths computation
    IV = h5["augmentation_index"].to_numpy()
    MED = h5["cluster_thinking_fraction"].to_numpy()
    DV = h5["task_conversation_share"].to_numpy()
    task_codes = pd.Categorical(h5["task_name"]).codes
    n_tasks_u = task_codes.max() + 1
    task_rows = [np.where(task_codes == t)[0] for t in range(n_tasks_u)]

    def indirect_np(idx):
        iv, med, dv = IV[idx], MED[idx], DV[idx]
        Xa = np.column_stack([np.ones(len(iv)), iv])
        a = np.linalg.lstsq(Xa, med, rcond=None)[0][1]
        Xb = np.column_stack([np.ones(len(iv)), iv, med])
        b = np.linalg.lstsq(Xb, dv, rcond=None)[0][2]
        return a * b

    # (i) IID bootstrap
    iid = np.array([indirect_np(rng.integers(0, n, n)) for _ in range(N_BOOT)])
    lo, hi = np.percentile(iid, [2.5, 97.5])
    lines.append(f"\nIID bootstrap ({len(iid)} valid): indirect 95% CI "
                 f"[{lo:.5f}, {hi:.5f}]  {'EXCLUDES 0' if lo*hi > 0 else 'includes 0'}")

    # (ii) Cluster bootstrap by task_name (resample tasks with replacement)
    cb = np.empty(N_BOOT)
    for i in range(N_BOOT):
        chosen = rng.integers(0, n_tasks_u, n_tasks_u)
        idx = np.concatenate([task_rows[t] for t in chosen])
        cb[i] = indirect_np(idx)
    lo_c, hi_c = np.percentile(cb, [2.5, 97.5])
    lines.append(f"Task-cluster bootstrap ({len(cb)} valid): indirect 95% CI "
                 f"[{lo_c:.5f}, {hi_c:.5f}]  {'EXCLUDES 0' if lo_c*hi_c > 0 else 'includes 0'}")

    # =====================================================================
    # PART 3: H3 WITH COMPLEXITY / NOVELTY PROXIES (Reviewer 1.2)
    # =====================================================================
    lines.append("\n" + "=" * 70)
    lines.append("PART 3: H3 SENSITIVITY — COMPLEXITY/NOVELTY PROXIES")
    lines.append("=" * 70)

    h3c = h3.copy()
    namecols = (h3c["cluster_name_0"].fillna("") + " " +
                h3c["cluster_name_1"].fillna("") + " " +
                h3c["cluster_name_2"].fillna("")).str.lower()
    h3c["technical_domain"] = namecols.apply(
        lambda s: int(any(k in s for k in TECH_KEYWORDS)))
    lines.append(f"Technical-domain clusters: {h3c['technical_domain'].sum()} of {len(h3c)}")

    ctrl_sets = [
        ("(+ technical domain)", ["cluster_augmentation_index", "technical_domain"]),
        ("(+ feedback-loop intensity)", ["cluster_augmentation_index", "cluster_feedback_loop"]),
        ("(+ volume & breadth)", ["cluster_augmentation_index", "percent_records", "percent_users"]),
        ("(full controls)", ["cluster_augmentation_index", "technical_domain",
                             "cluster_feedback_loop", "percent_records", "percent_users"]),
    ]
    for label, cols in ctrl_sets:
        sub = h3c.dropna(subset=cols)
        res = fit(sub["cluster_thinking_fraction"], sub[cols],
                  "cluster", sub["task_name"])
        lines += report(res, cols, f"H3 {label}, task-clustered SEs")

    out = os.path.join(RESULTS_DIR, "batchB_results.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"Written: {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
