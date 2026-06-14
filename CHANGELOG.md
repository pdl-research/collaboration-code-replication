# Changelog

## v2.1 — Replication package for public release (2026-06-14)

Packaging-only update; no analysis results changed.

### Fixed
- **`make_figures.py`** used absolute, machine-specific paths and so failed on
  any other machine. Paths are now resolved relative to the script location, so
  the figures regenerate from `./data/` on a fresh clone.

### Added
- `matplotlib` added to `requirements.txt` (needed by `make_figures.py`).
- Public CC-BY source files included under `data/raw/`, so the build runs fully
  offline; `.gitignore` updated to track `data/raw/` while ignoring the
  generated build outputs and `results/`.
- `README.md` updated to document `make_figures.py` and the offline build.
- Supplementary Materials markdown refreshed to include the new Robustness
  Checks section (model comparison, high-share-task exclusion, clustered SEs,
  bootstrap CIs, Cook's-D, H3 complexity proxies; Supplementary Tables 3-4).

## v2.0 — Revise & Resubmit update (2026-06-05)

Extensions added in response to peer review. All v1.0 results remain
reproducible via `01_build_dataset.py` + `02_analysis.py`.

### Fixed
- **`02_analysis.py` fractional logit robustness checks previously failed**
  with "exog contains inf or nans" (one task row carries NaN interaction
  components). Now estimated on explicit complete cases (n = 3,364).
- Fractional logit DV rescaling: previously divided by `max × 1.01` (ad hoc);
  now divides by 100, since `task_conversation_share` is each task's
  percentage of all conversations (sums to ~100), making DV/100 the true
  proportion in (0,1) per Papke & Wooldridge (1996).

### Added
- **`03_reviewer_models.py`** — model comparison suite for the highly skewed
  DV (OLS, log-linear OLS with Duan smearing, fractional logit, GLM
  Gamma(log), Poisson QMLE) with raw-scale RMSE comparison, plus robustness
  checks excluding high-share tasks (> 1.0 share; 7 in the complete-case
  sample). Output: `results/batchA_reviewer_models.txt`.
- **`04_batchB.py`** — (1) cluster-robust standard errors grouped by
  `task_name` (sensitivity: by intermediate cluster `cluster_name_1`) for all
  cluster-level models, addressing non-independence of the 593 cluster rows
  derived from 346 unique tasks; (2) bootstrapped percentile CIs for the H5
  indirect effect (5,000 replications, seed 42; IID and task-cluster
  resampling), replacing the Baron–Kenny/Sobel approach; (3) H3 sensitivity
  analyses with complexity/novelty proxies (technical-domain indicator,
  feedback-loop intensity, volume, breadth). Output:
  `results/batchB_results.txt`.
- **`EXPECTED_RESULTS.md`** — key coefficients from a verified run, so
  replicators can check their output.

### Notes for replicators
- Interaction-pattern proportions at the cluster level sum to ~1, so
  composite indices built from them are compositionally dependent
  (e.g., corr(augmentation index, feedback loop) ≈ −.62). Interpret models
  that include multiple composition components with care.
- Of 3,365 unique tasks, 1 row has NaN interaction components; regression
  samples are therefore n = 3,364.
