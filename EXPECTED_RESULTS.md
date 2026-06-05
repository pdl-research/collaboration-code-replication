# Expected Results

Key values from a verified run (Python 3.11, pinned packages). Your numbers
should match to the precision shown. Seed-dependent values (bootstrap CIs)
match exactly only with seed 42 and numpy's default_rng.

## Build verification (`data/build_log.txt`)

| Check | Expected |
|---|---|
| Integrated rows | 3,612 |
| Task-level rows | 3,365 |
| Cluster-level rows | 593 |
| Unique tasks | 3,365 |
| Tasks with clusters | 346 |
| Multi-cluster tasks | 94 |
| task_conversation_share (dedup) | mean 0.0297, max 6.65, sums to ~100 |

## 02_analysis.py (HC3 robust SEs; task models n = 3,364)

| Model | Coefficient | Value | p |
|---|---|---|---|
| H1 OLS | augmentation_index | 0.0596 | <.001 |
| H1 OLS components | task_iteration / learning | 0.0794 / 0.0401 | <.001 / <.001 |
| H1 log-linear | augmentation_index | 2.8144 (R² = .399) | <.001 |
| H2a OLS | risk_management_index | 0.6649 (R² = .085) | <.001 |
| H2b OLS | risk_management_index | 0.2630 | <.001 |
| H2c OLS | risk_management_index | 0.1553 | <.001 |
| H3 (n=508) | cluster_augmentation_index | −0.0611 (R² = .107) | <.001 |
| H4 (n=488) | automation / augmentation | 0.0016 / −0.0579 | .974 / .244 |
| H5 paths | a / b / c′ | −0.0169 / 6.194 / −0.903 | .007 / <.001 / <.001 |
| H6 Model a (n=508) | cluster_augmentation_index | −0.0274 (R² = .013) | .012 |
| Fractional logit H1 | augmentation_index | 2.3019 | <.001 |

## 03_reviewer_models.py (`results/batchA_reviewer_models.txt`)

| Model (H1, augmentation_index) | coef | (pseudo-)R² |
|---|---|---|
| Fractional logit (DV/100) | 2.302 | .099 |
| GLM Gamma (log) | 4.396 | .303 |
| Poisson QMLE (log) | 2.301 | .099 |
| Log-linear OLS, trimmed (share ≤ 1.0, n = 3,357) | 2.804 (R² = .414) | — |

High-share tasks excluded in trimming: 7 (complete-case sample).

## 04_batchB.py (`results/batchB_results.txt`)

| Result | Expected |
|---|---|
| H3 clustered by task_name | b = −0.0611, SE = 0.0088, p = 3.8e-12 |
| H3 clustered by cluster_name_1 | b = −0.0611, SE = 0.0130, p = 2.6e-06 |
| H4 clustered by task_name | augmentation b = −0.0579, p = .051 |
| H5 indirect effect (point) | −0.10457 |
| H5 IID bootstrap 95% CI (seed 42) | [−0.206, −0.029] |
| H5 task-cluster bootstrap 95% CI (seed 42) | [−0.262, −0.008] |
| H3 + technical domain | aug b = −0.0386 (p < .001); tech b = 0.0203 (p < .001) |
| H3 + feedback-loop intensity | aug b = −0.0032 (p = .77, n.s.) |
| H3 full controls (n=488) | aug b = −0.0015 (p = .88, n.s.); R² = .299 |
