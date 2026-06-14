# Reproduction Verification (2026-06-13)

This bundle was run end-to-end, offline (no network), from the public Anthropic
Economic Index source files. `01_build_dataset.py` -> `02_analysis.py` ->
`03_reviewer_models.py` -> `04_batchB.py`. All sample sizes and headline
coefficients reproduced and tie to the manuscript and to `EXPECTED_RESULTS.md`.

## Build (data/build_log.txt) — PASS
| Check | Value |
|---|---|
| Integrated rows | 3,612 |
| Task-level rows | 3,365 |
| Cluster-level rows | 593 |
| Unique tasks | 3,365 |
| Tasks with clusters | 346 |
| Multi-cluster tasks | 94 |
| task_conversation_share | mean 0.0297, max 6.65 |

SHA-256 (verified run):
- integrated_dataset.csv: 2096bed5190b3363aaa9c9142804af4f44970ef6f4d4dd7ec118e462d50a16b4
- task_level_dataset.csv: 0c53fc1d41c48e18c8e5ad3166d2426f0f7e63f94d2b8a2152a7b9e19077dc85
- cluster_level_dataset.csv: 48f11d14d907ef7e840855b5b0f02587556808c9b92a99a00417866faf37d367

(Checksums depend on pinned package versions per requirements.txt.)

## Analytical sample sizes — reproduced, tied to manuscript
| Hypothesis | n | Source DV |
|---|---|---|
| H1 / H2 (task, complete-case) | 3,364 | task_conversation_share |
| H3 (cluster) | 508 | cluster_thinking_fraction |
| H4 (cluster) | 488 | cluster_thinking_fraction |
| H5 (cross-level mediation) | 568 obs / 328 unique tasks | task_conversation_share |
| H6 (cluster) | 508 | cluster_user_share |

Constructed cluster dataset = 593 task-cluster mappings (346 unique tasks, 94 multi-cluster).
Reduction from 593 to per-model n is listwise deletion of privacy-suppressed (NaN) values.

## Headline coefficients (match EXPECTED_RESULTS.md)
H1 log-linear b=2.8144 (R^2=.399); H1 augmentation b=0.0596; H2a b=0.6649;
H3 b=-0.0611 (R^2=.107); H4 automation 0.0016 / augmentation -0.0579;
H5 a=-0.0169, b=6.194, c'=-0.903, indirect=-0.10457, IID bootstrap 95% CI [-0.206,-0.029],
task-cluster CI [-0.262,-0.008]; H6 Model 1 augmentation b=-0.0274 (R^2=.013).

## Cross-artifact consistency
README.md, EXPECTED_RESULTS.md, CHANGELOG.md, the supplementary materials
(.docx and .md), and the manuscript all state the same sample sizes. No artifact
required a sample-size edit. The deprecated pre-deduplication "62 high-share tasks"
figure appears only as a labeled historical note in CHANGELOG.md; the live value is
7 high-share tasks in the complete-case sample.
