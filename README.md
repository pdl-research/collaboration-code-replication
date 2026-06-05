# Replication Materials: Lynch et al. Analysis of the Anthropic Economic Index

This repository contains code and instructions to reproduce the dataset construction and analyses reported in Lynch et al. All source data come from the publicly available [Anthropic Economic Index](https://huggingface.co/datasets/Anthropic/EconomicIndex) (CC-BY license, March 2025 release).

## Quick Start

There are three ways to run this code, listed from easiest to most manual. All three produce identical results.

### Option A: GitHub Codespaces (no local setup required)

This is the simplest option. It runs everything in the cloud using GitHub's free tier.

1. Go to this repository on GitHub.
2. Click the green **Code** button, then select the **Codespaces** tab.
3. Click **Create codespace on main**.
4. Wait 2-3 minutes for the environment to build. You will see a VS Code editor in your browser.
5. Open a terminal (Terminal menu > New Terminal) and run:
   ```
   python 01_build_dataset.py
   ```
6. Check the `data/` folder for outputs and the `data/build_log.txt` for verification.

### Option B: VS Code or Cursor with Dev Containers (local, but automated)

This runs the same container on your own machine. You need Docker installed.

1. **Install Docker Desktop**: Download from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/) and install it. Open Docker Desktop and let it finish starting up (you will see "Docker Desktop is running" in the system tray or menu bar).

2. **Install VS Code or Cursor**: If you do not already have one of these, download [VS Code](https://code.visualstudio.com/) or [Cursor](https://cursor.com/).

3. **Install the Dev Containers extension**: Open VS Code or Cursor. Press `Ctrl+Shift+X` (Windows/Linux) or `Cmd+Shift+X` (Mac) to open Extensions. Search for "Dev Containers" and install the one published by Microsoft.

4. **Open this project folder**: In VS Code/Cursor, go to File > Open Folder and select this project folder.

5. **Reopen in container**: You will see a notification in the bottom-right corner that says "Folder contains a Dev Container configuration file." Click **Reopen in Container**. If you miss the notification, press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac), type "Dev Containers: Reopen in Container", and select it. The first time, this will take 2-5 minutes to build the Docker image and install packages.

6. **Run the build script**: Open a terminal (Terminal > New Terminal) and run:
   ```
   python 01_build_dataset.py
   ```

7. **Check outputs**: Look in the `data/` folder for the three output CSV files and `build_log.txt`.

### Option C: Manual setup (no Docker)

If you prefer not to use Docker, you can set up the environment manually with Python and pip.

1. **Install Python 3.11**: Download from [python.org/downloads](https://www.python.org/downloads/). During installation on Windows, check the box that says "Add Python to PATH." To verify, open a terminal and run:
   ```
   python --version
   ```
   You should see `Python 3.11.x`.

2. **Create a virtual environment**: Open a terminal, navigate to this project folder, and run:
   ```
   python -m venv .venv
   ```

3. **Activate the virtual environment**:
   - On Mac/Linux: `source .venv/bin/activate`
   - On Windows: `.venv\Scripts\activate`

   You should see `(.venv)` at the start of your terminal prompt.

4. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

5. **Run the build script**:
   ```
   python 01_build_dataset.py
   ```

6. **Check outputs**: Look in the `data/` folder.

## What the Build Script Does

`01_build_dataset.py` performs the following steps, corresponding to the Dataset Construction section of the Supplementary Materials:

1. **Downloads** four source files from HuggingFace (task_pct_v2.csv, automation_vs_augmentation_by_task.csv, task_thinking_fractions.csv, and cluster_level_dataset.tsv). If the files already exist locally from a prior run, they are reused without contacting HuggingFace.

2. **Merges** the three task-level files on `task_name` to form a base of 3,365 O*NET tasks.

3. **Left-joins** cluster-level data, expanding to 3,612 rows (because 94 tasks map to multiple semantic clusters).

4. **Preserves** privacy-suppressed NaN values in cluster-level collaboration ratios. Anthropic's Clio system withholds ratios below approximately 0.5% to protect user privacy; these appear as NaN in the source data and are kept as-is (not replaced with zero).

5. **Computes** three composite indices at both task and cluster levels: automation index, augmentation index, and risk management index. When any component ratio is NaN, the composite index is also NaN, which naturally reduces the analytical sample size for models that use those indices.

6. **Extracts** two analytical samples: a task-level dataset (3,365 rows for H1 and H2) and a cluster-level dataset (593 rows for H3-H6).

7. **Writes** a verification log with row counts, expected counts, and SHA-256 checksums so you can confirm your output matches.

## Output Files

| File | Rows | Description |
|------|------|-------------|
| `data/integrated_dataset.csv` | 3,612 | Full multi-level dataset |
| `data/unified_dataset_v2_full.csv` | 3,612 | Same as above (backward-compatible name) |
| `data/task_level_dataset.csv` | 3,365 | Deduplicated task-level sample (H1, H2) |
| `data/cluster_level_dataset.csv` | 593 | Task-cluster combinations (H3-H6) |
| `data/build_log.txt` | — | Verification log with row counts, structural checks, and SHA-256 checksums |

## Verifying Your Results

After running the build script, open `data/build_log.txt`. You should see:

```
MATCH STATUS
  Integrated:    PASS
  Task-level:    PASS
  Cluster-level: PASS
```

The SHA-256 checksums in the log file should be identical across all machines running the same pinned package versions. If any row count shows "CHECK" instead of "PASS", the upstream data may have changed — contact the authors.

## Running the Analysis

After the build script completes successfully:

```
python 02_analysis.py
python 03_reviewer_models.py
python 04_batchB.py
```

`02_analysis.py` runs all six hypotheses (H1-H6) with HC3 robust standard errors, plus robustness checks (fractional logit, Cook's D trimming). Cluster-level sample sizes vary by hypothesis because OLS automatically drops rows with NaN in any model variable: H3 uses n=508, H4 uses n=488 (the additional exclusions come from NaN propagation through the two composite indices that H4 requires).

`03_reviewer_models.py` runs the model comparison suite for the highly skewed dependent variable — OLS, log-linear OLS (with Duan smearing for raw-scale RMSE), fractional logit on DV/100 (Papke & Wooldridge, 1996), GLM Gamma with log link, and Poisson QMLE — plus robustness checks excluding high-share tasks (share > 1.0).

`04_batchB.py` runs (1) cluster-robust standard errors grouped by task for all cluster-level models, addressing non-independence of the 593 cluster rows derived from 346 unique tasks; (2) bootstrapped percentile confidence intervals for the H5 indirect effect (5,000 replications, seed 42; both IID and task-cluster resampling); and (3) H3 sensitivity analyses with complexity/novelty proxies.

All results are saved to the `results/` directory. Compare your output against `EXPECTED_RESULTS.md`. See `CHANGELOG.md` for the v2.0 revision history.

## Project Structure

```
.
├── .devcontainer/
│   ├── devcontainer.json      # VS Code / Codespaces container config
│   └── Dockerfile             # Container image definition
├── data/                      # Created by 01_build_dataset.py
│   ├── raw/                   # Downloaded source files
│   ├── integrated_dataset.csv
│   ├── task_level_dataset.csv
│   ├── cluster_level_dataset.csv
│   └── build_log.txt
├── results/                   # Created by 02_analysis.py
│   ├── H1_results.txt ... H6_results.txt
│   ├── robustness_checks.txt
│   ├── batchA_reviewer_models.txt
│   ├── batchB_results.txt
│   └── analysis_log.txt
├── 01_build_dataset.py        # Data construction (run this first)
├── 02_analysis.py             # Hypothesis testing (H1-H6 + robustness)
├── 03_reviewer_models.py      # Model comparison suite + outlier robustness
├── 04_batchB.py               # Clustered SEs, bootstrap mediation, H3 controls
├── EXPECTED_RESULTS.md        # Key coefficients from a verified run
├── CHANGELOG.md               # Revision history
├── Lynch Supplementary Materials Data Sources and Dataset Construction.md
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Requirements

- Python 3.11
- Internet connection (to download data from HuggingFace on first run)
- Packages listed in `requirements.txt` (installed automatically via the dev container or manually via pip)

## License

Analysis code: MIT License. Source data: CC-BY (Anthropic, 2025).

## Citation

If you use this code, please cite both this repository and the Anthropic Economic Index:

> Handa, K., Tamkin, A., McCain, M., et al. (2025). Which economic tasks are performed with AI? Evidence from millions of Claude conversations. arXiv preprint arXiv:2503.04761.
