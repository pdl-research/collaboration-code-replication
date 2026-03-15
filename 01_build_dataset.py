"""
01_build_dataset.py
====================
Data construction script for Lynch et al. analysis of the Anthropic Economic Index.

Downloads four source files from the March 2025 release of the Anthropic Economic
Index on HuggingFace, merges them into an integrated multi-level research dataset,
and computes composite indices as described in the Supplementary Materials.

Source: https://huggingface.co/datasets/Anthropic/EconomicIndex
License: CC-BY

Usage:
    python 01_build_dataset.py

Outputs (saved to data/ directory):
    - task_level_dataset.csv          : 3,365 unique O*NET tasks (for H1, H2)
    - cluster_level_dataset.csv       : 593 task-cluster combinations (for H3, H4, H5, H6)
    - integrated_dataset.csv          : Full 3,612-row multi-level dataset
    - build_log.txt                   : Verification log with row counts and checksums
"""

import os
import hashlib
import logging
from datetime import datetime, timezone

import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ID = "Anthropic/EconomicIndex"
RELEASE_DIR = "release_2025_03_27"
OUTPUT_DIR = "data"
RAW_DIR = os.path.join(OUTPUT_DIR, "raw")

# Source files (Table 1 in Supplementary Materials)
SOURCE_FILES = {
    "task_pct": f"{RELEASE_DIR}/task_pct_v2.csv",
    "interaction": f"{RELEASE_DIR}/automation_vs_augmentation_by_task.csv",
    "thinking": f"{RELEASE_DIR}/task_thinking_fractions.csv",
    "clusters": f"{RELEASE_DIR}/cluster_level_data/cluster_level_dataset.tsv",
}

# ---------------------------------------------------------------------------
# Logging setup
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

def sha256_of_file(filepath: str) -> str:
    """Compute SHA-256 hash of a file for reproducibility verification."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of_dataframe(df: pd.DataFrame) -> str:
    """Compute SHA-256 hash of a DataFrame's CSV representation."""
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


def normalize_task_name(s: pd.Series) -> pd.Series:
    """
    Normalize task_name for consistent merging across source files.

    The task-level files (task_pct, interaction, thinking) use lowercase
    task names, while the cluster-level file uses sentence case. This
    function lowercases and strips whitespace to ensure merge keys match.
    """
    return s.str.lower().str.strip()


# ---------------------------------------------------------------------------
# Step 1: Download source files from HuggingFace
# ---------------------------------------------------------------------------

def download_source_files() -> dict[str, str]:
    """Download all source files and return local paths."""
    os.makedirs(RAW_DIR, exist_ok=True)
    local_paths = {}

    for key, hf_path in SOURCE_FILES.items():
        logger.info(f"Downloading {hf_path}...")
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=hf_path,
            repo_type="dataset",
            local_dir=RAW_DIR,
        )
        local_paths[key] = local_path
        file_hash = sha256_of_file(local_path)
        logger.info(f"  -> {local_path} (SHA-256: {file_hash[:16]}...)")

    return local_paths


# ---------------------------------------------------------------------------
# Step 2: Load and validate source files
# ---------------------------------------------------------------------------

def load_source_files(paths: dict[str, str]) -> dict[str, pd.DataFrame]:
    """Load source CSVs/TSV into DataFrames with validation."""
    frames = {}

    # Task prevalence (Table 1, row 1): task_name, pct
    frames["task_pct"] = pd.read_csv(paths["task_pct"])
    frames["task_pct"]["task_name"] = normalize_task_name(frames["task_pct"]["task_name"])
    logger.info(
        f"task_pct_v2.csv: {len(frames['task_pct'])} rows, "
        f"columns: {list(frames['task_pct'].columns)}"
    )

    # Interaction patterns (Table 1, row 2): task_name, directive, feedback_loop, etc.
    frames["interaction"] = pd.read_csv(paths["interaction"])
    frames["interaction"]["task_name"] = normalize_task_name(frames["interaction"]["task_name"])
    logger.info(
        f"automation_vs_augmentation_by_task.csv: {len(frames['interaction'])} rows, "
        f"columns: {list(frames['interaction'].columns)}"
    )

    # Extended thinking fractions (Table 1, row 3): task_name, thinking_fraction
    frames["thinking"] = pd.read_csv(paths["thinking"])
    frames["thinking"]["task_name"] = normalize_task_name(frames["thinking"]["task_name"])
    logger.info(
        f"task_thinking_fractions.csv: {len(frames['thinking'])} rows, "
        f"columns: {list(frames['thinking'].columns)}"
    )

    # Cluster-level dataset (Table 1, row 4): TSV with onet_task as merge key
    frames["clusters"] = pd.read_csv(paths["clusters"], sep="\t")
    logger.info(
        f"cluster_level_dataset.tsv: {len(frames['clusters'])} rows, "
        f"columns: {list(frames['clusters'].columns)}"
    )

    return frames


# ---------------------------------------------------------------------------
# Step 3: Build task-level base (merge three task-level files)
# ---------------------------------------------------------------------------

def build_task_base(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge the three task-level files on task_name to form the base of 3,365
    O*NET tasks with observed Claude usage.

    Merge strategy:
    - Start with task_pct (3,365 tasks) as the base
    - Inner join with interaction patterns (should also be 3,365)
    - Left join with thinking fractions (available for subset of tasks)
    """
    # Check for task_name mismatches before merging
    pct_tasks = set(frames["task_pct"]["task_name"])
    int_tasks = set(frames["interaction"]["task_name"])
    only_in_pct = pct_tasks - int_tasks
    only_in_int = int_tasks - pct_tasks
    if only_in_pct:
        logger.warning(
            f"  {len(only_in_pct)} task(s) in task_pct but not interaction: "
            f"{list(only_in_pct)[:5]}"
        )
    if only_in_int:
        logger.warning(
            f"  {len(only_in_int)} task(s) in interaction but not task_pct: "
            f"{list(only_in_int)[:5]}"
        )

    # Merge task_pct with interaction patterns
    # Use outer join to preserve all tasks from both files, then log any gaps
    task_base = frames["task_pct"].merge(
        frames["interaction"],
        on="task_name",
        how="outer",
        indicator=True,
    )
    merge_counts = task_base["_merge"].value_counts()
    logger.info(f"task_pct + interaction merge breakdown: {merge_counts.to_dict()}")
    # Keep all rows (outer) so we don't silently drop tasks
    task_base = task_base.drop(columns=["_merge"])
    logger.info(
        f"After merging task_pct + interaction: {len(task_base)} rows"
    )

    # Left join with thinking fractions
    task_base = task_base.merge(
        frames["thinking"],
        on="task_name",
        how="left",
    )
    thinking_available = task_base["thinking_fraction"].notna().sum()
    logger.info(
        f"After merging with thinking: {len(task_base)} rows "
        f"({thinking_available} with thinking data, "
        f"{thinking_available / len(task_base) * 100:.1f}%)"
    )

    # Compute composite indices (Supplementary Materials)
    task_base["automation_index"] = (
        task_base["directive"] + task_base["feedback_loop"]
    )
    task_base["augmentation_index"] = (
        task_base["task_iteration"] + task_base["learning"] + task_base["validation"]
    )
    task_base["risk_management_index"] = (
        task_base["feedback_loop"] + task_base["validation"]
    )

    # Rename pct -> task_conversation_share for manuscript consistency
    task_base = task_base.rename(columns={"pct": "task_conversation_share"})

    logger.info(
        f"Task-level base complete: {len(task_base)} rows, "
        f"{len(task_base.columns)} columns"
    )

    return task_base


# ---------------------------------------------------------------------------
# Step 4: Prepare cluster-level data and merge
# ---------------------------------------------------------------------------

def prepare_cluster_data(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Prepare cluster-level data for joining to the task base.

    The cluster dataset uses 'onet_task' as the task description column,
    which corresponds to 'task_name' in the task-level files.

    Cluster-level collaboration columns use the prefix 'collaboration:'.
    We rename them for consistency and compute cluster-level composite indices.
    """
    clusters = frames["clusters"].copy()

    # Rename onet_task -> task_name for merge key consistency
    # Note: cluster file uses sentence case (e.g., "Compile terminology...")
    # while task-level files use lowercase. Normalize to match.
    clusters = clusters.rename(columns={"onet_task": "task_name"})
    clusters["task_name"] = normalize_task_name(clusters["task_name"])

    # Rename collaboration columns to cluster-level names
    collab_renames = {
        "collaboration:directive_ratio": "cluster_directive",
        "collaboration:feedback loop_ratio": "cluster_feedback_loop",
        "collaboration:task iteration_ratio": "cluster_task_iteration",
        "collaboration:learning_ratio": "cluster_learning",
        "collaboration:validation_ratio": "cluster_validation",
        "collaboration:none_ratio": "cluster_none",
        "has_thinking_ratio": "cluster_thinking_fraction",
    }
    clusters = clusters.rename(columns=collab_renames)

    # Convert cluster numeric columns to float (some cells are empty strings
    # in the TSV, which pandas reads as NaN — fill with 0 for ratio columns)
    cluster_ratio_cols = [
        "cluster_directive", "cluster_feedback_loop", "cluster_task_iteration",
        "cluster_learning", "cluster_validation", "cluster_none",
    ]
    for col in cluster_ratio_cols:
        if col in clusters.columns:
            clusters[col] = pd.to_numeric(clusters[col], errors="coerce").fillna(0.0)
    # cluster_thinking_fraction: keep NaN where not available (don't fill with 0)
    if "cluster_thinking_fraction" in clusters.columns:
        clusters["cluster_thinking_fraction"] = pd.to_numeric(
            clusters["cluster_thinking_fraction"], errors="coerce"
        )

    # Compute cluster-level composite indices
    clusters["cluster_automation_index"] = (
        clusters["cluster_directive"] + clusters["cluster_feedback_loop"]
    )
    clusters["cluster_augmentation_index"] = (
        clusters["cluster_task_iteration"]
        + clusters["cluster_learning"]
        + clusters["cluster_validation"]
    )
    clusters["cluster_risk_management_index"] = (
        clusters["cluster_feedback_loop"] + clusters["cluster_validation"]
    )

    unique_tasks_in_clusters = clusters["task_name"].nunique()
    logger.info(
        f"Cluster data prepared: {len(clusters)} task-cluster mappings "
        f"from {unique_tasks_in_clusters} unique tasks"
    )

    return clusters


def build_integrated_dataset(
    task_base: pd.DataFrame, clusters: pd.DataFrame
) -> pd.DataFrame:
    """
    Join cluster-level data to the task base using a left join on task_name.

    Per Supplementary Materials:
    - 3,365 tasks in base; 346 appear in cluster taxonomy
    - 94 tasks map to multiple clusters -> 593 task-cluster mappings
    - Final dataset: 3,019 rows without clusters + 593 with = 3,612 rows
    """
    integrated = task_base.merge(
        clusters,
        on="task_name",
        how="left",
    )

    # Diagnostics
    has_cluster = integrated["cluster_name_0"].notna()
    tasks_with_clusters = integrated.loc[has_cluster, "task_name"].nunique()
    tasks_without_clusters = integrated.loc[~has_cluster, "task_name"].nunique()
    multi_cluster_tasks = (
        integrated.loc[has_cluster]
        .groupby("task_name")
        .size()
        .pipe(lambda s: (s > 1).sum())
    )

    logger.info(f"Integrated dataset: {len(integrated)} total rows")
    logger.info(f"  Rows with cluster assignment: {has_cluster.sum()}")
    logger.info(f"  Rows without cluster assignment: {(~has_cluster).sum()}")
    logger.info(f"  Unique tasks with clusters: {tasks_with_clusters}")
    logger.info(f"  Unique tasks without clusters: {tasks_without_clusters}")
    logger.info(f"  Tasks mapping to multiple clusters: {multi_cluster_tasks}")

    return integrated


# ---------------------------------------------------------------------------
# Step 5: Extract analytical samples
# ---------------------------------------------------------------------------

def extract_analytical_samples(
    integrated: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract the two primary analytical samples from the integrated dataset.

    Task-level sample (H1, H2): 3,365 unique tasks, deduplicated from
    any multi-cluster rows (task-level values are identical across
    cluster assignments).

    Cluster-level sample (H3, H4, H5, H6): 593 task-cluster combinations
    where both task and cluster data are available.
    """
    # Task-level: deduplicate to one row per unique task
    task_level = integrated.drop_duplicates(subset=["task_name"]).copy()

    # Drop cluster-specific columns for the task-level sample
    cluster_cols = [
        c for c in task_level.columns
        if c.startswith("cluster_") or c in [
            "percent_records", "percent_users",
            "cluster_description_0", "cluster_description_1",
            "cluster_description_2",
        ]
    ]
    task_level = task_level.drop(columns=cluster_cols, errors="ignore")

    logger.info(f"Task-level sample: {len(task_level)} unique tasks")

    # Cluster-level: rows where cluster assignment exists
    cluster_level = integrated[integrated["cluster_name_0"].notna()].copy()
    logger.info(f"Cluster-level sample: {len(cluster_level)} task-cluster rows")

    return task_level, cluster_level


# ---------------------------------------------------------------------------
# Step 6: Save outputs and verification log
# ---------------------------------------------------------------------------

def save_outputs(
    integrated: pd.DataFrame,
    task_level: pd.DataFrame,
    cluster_level: pd.DataFrame,
) -> None:
    """Save all datasets and a verification log."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save datasets
    integrated.to_csv(
        os.path.join(OUTPUT_DIR, "integrated_dataset.csv"), index=False
    )
    task_level.to_csv(
        os.path.join(OUTPUT_DIR, "task_level_dataset.csv"), index=False
    )
    cluster_level.to_csv(
        os.path.join(OUTPUT_DIR, "cluster_level_dataset.csv"), index=False
    )
    # Also save as unified_dataset_v2_full.csv for backward compatibility
    integrated.to_csv(
        os.path.join(OUTPUT_DIR, "unified_dataset_v2_full.csv"), index=False
    )

    logger.info(f"Datasets saved to {OUTPUT_DIR}/")

    # Build verification log
    log_lines = [
        "=" * 70,
        "BUILD VERIFICATION LOG",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "=" * 70,
        "",
        "ROW COUNTS",
        f"  Integrated dataset:   {len(integrated):>6} rows",
        f"  Task-level sample:    {len(task_level):>6} rows",
        f"  Cluster-level sample: {len(cluster_level):>6} rows",
        "",
        "EXPECTED COUNTS (from Supplementary Materials)",
        f"  Integrated dataset:   {3612:>6} rows",
        f"  Task-level sample:    {3365:>6} rows",
        f"  Cluster-level sample: {593:>6} rows",
        "",
        "COLUMN COUNTS",
        f"  Integrated dataset:   {len(integrated.columns):>6} columns",
        f"  Task-level sample:    {len(task_level.columns):>6} columns",
        f"  Cluster-level sample: {len(cluster_level.columns):>6} columns",
        "",
        "STRUCTURAL VERIFICATION",
        f"  Unique task_name values:       {integrated['task_name'].nunique():>6}  (expected: 3,365)",
        f"  Rows with cluster assignment:  {integrated['cluster_name_0'].notna().sum():>6}  (expected: 593)",
        f"  Rows without cluster:          {integrated['cluster_name_0'].isna().sum():>6}  (expected: 3,019)",
        f"  Unique tasks with clusters:    {integrated.loc[integrated['cluster_name_0'].notna(), 'task_name'].nunique():>6}  (expected: 346)",
        f"  Multi-cluster tasks:           {integrated.loc[integrated['cluster_name_0'].notna()].groupby('task_name').size().pipe(lambda s: (s > 1).sum()):>6}  (expected: 94)",
        "",
        "MATCH STATUS",
        f"  Integrated rows:     {'PASS' if len(integrated) == 3612 else 'CHECK - expected 3612'}",
        f"  Task-level rows:     {'PASS' if len(task_level) == 3365 else 'CHECK - expected 3365'}",
        f"  Cluster-level rows:  {'PASS' if len(cluster_level) == 593 else 'CHECK - expected 593'}",
        f"  Unique tasks:        {'PASS' if integrated['task_name'].nunique() == 3365 else 'CHECK - expected 3365'}",
        f"  Tasks with clusters: {'PASS' if integrated.loc[integrated['cluster_name_0'].notna(), 'task_name'].nunique() == 346 else 'CHECK - expected 346'}",
        f"  Multi-cluster tasks: {'PASS' if integrated.loc[integrated['cluster_name_0'].notna()].groupby('task_name').size().pipe(lambda s: (s > 1).sum()) == 94 else 'CHECK - expected 94'}",
        "",
        "SHA-256 CHECKSUMS (for cross-machine verification)",
        f"  integrated_dataset.csv:   {sha256_of_dataframe(integrated)}",
        f"  task_level_dataset.csv:   {sha256_of_dataframe(task_level)}",
        f"  cluster_level_dataset.csv:{sha256_of_dataframe(cluster_level)}",
        "",
        "DESCRIPTIVE SUMMARY",
        f"  task_conversation_share: "
        f"mean={task_level['task_conversation_share'].mean():.6f}, "
        f"std={task_level['task_conversation_share'].std():.6f}, "
        f"min={task_level['task_conversation_share'].min():.6f}, "
        f"max={task_level['task_conversation_share'].max():.6f}",
        f"  augmentation_index (task): "
        f"mean={task_level['augmentation_index'].mean():.6f}, "
        f"std={task_level['augmentation_index'].std():.6f}",
        f"  automation_index (task): "
        f"mean={task_level['automation_index'].mean():.6f}, "
        f"std={task_level['automation_index'].std():.6f}",
        f"  Tasks with thinking data: "
        f"{task_level['thinking_fraction'].notna().sum()} of {len(task_level)}",
        "",
        "=" * 70,
    ]

    log_path = os.path.join(OUTPUT_DIR, "build_log.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))

    logger.info(f"Verification log saved to {log_path}")

    # Print the log to stdout as well
    print("\n".join(log_lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("Starting dataset construction...")
    logger.info(f"Repository: {REPO_ID}")
    logger.info(f"Release: {RELEASE_DIR}")

    # Step 1: Download
    paths = download_source_files()

    # Step 2: Load
    frames = load_source_files(paths)

    # Step 3: Build task-level base
    task_base = build_task_base(frames)

    # Step 4: Prepare clusters and build integrated dataset
    clusters = prepare_cluster_data(frames)
    integrated = build_integrated_dataset(task_base, clusters)

    # Step 5: Extract analytical samples
    task_level, cluster_level = extract_analytical_samples(integrated)

    # Step 6: Save and verify
    save_outputs(integrated, task_level, cluster_level)

    logger.info("Dataset construction complete.")


if __name__ == "__main__":
    main()
