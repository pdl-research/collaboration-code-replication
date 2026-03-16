# Data Sources and Dataset Construction

We analyzed publicly available data from the Anthropic Economic Index (Anthropic, 2025b), hosted on HuggingFace ([https://huggingface.co/datasets/Anthropic/EconomicIndex](https://huggingface.co/datasets/Anthropic/EconomicIndex)). The source data derive from approximately 1 million anonymized conversations between Claude.ai users (Free and Pro tiers) and Claude 3.7 Sonnet, collected between December 2024 and March 2025\. Conversations were classified using Clio, a privacy-preserving methodology that maps natural language exchanges to standardized occupational tasks (Tamkin et al., 2024). In validation tests, Clio achieved 90.7% agreement with expert human raters on a sample of 150 conversations (Handa et al., 2025). Each conversation was assigned to one of approximately 19,500 tasks from the U.S. Department of Labor's O*NET occupational taxonomy (National Center for O*NET Development, 2025\) and classified into one of five mutually exclusive interaction patterns: directive (complete task delegation), feedback loop (iteration driven by system errors or environment feedback), task iteration (collaborative human-AI refinement), learning (knowledge acquisition and explanation-seeking), and validation (verification of human-completed work). Following Handa et al. (2025), the first two patterns characterize automation-oriented use (43% of conversations), while the latter three characterize augmentation-oriented use (57% of conversations). In addition, the March 2025 release introduced measurements of extended thinking mode, a Claude 3.7 Sonnet feature that enables deeper cognitive processing when explicitly activated by the user.

The March 2025 release also included a bottom-up taxonomy of semantic usage clusters derived from unsupervised learning on conversation embeddings. Unlike the O\*NET mapping, which imposes a top-down occupational structure, these clusters represent emergent patterns of actual use—naturally occurring categories of tasks that users bring to the system. Clusters are organized hierarchically across three levels of granularity: 593 primary clusters (e.g., "Solve geometry and trigonometry problems"), 143 intermediate clusters (e.g., "Solve advanced mathematical problems with detailed explanations"), and 30 broad clusters (e.g., "Solve scientific and mathematical problems with explanations"). Each cluster includes aggregated interaction ratios, extended thinking usage rates, and adoption metrics indicating the share of users and conversations within that cluster.

### Dataset Construction

We constructed a multi-level research dataset by integrating four source files from the March 2025 release, using the normalized task description (`task_name`) as the common merge key (see Table 1 for source file specifications). Three task-level files were joined to form a base of 3,365 O*NET tasks with observed Claude usage: task prevalence data (`task_pct_v2.csv`), interaction pattern ratios (`automation_vs_augmentation_by_task.csv`), and extended thinking fractions (`task_thinking_fractions.csv`). Extended thinking data were available for 620 of these tasks (18.4%), reflecting the proportion of O*NET tasks for which users activated this mode during the observation period.

We then joined cluster-level data from the `cluster_level_dataset` directory using a left join on `task_name`, preserving all task-level rows. Of the 3,365 tasks, 346 appeared in Anthropic's cluster taxonomy. Because the clustering methodology permits a single O\*NET task to map to multiple semantic clusters—for example, "administer, proctor, or score academic or diagnostic assessments" maps to both "Help with course enrollment, attendance, and exam administration" and "Create or implement academic assessment and scoring systems"—94 tasks were assigned to more than one cluster, yielding 593 task-cluster mappings. This expanded the dataset from 3,365 to 3,612 rows: 3,019 rows representing tasks without cluster assignments (one row per task) and 593 rows representing task-cluster combinations.

We computed three composite indices at both the task and cluster levels: an automation index (directive \+ feedback loop), an augmentation index (task iteration \+ learning \+ validation), and a risk management index (feedback loop \+ validation). The risk management index operationalizes Protection Motivation Theory as risk-aware interaction during AI use rather than pre-adoption threat avoidance, consistent with recent reframings of PMT for generative AI contexts (Shrivastava, 2025). Complete variable definitions and coding specifications are provided in the data dictionary (Supplementary Material).

### Analytical Samples

The multi-level dataset structure yields distinct analytical samples for each hypothesis family. Task-level hypotheses (H1 and H2), which test Technology Acceptance Model and Protection Motivation Theory predictions about adoption and risk-aware engagement, use the 3,365 unique O\*NET tasks as the unit of analysis. For these analyses, the 94 tasks assigned to multiple clusters were deduplicated to their task-level values, which are identical across cluster assignments. Cluster-level hypotheses (H3, H4, and H6), which test Social Exchange Theory and Socio-Technical Systems Theory predictions about trust emergence and system-level adoption, use the 593 task-cluster combinations as the unit of analysis. The cross-level mediation hypothesis (H5) examines how task-level collaborative interaction relates to adoption through cluster-level extended thinking, drawing on the 593 rows where both task and cluster data are available. Table 2 summarizes the analytical sample, dependent variable, and independent variables for each hypothesis.

### TABLE 1 

**Table 1\.** Source data files from the Anthropic Economic Index March 2025 release and their contribution to the integrated research dataset.

| Source File | Contents | Key Variables | Records | Role in Dataset |
| :---- | :---- | :---- | :---- | :---- |
| `task_pct_v2.csv` | Task prevalence | `task_name`, `pct` | 3,365 | Base file; adoption proxy |
| `automation_vs_augmentation_by_task.csv` | Interaction pattern ratios | `directive`, `feedback_loop`, `task_iteration`, `learning`, `validation`, `filtered` | 3,365 | Interaction patterns |
| `task_thinking_fractions.csv` | Extended thinking usage | `thinking_fraction` | 620 | Trust indicator (task level) |
| `cluster_level_dataset/` | Semantic cluster assignments, cluster-aggregated interaction ratios, adoption metrics | `cluster_name_0/1/2`, `cluster_thinking_fraction`, `percent_users`, `percent_records` | 593 mappings (346 unique tasks) | Cluster-level variables |

Note: All files merged on `task_name`. Source data available under CC-BY license at [https://huggingface.co/datasets/Anthropic/EconomicIndex](https://huggingface.co/datasets/Anthropic/EconomicIndex).

### TABLE 2

**Table 2\.** Hypothesis-to-data mapping showing analytical level, sample, and variable operationalization for each hypothesis.

| Hypothesis | Theory | Level | n | DV | IV(s) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| H1 | TAM | Task | 3,365 | `pct` | `task_iteration`, `learning`, `augmentation_index` |
| H2 | PMT (reframed) | Task | 3,365 | `pct` | `risk_management_index` |
| H3 | SET | Cluster | 593 | `cluster_thinking_fraction` | `cluster_augmentation_index` |
| H4 | STS | Cluster | 593 | `cluster_thinking_fraction` | `cluster_automation_index`, `cluster_augmentation_index` |
| H5 | TAM × SET | Cross-level | 593 | `pct` | `augmentation_index` → `cluster_thinking_fraction` |
| H6 | SET × STS | Cluster | 593 | `percent_users` | `cluster_augmentation_index`, `cluster_thinking_fraction` |

Note: TAM \= Technology Acceptance Model; PMT \= Protection Motivation Theory; SET \= Social Exchange Theory; STS \= Socio-Technical Systems Theory. DV \= dependent variable; IV \= independent variable. Variable pct was named task\_conversation\_share for this manuscript.

### REFERENCES 

Anthropic. (2025b). Anthropic Economic Index \[Data set\]. Hugging Face. [https://huggingface.co/datasets/Anthropic/EconomicIndex](https://huggingface.co/datasets/Anthropic/EconomicIndex)

Handa, K., Carlin, B., Dragan, A., Ganguli, D., Grosse, R., Goodman, N., et al. (2025). Which economic tasks are performed with AI? Evidence from millions of Claude conversations. Anthropic Research. [https://assets.anthropic.com/m/2e23255f1e84ca97/original/Economic\_Tasks\_AI\_Paper.pdf](https://assets.anthropic.com/m/2e23255f1e84ca97/original/Economic_Tasks_AI_Paper.pdf)

National Center for O*NET Development. (2025). O*NET OnLine. [https://www.onetonline.org/](https://www.onetonline.org/)

Shrivastava, P. (2025). Understanding acceptance and resistance toward generative AI technologies: A multi-theoretical framework integrating functional, risk, and sociolegal factors. Front. Artif. Intell. 8:1565927. doi: 10.3389/frai.2025.1565927

Tamkin, A., Askell, A., Lovitt, L., Durmus, E., Joseph, N., Kravec, S., et al. (2024). CLIO: Privacy-preserving insights into real-world AI use. arXiv preprint arXiv:2410.13265.  
