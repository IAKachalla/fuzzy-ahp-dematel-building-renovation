# Fuzzy AHP-DEMATEL Code for Sustainable Building Renovation Barriers

This project implements the methodology described in the paper:

**Systemic Barriers to Sustainable Building Renovation: A Life Cycle-Based Fuzzy AHP-DEMATEL Approach**

The code supports:

- Fuzzy AHP priority weighting of barrier clusters and sub-barriers
- Consistency-ratio checking of pairwise comparison matrices
- Fuzzy DEMATEL causal mapping using expert influence matrices, with arithmetic aggregation of expert TFNs
- Sensitivity analysis using Monte Carlo perturbation of expert judgments
- Policy scenario modelling using DEMATEL influence pathways
- Export of result tables and publication-ready figures

The included data are **simulated demonstration inputs** aligned with the paper structure. Replace the CSV files in `data/` with real expert questionnaire responses when available.

## Folder structure

```text
fuzzy_ahp_dematel_renovation_code/
├── data/
│   ├── barriers.csv
│   ├── sample_cluster_pairwise.csv
│   ├── sample_subbarrier_pairwise.csv
│   └── sample_dematel_influence.csv
├── fuzzy_mcdm/
│   ├── __init__.py
│   ├── tfn.py
│   ├── scales.py
│   ├── fuzzy_ahp.py
│   ├── fuzzy_dematel.py
│   ├── sensitivity.py
│   ├── scenario_modeling.py
│   └── visualization.py
├── outputs/
├── requirements.txt
└── run_pipeline.py
```

## Installation

```bash
cd fuzzy_ahp_dematel_renovation_code
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the full analysis

```bash
python run_pipeline.py
```

This creates CSV result tables and PNG figures in `outputs/`.

## Main outputs

- `cluster_priority.csv` — Fuzzy AHP weights and ranks for Financial, Institutional, Technical, and Social barrier clusters.
- `subbarrier_priority.csv` — Local and global priority ranking of sub-barriers.
- `dematel_results.csv` — DEMATEL prominence (`D+R`), relation (`D-R`), and cause/effect classification.
- `dematel_total_relation_matrix.csv` — Total relation matrix from DEMATEL.
- `sensitivity_rank_stability.csv` — Monte Carlo rank stability of sub-barriers.
- `policy_scenarios.csv` — Estimated leverage effects of selected intervention scenarios.
- `cluster_weights.png`, `subbarrier_global_ranking.png`, `dematel_cause_effect_map.png`, `policy_scenario_impacts.png`.

## How to enter real expert data

### Fuzzy AHP input format

Use the following long-format CSV columns:

```text
expert,context,item_i,item_j,l,m,u
```

Example:

```text
E1,clusters,F,I,1.06,1.17,1.28
```

Only the upper triangle is required. The code automatically adds diagonal values and reciprocal values.

### DEMATEL input format

Use the following CSV columns:

```text
expert,source,target,l,m,u
```

Example:

```text
E1,I1,T1,0.35,0.40,0.45
```

Diagonal values are automatically treated as zero influence.

## Notes for the paper

The sample data are deliberately transparent. They are meant for methodological demonstration and article figure generation. For journal submission, replace them with empirical expert inputs and report:

1. number of experts,
2. expert screening criteria,
3. consistency ratios,
4. fuzzy scales used,
5. sensitivity analysis settings,
6. causal threshold used for DEMATEL network interpretation.


## Interactive execution with MyBinder

For journal review, launch `Journal_Reproducibility.ipynb` through MyBinder. The notebook runs the complete analysis, executes the verification checks, displays key result tables, and renders the generated figures.

See `BINDER_INSTRUCTIONS.md` for the exact setup procedure.
