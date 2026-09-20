# Results Summary

The demonstration run completed successfully using the simulated expert inputs in `data/`.

## Main ranking from Fuzzy AHP

Top five global sub-barriers:

1. I1 — Complex workflow and fragmented responsibilities
2. I2 — Multi-stakeholder misalignment
3. F1 — High investment cost
4. T1 — Lack of skilled professionals
5. F3 — Split incentives

These outputs are saved in `outputs/subbarrier_priority.csv`.

## DEMATEL interpretation

The default simulated DEMATEL input classifies institutional and financial barriers as the main causal drivers, while social and technical barriers are mainly downstream effects. Full results are saved in `outputs/dematel_results.csv` and visualised in `outputs/dematel_cause_effect_map.png`.

## Replacing simulated data

Replace these files with real expert data:

- `data/sample_cluster_pairwise.csv`
- `data/sample_subbarrier_pairwise.csv`
- `data/sample_dematel_influence.csv`

Then re-run:

```bash
python run_pipeline.py
```
