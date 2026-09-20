# Verification Report

Date checked: 2026-06-27

## Status

The project was inspected and run from the command line using:

```bash
python run_pipeline.py --sensitivity-iterations 100
python -m py_compile run_pipeline.py fuzzy_mcdm/*.py
python verify_pipeline.py
```

All checks passed.

## What was verified

- The full pipeline runs without runtime errors.
- Python files compile successfully.
- Cluster AHP weights sum to 1.
- Sub-barrier local AHP weights sum to 1 within each cluster.
- AHP consistency ratios are below 0.10 for the included simulated data.
- DEMATEL outputs contain finite D, R, D+R and D-R values.
- Expected CSV outputs and PNG figures are produced.

## Correction made during verification

The DEMATEL module was updated to aggregate expert triangular fuzzy direct-influence judgments using arithmetic averaging rather than geometric averaging. Arithmetic averaging is the safer default for fuzzy DEMATEL direct-influence matrices.

## Remaining caveat

The code is computationally correct for the implemented workflow and the simulated demonstration data. The paper-level validity still depends on replacing the simulated CSV inputs with real expert questionnaire data and reporting the expert panel, scales, consistency ratios, and sensitivity analysis.
