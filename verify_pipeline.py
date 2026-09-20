"""Lightweight verification checks for the Fuzzy AHP-DEMATEL project.

Run after `python run_pipeline.py`:
    python verify_pipeline.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("outputs")
REQUIRED = [
    "cluster_priority.csv",
    "subbarrier_priority.csv",
    "cluster_expert_consistency.csv",
    "subbarrier_expert_consistency.csv",
    "dematel_results.csv",
    "dematel_total_relation_matrix.csv",
    "sensitivity_rank_stability.csv",
    "policy_scenarios.csv",
]


def assert_close(value: float, target: float, tol: float = 1e-6, message: str = "") -> None:
    if abs(value - target) > tol:
        raise AssertionError(message or f"Expected {target}, got {value}")


def main() -> None:
    missing = [f for f in REQUIRED if not (OUT / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing expected output files: {missing}. Run `python run_pipeline.py` first.")

    clusters = pd.read_csv(OUT / "cluster_priority.csv")
    sub = pd.read_csv(OUT / "subbarrier_priority.csv")
    c_cr = pd.read_csv(OUT / "cluster_expert_consistency.csv")
    s_cr = pd.read_csv(OUT / "subbarrier_expert_consistency.csv")
    dematel = pd.read_csv(OUT / "dematel_results.csv")
    total_relation = pd.read_csv(OUT / "dematel_total_relation_matrix.csv", index_col=0)

    assert_close(float(clusters["weight"].sum()), 1.0, message="Cluster weights do not sum to 1.")
    for cluster_code, group in sub.groupby("cluster_code"):
        assert_close(float(group["local_weight"].sum()), 1.0, tol=1e-5, message=f"Local weights for {cluster_code} do not sum to 1.")

    if (c_cr["consistency_ratio"] >= 0.10).any():
        raise AssertionError("At least one cluster-level AHP consistency ratio is >= 0.10.")
    if (s_cr["consistency_ratio"] >= 0.10).any():
        raise AssertionError("At least one sub-barrier AHP consistency ratio is >= 0.10.")

    expected_cols = {"D", "R", "prominence_D_plus_R", "relation_D_minus_R", "type"}
    if not expected_cols.issubset(dematel.columns):
        raise AssertionError(f"DEMATEL output is missing columns: {sorted(expected_cols.difference(dematel.columns))}")
    if not np.isfinite(dematel[["D", "R", "prominence_D_plus_R", "relation_D_minus_R"]].to_numpy()).all():
        raise AssertionError("DEMATEL output contains non-finite values.")
    if total_relation.shape[0] != total_relation.shape[1]:
        raise AssertionError("DEMATEL total relation matrix is not square.")

    print("Verification passed: outputs exist, weights normalize correctly, CR < 0.10, and DEMATEL values are finite.")


if __name__ == "__main__":
    main()
