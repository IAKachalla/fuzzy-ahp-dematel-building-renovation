"""Simple policy scenario modelling based on DEMATEL total relation values."""

from __future__ import annotations

import numpy as np
import pandas as pd


def intervention_effect(total_relation_matrix: np.ndarray, items: list[str], interventions: dict[str, float]) -> dict[str, float]:
    """Estimate system-wide leverage of targeted interventions.

    Parameters
    ----------
    total_relation_matrix:
        DEMATEL total relation matrix. Rows are influencing barriers and columns are influenced barriers.
    items:
        Barrier item codes in the same order as the matrix.
    interventions:
        Dictionary of barrier code -> assumed intervention strength from 0 to 1.

    Returns
    -------
    Dictionary with affected barrier reductions and total leverage score.
    """
    pos = {item: i for i, item in enumerate(items)}
    impact_vector = np.zeros(len(items), dtype=float)

    for barrier, strength in interventions.items():
        if barrier not in pos:
            raise KeyError(f"Unknown intervention barrier: {barrier}")
        if not 0 <= strength <= 1:
            raise ValueError("Intervention strength must be between 0 and 1.")
        impact_vector += strength * total_relation_matrix[pos[barrier], :]

    if impact_vector.max() > 0:
        normalized = impact_vector / impact_vector.max()
    else:
        normalized = impact_vector

    result = {item: float(normalized[i]) for i, item in enumerate(items)}
    result["total_leverage_score"] = float(impact_vector.sum())
    return result


def evaluate_policy_scenarios(total_relation_matrix: np.ndarray, items: list[str]) -> pd.DataFrame:
    scenarios = {
        "S1_governance_reform": {"I1": 0.35, "I2": 0.25},
        "S2_financing_mechanisms": {"F1": 0.35, "F3": 0.25},
        "S3_data_infrastructure": {"I3": 0.35},
        "S4_integrated_package": {"I1": 0.30, "I2": 0.25, "F1": 0.25, "F3": 0.20, "I3": 0.15},
    }
    rows = []
    for name, interventions in scenarios.items():
        out = intervention_effect(total_relation_matrix, items, interventions)
        row = {"scenario": name, "interventions": "; ".join(f"{k}:{v:.2f}" for k, v in interventions.items())}
        row.update(out)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("total_leverage_score", ascending=False)
