"""Sensitivity analysis tools for Fuzzy AHP results."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .fuzzy_ahp import compute_fuzzy_ahp


def perturb_fuzzy_matrix(matrix: np.ndarray, rng: np.random.Generator, pct: float = 0.10) -> np.ndarray:
    """Apply bounded random perturbation to a reciprocal TFN pairwise matrix."""
    mat = matrix.copy().astype(float)
    n = mat.shape[0]
    for i in range(n):
        mat[i, i, :] = (1.0, 1.0, 1.0)
        for j in range(i + 1, n):
            factor = rng.uniform(1 - pct, 1 + pct)
            perturbed = mat[i, j, :] * factor
            perturbed = np.sort(perturbed)
            mat[i, j, :] = perturbed
            mat[j, i, :] = (1 / perturbed[2], 1 / perturbed[1], 1 / perturbed[0])
    return mat


def rank_stability(
    matrices: dict[str, np.ndarray],
    items: Iterable[str],
    *,
    iterations: int = 500,
    pct: float = 0.10,
    seed: int = 42,
) -> pd.DataFrame:
    """Monte Carlo rank stability for an AHP comparison problem."""
    rng = np.random.default_rng(seed)
    item_list = list(items)
    records = []
    expert_names = list(matrices)

    for k in range(iterations):
        perturbed = {
            name: perturb_fuzzy_matrix(matrices[name], rng, pct=pct)
            for name in expert_names
        }
        res = compute_fuzzy_ahp(perturbed, item_list)
        ranks = pd.Series(res.crisp_weights, index=item_list).rank(ascending=False, method="min").astype(int)
        for item in item_list:
            records.append({"iteration": k + 1, "item": item, "rank": int(ranks[item]), "weight": float(res.crisp_weights[item_list.index(item)])})

    raw = pd.DataFrame(records)
    summary = raw.groupby("item").agg(
        mean_rank=("rank", "mean"),
        median_rank=("rank", "median"),
        best_rank=("rank", "min"),
        worst_rank=("rank", "max"),
        mean_weight=("weight", "mean"),
        std_weight=("weight", "std"),
    ).reset_index()
    summary["stable_top_3_probability"] = raw.assign(top3=raw["rank"] <= 3).groupby("item")["top3"].mean().reindex(summary["item"]).values
    return summary.sort_values(["mean_rank", "item"])
