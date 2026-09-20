"""Fuzzy DEMATEL implementation for causal mapping of barriers.

The implementation aggregates expert triangular fuzzy direct-influence matrices
with arithmetic averaging, defuzzifies them using Center of Area, normalizes the
matrix, computes the total relation matrix, and derives D, R, D+R, and D-R
indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .tfn import defuzzify, validate_tfn


@dataclass
class DEMATELResult:
    items: list[str]
    direct_matrix_fuzzy: np.ndarray
    direct_matrix_crisp: np.ndarray
    normalized_matrix: np.ndarray
    total_relation_matrix: np.ndarray
    threshold: float
    results: pd.DataFrame


def long_influence_to_matrices(df: pd.DataFrame, items: Iterable[str]) -> dict[str, np.ndarray]:
    required = {"expert", "source", "target", "l", "m", "u"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required DEMATEL influence columns: {sorted(missing)}")

    item_list = list(items)
    pos = {item: i for i, item in enumerate(item_list)}
    n = len(item_list)
    matrices: dict[str, np.ndarray] = {}

    for expert, group in df.groupby("expert"):
        mat = np.zeros((n, n, 3), dtype=float)
        for _, row in group.iterrows():
            source = row["source"]
            target = row["target"]
            if source not in pos or target not in pos or source == target:
                continue
            tfn = validate_tfn(np.array([row["l"], row["m"], row["u"]], dtype=float), name="influence")
            mat[pos[source], pos[target], :] = tfn
        # Require explicit off-diagonal entries so that missing survey responses
        # are not silently interpreted as "no influence". A true no-influence
        # judgment should be entered as the TFN (0, 0, 0).
        seen_pairs = {(str(row["source"]), str(row["target"])) for _, row in group.iterrows()}
        missing_pairs = []
        for source in item_list:
            for target in item_list:
                if source != target and (source, target) not in seen_pairs:
                    missing_pairs.append((source, target))
        if missing_pairs:
            raise ValueError(f"Expert {expert} influence matrix has missing off-diagonal entries: {missing_pairs[:10]}")
        matrices[str(expert)] = mat
    return matrices


def compute_fuzzy_dematel(matrices: dict[str, np.ndarray], items: Iterable[str], threshold: float | None = None) -> DEMATELResult:
    item_list = list(items)
    if not matrices:
        raise ValueError("At least one expert influence matrix is required.")

    stack = np.stack([validate_tfn(m, name=f"matrix_{i}") for i, m in enumerate(matrices.values())], axis=0)
    # Fuzzy DEMATEL commonly aggregates expert direct-influence judgments by
    # arithmetic mean of the TFN components. This keeps zero/low influence values
    # from being over-compressed by geometric averaging.
    direct_fuzzy = stack.mean(axis=0)
    direct_crisp = defuzzify(direct_fuzzy)
    np.fill_diagonal(direct_crisp, 0.0)

    max_row_sum = direct_crisp.sum(axis=1).max()
    max_col_sum = direct_crisp.sum(axis=0).max()
    normalizer = max(max_row_sum, max_col_sum)
    if normalizer <= 0:
        raise ValueError("The direct influence matrix has no positive influence values.")

    normalized = direct_crisp / normalizer
    identity = np.eye(len(item_list))
    total_relation = normalized @ np.linalg.inv(identity - normalized)

    D = total_relation.sum(axis=1)
    R = total_relation.sum(axis=0)
    prominence = D + R
    relation = D - R

    if threshold is None:
        off_diag = total_relation[~np.eye(len(item_list), dtype=bool)]
        threshold = float(off_diag.mean())

    result_df = pd.DataFrame({
        "barrier": item_list,
        "D": D,
        "R": R,
        "prominence_D_plus_R": prominence,
        "relation_D_minus_R": relation,
        "type": np.where(relation >= 0, "Cause", "Effect"),
    })
    result_df["rank_by_prominence"] = result_df["prominence_D_plus_R"].rank(ascending=False, method="min").astype(int)
    result_df = result_df.sort_values("rank_by_prominence")

    return DEMATELResult(
        items=item_list,
        direct_matrix_fuzzy=direct_fuzzy,
        direct_matrix_crisp=direct_crisp,
        normalized_matrix=normalized,
        total_relation_matrix=total_relation,
        threshold=threshold,
        results=result_df,
    )


def run_fuzzy_dematel_from_influence(csv_path: str, items: Iterable[str], threshold: float | None = None) -> DEMATELResult:
    df = pd.read_csv(csv_path)
    matrices = long_influence_to_matrices(df, items)
    return compute_fuzzy_dematel(matrices, items, threshold=threshold)
