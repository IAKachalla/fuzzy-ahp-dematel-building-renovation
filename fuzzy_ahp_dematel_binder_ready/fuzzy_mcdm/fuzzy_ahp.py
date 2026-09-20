"""Fuzzy AHP implementation using triangular fuzzy numbers.

The workflow follows the paper structure:
1. aggregate expert pairwise matrices using fuzzy geometric mean,
2. compute row-wise fuzzy geometric means,
3. normalize fuzzy weights,
4. defuzzify by Center of Area,
5. normalize crisp priorities,
6. calculate consistency ratio using the defuzzified pairwise matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .tfn import defuzzify, geometric_mean, multiply, normalize_crisp, reciprocal, validate_tfn

RI_TABLE = {
    1: 0.00,
    2: 0.00,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49,
    11: 1.51,
    12: 1.48,
    13: 1.56,
    14: 1.57,
    15: 1.59,
}


@dataclass
class FAHPResult:
    items: list[str]
    fuzzy_weights: np.ndarray
    crisp_weights: np.ndarray
    aggregated_matrix: np.ndarray
    consistency_ratio: float
    expert_consistency: pd.DataFrame

    def as_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "item": self.items,
            "fuzzy_l": self.fuzzy_weights[:, 0],
            "fuzzy_m": self.fuzzy_weights[:, 1],
            "fuzzy_u": self.fuzzy_weights[:, 2],
            "weight": self.crisp_weights,
        })
        df["rank"] = df["weight"].rank(ascending=False, method="min").astype(int)
        return df.sort_values("rank")


def consistency_ratio(crisp_matrix: np.ndarray) -> float:
    """Saaty consistency ratio for a positive reciprocal matrix."""
    mat = np.asarray(crisp_matrix, dtype=float)
    if mat.shape[0] != mat.shape[1]:
        raise ValueError("Consistency ratio requires a square matrix.")
    n = mat.shape[0]
    if n <= 2:
        return 0.0
    eigvals = np.linalg.eigvals(mat)
    lambda_max = float(np.max(eigvals.real))
    ci = (lambda_max - n) / (n - 1)
    ri = RI_TABLE.get(n)
    if ri is None:
        raise ValueError(f"No RI value is available for n={n}. Extend RI_TABLE if needed.")
    return float(ci / ri) if ri > 0 else 0.0


def long_pairwise_to_matrices(df: pd.DataFrame, items: Iterable[str]) -> dict[str, np.ndarray]:
    """Convert long-format pairwise data to one TFN matrix per expert.

    Expected columns: expert, item_i, item_j, l, m, u
    Only upper-triangle comparisons are required; reciprocals are added automatically.
    """
    required = {"expert", "item_i", "item_j", "l", "m", "u"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required pairwise columns: {sorted(missing)}")

    item_list = list(items)
    pos = {item: i for i, item in enumerate(item_list)}
    n = len(item_list)
    matrices: dict[str, np.ndarray] = {}

    for expert, group in df.groupby("expert"):
        mat = np.zeros((n, n, 3), dtype=float)
        for i in range(n):
            mat[i, i, :] = (1.0, 1.0, 1.0)

        for _, row in group.iterrows():
            a = row["item_i"]
            b = row["item_j"]
            if a not in pos or b not in pos:
                continue
            tfn = validate_tfn(np.array([row["l"], row["m"], row["u"]], dtype=float))
            i, j = pos[a], pos[b]
            mat[i, j, :] = tfn
            if i != j:
                mat[j, i, :] = reciprocal(tfn)

        if np.any(mat == 0):
            missing_pairs = []
            for i in range(n):
                for j in range(n):
                    if np.all(mat[i, j, :] == 0):
                        missing_pairs.append((item_list[i], item_list[j]))
            raise ValueError(f"Expert {expert} matrix has missing comparisons: {missing_pairs[:10]}")
        matrices[str(expert)] = mat

    return matrices


def compute_fuzzy_ahp(matrices: dict[str, np.ndarray], items: Iterable[str]) -> FAHPResult:
    item_list = list(items)
    if not matrices:
        raise ValueError("At least one expert matrix is required.")

    stack = np.stack([validate_tfn(m) for m in matrices.values()], axis=0)
    aggregated = geometric_mean(stack, axis=0)

    n = len(item_list)
    row_geomeans = geometric_mean(aggregated, axis=1)
    sum_geomeans = row_geomeans.sum(axis=0)
    inv_sum = reciprocal(sum_geomeans)
    fuzzy_weights = multiply(row_geomeans, inv_sum)
    crisp_weights = normalize_crisp(defuzzify(fuzzy_weights))

    agg_crisp = defuzzify(aggregated)
    cr = consistency_ratio(agg_crisp)

    expert_rows = []
    for expert, mat in matrices.items():
        expert_rows.append({
            "expert": expert,
            "consistency_ratio": consistency_ratio(defuzzify(mat)),
        })
    expert_consistency = pd.DataFrame(expert_rows)

    return FAHPResult(
        items=item_list,
        fuzzy_weights=fuzzy_weights,
        crisp_weights=crisp_weights,
        aggregated_matrix=aggregated,
        consistency_ratio=cr,
        expert_consistency=expert_consistency,
    )


def run_fuzzy_ahp_from_pairwise(csv_path: str, context: str, items: Iterable[str]) -> FAHPResult:
    df = pd.read_csv(csv_path)
    if "context" in df.columns:
        df = df[df["context"] == context].copy()
    matrices = long_pairwise_to_matrices(df, items)
    return compute_fuzzy_ahp(matrices, items)
