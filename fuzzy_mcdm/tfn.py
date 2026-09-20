"""Triangular fuzzy number utilities.

A triangular fuzzy number is represented as a NumPy array: [lower, modal, upper].
All functions support broadcasting over the final dimension of length 3.
"""

from __future__ import annotations

import numpy as np


def as_tfn_array(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape[-1] != 3:
        raise ValueError("Triangular fuzzy numbers must have a final dimension of size 3: (l, m, u).")
    return arr


def validate_tfn(values, *, name: str = "TFN") -> np.ndarray:
    arr = as_tfn_array(values)
    if np.any(arr[..., 0] > arr[..., 1]) or np.any(arr[..., 1] > arr[..., 2]):
        raise ValueError(f"{name} must satisfy lower <= modal <= upper for every entry.")
    return arr


def reciprocal(values) -> np.ndarray:
    """Return reciprocal TFNs: (1/u, 1/m, 1/l)."""
    arr = validate_tfn(values)
    if np.any(arr <= 0):
        raise ValueError("Reciprocal TFNs require strictly positive values.")
    return np.stack([1.0 / arr[..., 2], 1.0 / arr[..., 1], 1.0 / arr[..., 0]], axis=-1)


def multiply(a, b) -> np.ndarray:
    """Multiply two positive TFNs component-wise."""
    a = validate_tfn(a, name="a")
    b = validate_tfn(b, name="b")
    return np.stack([a[..., 0] * b[..., 0], a[..., 1] * b[..., 1], a[..., 2] * b[..., 2]], axis=-1)


def geometric_mean(values, axis=0) -> np.ndarray:
    """Component-wise fuzzy geometric mean along an axis."""
    arr = validate_tfn(values)
    if np.any(arr < 0):
        raise ValueError("Geometric mean requires non-negative TFN components.")
    n = arr.shape[axis]
    return np.prod(arr, axis=axis) ** (1.0 / n)


def defuzzify(values, method: str = "coa") -> np.ndarray:
    """Defuzzify TFNs.

    Parameters
    ----------
    method:
        "coa" or "centroid" uses (l + m + u) / 3.
    """
    arr = validate_tfn(values)
    if method.lower() not in {"coa", "centroid", "center_of_area"}:
        raise ValueError("Only Center of Area / centroid defuzzification is implemented.")
    return arr.mean(axis=-1)


def normalize_crisp(weights) -> np.ndarray:
    arr = np.asarray(weights, dtype=float)
    total = arr.sum()
    if total <= 0:
        raise ValueError("Weights must have a positive sum to normalize.")
    return arr / total
