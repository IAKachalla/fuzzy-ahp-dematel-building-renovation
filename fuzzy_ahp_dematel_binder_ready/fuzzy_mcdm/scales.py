"""Default linguistic scales used by the paper-style workflow."""

from __future__ import annotations

FAHP_SCALE = {
    "EI": (1.0, 1.0, 1.0),       # Equally important
    "SMI": (1.0, 2.0, 3.0),      # Slightly more important
    "MMI": (2.0, 3.0, 4.0),      # Moderately more important
    "STR": (4.0, 5.0, 6.0),      # Strongly more important
    "VSTR": (6.0, 7.0, 8.0),     # Very strongly more important
    "EXTR": (8.0, 9.0, 10.0),    # Extremely more important
}

DEMATEL_SCALE = {
    "NO": (0.00, 0.00, 0.00),
    "LOW": (0.00, 0.25, 0.50),
    "MEDIUM": (0.25, 0.50, 0.75),
    "HIGH": (0.50, 0.75, 1.00),
    "VERY_HIGH": (0.75, 1.00, 1.00),
}


def reciprocal_scale(tfn):
    l, m, u = tfn
    return (1 / u, 1 / m, 1 / l)
