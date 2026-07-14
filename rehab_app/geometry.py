from __future__ import annotations

import numpy as np


def calc_angle(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    """Calculate angle ABC in degrees."""
    point_a = np.array(a, dtype=float)
    point_b = np.array(b, dtype=float)
    point_c = np.array(c, dtype=float)

    ba = point_a - point_b
    bc = point_c - point_b
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator == 0:
        return 0.0

    cosine = float(np.dot(ba, bc) / denominator)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))
