from __future__ import annotations

from dataclasses import dataclass
from math import acos, degrees
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0

    @classmethod
    def from_iterable(cls, values: Iterable[float]) -> "Point3D":
        items = list(values)
        if len(items) == 2:
            return cls(items[0], items[1])
        if len(items) == 3:
            return cls(items[0], items[1], items[2])
        return cls(items[0], items[1], items[2], items[3])

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)


def angle_between_points(a: Point3D, b: Point3D, c: Point3D) -> float:
    """Return angle ABC in degrees."""
    ba = a.as_array() - b.as_array()
    bc = c.as_array() - b.as_array()
    return angle_between_vectors(ba, bc)


def angle_between_vectors(first: np.ndarray, second: np.ndarray) -> float:
    first_norm = np.linalg.norm(first)
    second_norm = np.linalg.norm(second)
    if first_norm == 0 or second_norm == 0:
        return 0.0

    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    cosine = max(-1.0, min(1.0, cosine))
    return degrees(acos(cosine))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def score_range(value: float, target_low: float, target_high: float, tolerance: float) -> float:
    """Score 0-100 for values inside or close to a target range."""
    if target_low <= value <= target_high:
        return 100.0
    if value < target_low:
        distance = target_low - value
    else:
        distance = value - target_high
    return clamp(100.0 - (distance / tolerance) * 100.0, 0.0, 100.0)
