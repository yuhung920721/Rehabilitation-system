from __future__ import annotations

from collections import defaultdict, deque
from statistics import fmean


class AngleSmoother:
    def __init__(self, window_size: int = 5) -> None:
        self.window_size = window_size
        self._values: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window_size))

    def update(self, key: str, value: float) -> float:
        values = self._values[key]
        values.append(value)
        return fmean(values)
