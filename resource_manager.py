from __future__ import annotations

import math
from collections import deque
from typing import Any, Deque, Dict, Optional


class ResourceManager:
    """Adaptive heuristics for concurrency and resource sampling."""

    def __init__(
        self,
        gpu_threshold: float = 0.8,
        cpu_threshold: float = 0.7,
        min_concurrency: int = 1,
        max_concurrency: int = 4,
    ) -> None:
        self.gpu_threshold = max(0.0, float(gpu_threshold))
        self.cpu_threshold = max(0.0, float(cpu_threshold))
        self.min_concurrency = max(1, int(min_concurrency))
        self.max_concurrency = max(self.min_concurrency, int(max_concurrency))
        self._history: Deque[Dict[str, float]] = deque(maxlen=100)
        self._adaptive_concurrency = self.min_concurrency

    def get_optimal_concurrency(self) -> int:
        """Return the current concurrency hint."""
        return self._adaptive_concurrency

    def update_resource_history(self, sample: Dict[str, Any]) -> None:
        """Record the latest resource metrics and adjust concurrency."""
        entry: Dict[str, float] = {}

        gpu = sample.get("gpu_util")
        if isinstance(gpu, (int, float)) and math.isfinite(gpu):
            value = float(gpu)
            if value > 1.0:
                value = value / 100.0
            entry["gpu_util"] = max(0.0, min(value, 1.0))

        cpu = sample.get("cpu_util")
        if isinstance(cpu, (int, float)) and math.isfinite(cpu):
            value = float(cpu)
            if value > 1.0:
                value = value / 100.0
            entry["cpu_util"] = max(0.0, min(value, 1.0))

        duration = sample.get("duration")
        if isinstance(duration, (int, float)) and math.isfinite(duration):
            entry["duration"] = max(0.0, float(duration))

        if not entry:
            return

        self._history.append(entry)
        self._adjust_concurrency(entry)

    def collection_interval(self, base_interval: int, concurrency_hint: Optional[int] = None) -> int:
        """Return adjusted sampling interval based on concurrency hint."""
        if base_interval <= 0:
            return 0
        hint = concurrency_hint if concurrency_hint is not None else self._adaptive_concurrency
        hint = max(1, int(hint))
        return max(1, base_interval // hint)

    def _adjust_concurrency(self, latest: Dict[str, float]) -> None:
        gpu = latest.get("gpu_util", 0.0)
        cpu = latest.get("cpu_util", 0.0)

        if gpu < self.gpu_threshold * 0.8 and cpu < self.cpu_threshold * 0.8:
            self._adaptive_concurrency = min(self._adaptive_concurrency + 1, self.max_concurrency)
        elif gpu > self.gpu_threshold or cpu > self.cpu_threshold:
            self._adaptive_concurrency = max(self._adaptive_concurrency - 1, self.min_concurrency)


__all__ = ["ResourceManager"]
