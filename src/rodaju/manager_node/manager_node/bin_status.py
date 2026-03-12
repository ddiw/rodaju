#!/usr/bin/env python3
"""분류함 용량 관리."""

import threading

from manager_node.constants import BIN_CAPACITY


class BinStatus:
    def __init__(self):
        self._counts: dict[str, int] = {k: 0 for k in BIN_CAPACITY}
        self._lock = threading.Lock()

    def add(self, bin_id: str):
        with self._lock:
            if bin_id in self._counts:
                self._counts[bin_id] += 1

    def count(self, bin_id: str) -> int:
        return self._counts.get(bin_id, 0)

    def remaining(self, bin_id: str) -> int:
        return max(0, BIN_CAPACITY.get(bin_id, 0) - self._counts.get(bin_id, 0))

    def percent_full(self, bin_id: str) -> float:
        cap = BIN_CAPACITY.get(bin_id, 1)
        return min(100.0, self._counts.get(bin_id, 0) / cap * 100.0)

    def is_full(self, bin_id: str) -> bool:
        return self.remaining(bin_id) <= 0

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._counts)
