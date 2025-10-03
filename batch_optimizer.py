from __future__ import annotations

from typing import Dict, List, Tuple


class BatchProcessor:
    """Create prompt batches grouped by approximate length."""

    def __init__(self, max_batch_size: int = 6, min_batch_size: int = 1) -> None:
        self.max_batch_size = max(1, int(max_batch_size))
        self.min_batch_size = max(1, int(min_batch_size))

    def generate_batches(self, prompts: Dict[str, List[str]]) -> List[List[Tuple[str, str]]]:
        tasks: List[Tuple[str, str, int]] = []
        for difficulty, items in prompts.items():
            if difficulty != "medium":
                continue
            for prompt in items:
                length = len(prompt.split())
                tasks.append((difficulty, prompt, length))

        if not tasks:
            return []

        tasks.sort(key=lambda item: item[2])
        batches: List[List[Tuple[str, str]]] = []
        index = 0
        while index < len(tasks):
            length = tasks[index][2]
            batch_size = self._batch_size_for_length(length)
            chunk = tasks[index:index + batch_size]
            batches.append([(difficulty, prompt) for difficulty, prompt, _ in chunk])
            index += batch_size
        return batches

    def _batch_size_for_length(self, length: int) -> int:
        if length < 50:
            return self.max_batch_size
        if length <= 200:
            return max(self.min_batch_size, self.max_batch_size // 2 or 1)
        return max(self.min_batch_size, self.max_batch_size // 4 or 1)


__all__ = ["BatchProcessor"]
