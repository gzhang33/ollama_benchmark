#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HTTP Query Runner Module
Encapsulates the async HTTP method used by the GUI to query models with streaming support.
"""

from typing import Any, Dict, List, Optional, Callable

import asyncio

from src.async_ollama_client import AsyncOllamaClient, ModelTaskQueue, StreamChunk


class HttpQueryRunner:
    """Encapsulated runner for HTTP async model queries."""

    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 60):
        self.base_url = base_url
        self.timeout = timeout
        self._async_client: Optional[AsyncOllamaClient] = None
        self._task_queue: Optional[ModelTaskQueue] = None

    async def _ensure_client(self) -> None:
        if self._async_client is None:
            self._async_client = AsyncOllamaClient(self.base_url, self.timeout)
            self._task_queue = ModelTaskQueue(self._async_client)

    async def execute(
        self,
        models: List[str],
        query: str,
        num_predict: int = 512,
        on_model_start: Optional[Callable[[str, int], Any]] = None,
        on_model_chunk: Optional[Callable[[StreamChunk], Any]] = None,
        on_model_complete: Optional[Callable[[str, Dict[str, Any], int], Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute HTTP streaming queries for the provided models sequentially.

        The callbacks mirror the ones used in the GUI layer.
        """
        await self._ensure_client()

        assert self._async_client is not None
        assert self._task_queue is not None

        async with self._async_client:
            results = await self._task_queue.execute_sequential_tasks(
                models,
                query,
                num_predict=num_predict,
                on_model_start=on_model_start,
                on_model_chunk=on_model_chunk,
                on_model_complete=on_model_complete,
            )
        return results


