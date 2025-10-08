#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified HTTP Query Runner Module
统一HTTP查询运行器 - 使用统一的httpx客户端
"""

from typing import Any, Dict, List, Optional, Callable
import asyncio
from src.unified_ollama_client import UnifiedOllamaClient, ModelTaskQueue, StreamChunk


class UnifiedHttpQueryRunner:
    """统一HTTP查询运行器"""

    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 60):
        self.base_url = base_url
        self.timeout = timeout
        self._client: Optional[UnifiedOllamaClient] = None
        self._task_queue: Optional[ModelTaskQueue] = None

    async def _ensure_client(self) -> None:
        if self._client is None:
            self._client = UnifiedOllamaClient(self.base_url)
            self._task_queue = ModelTaskQueue(self._client)

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
        执行HTTP流式查询
        
        Args:
            models: 模型列表
            query: 查询文本
            num_predict: 最大预测token数
            on_model_start: 模型开始回调
            on_model_chunk: 数据块回调
            on_model_complete: 模型完成回调
            
        Returns:
            所有模型的执行结果
        """
        await self._ensure_client()

        assert self._client is not None
        assert self._task_queue is not None

        # 使用统一的客户端执行任务
        results = await self._task_queue.execute_sequential_tasks(
            models,
            query,
            num_predict=num_predict,
            on_model_start=on_model_start,
            on_model_chunk=on_model_chunk,
            on_model_complete=on_model_complete,
        )
        return results

    async def close(self):
        """关闭客户端连接"""
        if self._client:
            await self._client.close_async()
            self._client = None
            self._task_queue = None
