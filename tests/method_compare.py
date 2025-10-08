#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Method comparison CLI: Direct vs HTTP total time with consistent settings.

Usage:
  python tests/method_compare.py "Your question here"
"""

import asyncio
import sys
import time
from typing import Any, Dict, List

from src.unified_ollama_client import UnifiedOllamaClient
from src.direct_ollama_tester import DirectOllamaTester
from src.unified_http_query_runner import UnifiedHttpQueryRunner


def pick_models(all_models: List[str], max_count: int = 2) -> List[str]:
    # Choose first N models to keep runtime predictable
    return all_models[:max_count]


def run_direct(models: List[str], prompt: str, num_predict: int = 400) -> float:
    tester = DirectOllamaTester()
    t0 = time.perf_counter()
    for m in models:
        _ = tester.test_output_generation_time_optimized(
            m, prompt, num_predict=num_predict, use_cache=True
        )
    return time.perf_counter() - t0


async def run_http_async(models: List[str], prompt: str, num_predict: int = 400) -> float:
    runner = UnifiedHttpQueryRunner()

    async def on_start(model: str, index: int):
        return None

    async def on_chunk(_chunk):
        return None

    async def on_complete(model: str, result: Dict[str, Any], index: int):
        return None

    t0 = time.perf_counter()
    await runner.execute(
        models,
        prompt,
        num_predict=num_predict,
        on_model_start=on_start,
        on_model_chunk=on_chunk,
        on_model_complete=on_complete,
    )
    return time.perf_counter() - t0


def run_http(models: List[str], prompt: str, num_predict: int = 400) -> float:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(run_http_async(models, prompt, num_predict))
    finally:
        loop.close()


def main() -> None:
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Explain quicksort in brief."

    client = UnifiedOllamaClient()
    all_models = client.get_available_models()
    client.close_sync()
    if not all_models:
        print("No models available. Ensure Ollama is running and models are pulled.")
        sys.exit(1)

    models = pick_models(all_models, max_count=2)
    print(f"Models: {models}")
    print(f"num_predict: 400")
    print(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

    direct_total = run_direct(models, prompt, num_predict=400)
    http_total = run_http(models, prompt, num_predict=400)

    print("\n=== Method Comparison ===")
    print(f"Direct total: {direct_total:.2f}s")
    print(f"HTTP total:   {http_total:.2f}s")


if __name__ == "__main__":
    main()



