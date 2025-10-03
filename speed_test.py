#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Speed Test Tool
Batch testing tool for multiple model performance evaluation
"""

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    import subprocess
    import sys
    print("Installing tqdm for progress bars...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    from tqdm import tqdm

from visualization import run_visualization_analysis
from ollama_client import OllamaClient
from ollama_utils import (
    get_test_prompts,
    export_results_to_csv,
    generate_summary_table,
    save_json_results,
    create_timestamp,
    ensure_output_dir
)
from resource_monitor import ResourceMonitor



# Test prompts moved to ollama_utils.py

# Resource monitoring moved to resource_monitor.py


def test_model_with_prompt(
    client: OllamaClient,
    model_name: str,
    prompt: str,
    collect_resources: bool = True,
) -> Dict[str, Any]:
    """Run a single prompt against a model and collect metrics."""
    monitor = ResourceMonitor() if collect_resources else None
    
    gpu_before = None
    cpu_before = None
    if collect_resources and monitor:
        snapshot = monitor.get_snapshot()
        gpu_before = snapshot.get("gpu")
        cpu_before = snapshot.get("cpu")

    # Use the client to query the model
    result = client.query_model(model_name, prompt, num_predict=400)

    gpu_after = None
    cpu_after = None
    if collect_resources and monitor:
        snapshot = monitor.get_snapshot()
        gpu_after = snapshot.get("gpu")
        cpu_after = snapshot.get("cpu")

    # Enhance result with resource monitoring data
    if result["success"]:
        result.update({
            "gpu_util_before": gpu_before,
            "gpu_util_after": gpu_after,
            "cpu_util_before": cpu_before,
            "cpu_util_after": cpu_after,
        })
        # Rename tokens to output_tokens for consistency
        result["output_tokens"] = result.pop("tokens")
    else:
        result.update({
            "duration": result.get("duration", 0.0),
            "gpu_util_before": gpu_before,
            "gpu_util_after": gpu_after,
            "cpu_util_before": cpu_before,
            "cpu_util_after": cpu_after,
        })

    return result


def run_speed_test(
    client: OllamaClient,
    model_name: str,
    prompts: List[str],
    output_dir: Path,
    progress: Optional['tqdm'] = None,
    collect_resources: bool = True,
) -> Dict[str, Any]:
    """Run speed test across all prompts for a model."""
    results: List[Dict[str, Any]] = []
    total_tokens = 0
    total_duration = 0.0
    successful_tests = 0

    for prompt in prompts:
        outcome = test_model_with_prompt(
            client,
            model_name,
            prompt,
            collect_resources=collect_resources,
        )

        if outcome["success"]:
            successful_tests += 1
            total_tokens += outcome["output_tokens"]
            total_duration += outcome["duration"]
            results.append(
                {
                    "model": model_name,
                    "prompt": prompt,
                    "output_tokens": outcome["output_tokens"],
                    "duration": outcome["duration"],
                    "tokens_per_second": outcome["tokens_per_second"],
                    "gpu_util_before": outcome.get("gpu_util_before"),
                    "gpu_util_after": outcome.get("gpu_util_after"),
                    "cpu_util_before": outcome.get("cpu_util_before"),
                    "cpu_util_after": outcome.get("cpu_util_after"),
                    "response_preview": outcome["response"][:200],
                }
            )

        if progress:
            progress.update(1)

    avg_speed = total_tokens / total_duration if total_duration > 0 else 0.0
    success_rate = successful_tests / len(prompts) if prompts else 0.0

    return {
        "model": model_name,
        "total_tokens": total_tokens,
        "total_duration": total_duration,
        "average_speed": avg_speed,
        "success_rate": success_rate,
        "test_count": len(results),
        "details": results,
    }


# CSV export and summary functions moved to ollama_utils.py
# Model listing function moved to ollama_client.py
# Resource monitoring functions moved to resource_monitor.py






def main() -> None:
    parser = argparse.ArgumentParser(
        description="Model speed test tool - batch testing with resource monitoring and analysis",
    )
    parser.add_argument("--base-url", default="http://localhost:11434", help="Base URL of Ollama service")
    parser.add_argument("--model", help="Specific model to test (test all if not specified)")
    parser.add_argument("--output-dir", default="speed_test_results", help="Output directory for raw results")
    parser.add_argument("--analysis-dir", default="analysis_results", help="Directory for analysis outputs")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip visualization and reporting")
    parser.add_argument("--collect-resources", action="store_true", default=True, help="Collect GPU/CPU usage metrics")

    args = parser.parse_args()

    # Initialize client and utilities
    client = OllamaClient(args.base_url, timeout=120)
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)

    print("Loading test prompts...")
    prompts = get_test_prompts()
    print(f"Loaded {len(prompts)} test prompts")

    if args.model:
        models = [args.model]
    else:
        print("\nDiscovering available models...")
        models = client.get_available_models()
        print(f"Found {len(models)} models: {', '.join(models)}")

    if not models:
        print("No models to test")
        return

    total_tests = len(models) * len(prompts)
    progress_bar = tqdm(
        total=total_tests,
        desc="Initializing speed test...",
        unit="test",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    all_results: List[Dict[str, Any]] = []
    timestamp = create_timestamp()

    try:
        for model in models:
            progress_bar.set_description(f"Testing model: {model}")
            result = run_speed_test(
                client,
                model,
                prompts,
                output_dir,
                progress_bar,
                collect_resources=args.collect_resources,
            )
            all_results.append(result)
    finally:
        progress_bar.close()

    detail_file = output_dir / f"speed_test_details_{timestamp}.csv"
    summary_file = output_dir / f"speed_test_summary_{timestamp}.csv"
    json_file = output_dir / f"speed_test_results_{timestamp}.json"

    export_results_to_csv(all_results, detail_file)
    generate_summary_table(all_results, summary_file)
    save_json_results(all_results, json_file, timestamp, len(models))

    print(f"\nAll results saved to: {output_dir}/")
    print(f"  - Detailed CSV: {detail_file.name}")
    print(f"  - Summary CSV: {summary_file.name}")
    print(f"  - JSON data: {json_file.name}")

    if args.skip_analysis:
        print("Analysis skipped by user request.")
        return

    analysis_dir = Path(args.analysis_dir)
    run_visualization_analysis(all_results, analysis_dir)


if __name__ == '__main__':
    main()