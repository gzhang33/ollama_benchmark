#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Speed Test Tool with Resource Monitoring
模型速度测试工具（含资源监控）
"""

import argparse
import csv
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm for progress bars...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    from tqdm import tqdm


def load_test_prompts(file_path: str = "test_prompts.txt") -> Dict[str, List[str]]:
    """Load test prompts from file grouped by difficulty level."""
    prompts = defaultdict(list)
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Handle comments: skip lines that start with # or remove inline comments
            if line.startswith("#"):
                continue
            
            # Remove inline comments (everything after #)
            if "#" in line:
                line = line.split("#")[0].strip()
                if not line:  # If only comment remains, skip this line
                    continue
            
            # Parse difficulty levels
            if line.startswith("[SIMPLE]"):
                prompt_text = line.replace("[SIMPLE]", "").strip()
                if prompt_text:  # Only add if there's actual content
                    prompts["simple"].append(prompt_text)
            elif line.startswith("[MEDIUM]"):
                prompt_text = line.replace("[MEDIUM]", "").strip()
                if prompt_text:
                    prompts["medium"].append(prompt_text)
            elif line.startswith("[COMPLEX]"):
                prompt_text = line.replace("[COMPLEX]", "").strip()
                if prompt_text:
                    prompts["complex"].append(prompt_text)
    
    return prompts


def get_gpu_usage() -> Optional[Dict[str, Any]]:
    """Get GPU usage via nvidia-smi if available."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", 
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            gpu_data = []
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    gpu_data.append({
                        "gpu_util": float(parts[0]),
                        "memory_used": float(parts[1]),
                        "memory_total": float(parts[2])
                    })
            return {"gpus": gpu_data} if gpu_data else None
    except Exception as e:
        print(f"  [Warning] GPU monitoring unavailable: {e}")
    return None


def get_cpu_usage() -> Optional[float]:
    """Get CPU usage percentage."""
    try:
        import psutil
        return psutil.cpu_percent(interval=0.5)
    except Exception as e:
        print(f"  [Warning] CPU monitoring unavailable: {e}")
    return None


def test_model_with_prompt(
    base_url: str,
    model_name: str,
    prompt: str,
    timeout: int = 120
) -> Dict[str, Any]:
    """Test a single model with one prompt and collect metrics."""
    url = f"{base_url}/api/generate"
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 512
        }
    }
    
    # Collect resource usage before test
    gpu_before = get_gpu_usage()
    cpu_before = get_cpu_usage()
    
    start_time = time.perf_counter()
    
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        data = response.json()
        response_text = data.get("response", "")
        
        # Token count estimation
        eval_count = data.get("eval_count", 0)
        if eval_count == 0:
            eval_count = len(response_text.split())
        
        tokens_per_second = eval_count / duration if duration > 0 else 0
        
        # Collect resource usage after test
        gpu_after = get_gpu_usage()
        cpu_after = get_cpu_usage()
        
        return {
            "success": True,
            "response": response_text,
            "output_tokens": eval_count,
            "duration": duration,
            "tokens_per_second": tokens_per_second,
            "gpu_before": gpu_before,
            "gpu_after": gpu_after,
            "cpu_before": cpu_before,
            "cpu_after": cpu_after
        }
        
    except requests.Timeout:
        return {
            "success": False,
            "error": "Timeout",
            "duration": timeout
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "duration": 0
        }


def run_speed_test(
    base_url: str,
    model_name: str,
    prompts: Dict[str, List[str]],
    output_dir: Path,
    progress_bar: tqdm = None
) -> Dict[str, Any]:
    """Run complete speed test for a model with progress tracking."""
    results = []
    total_tokens = 0
    total_duration = 0
    test_times = []  # Track individual test times for ETA calculation
    
    # Flatten all prompts for progress tracking
    all_tests = []
    for difficulty, prompt_list in prompts.items():
        for prompt in prompt_list:
            all_tests.append((difficulty, prompt))
    
    for idx, (difficulty, prompt) in enumerate(all_tests):
        test_start_time = time.time()
        
        result = test_model_with_prompt(base_url, model_name, prompt)
        
        test_duration = time.time() - test_start_time
        test_times.append(test_duration)
        
        if result["success"]:
            total_tokens += result["output_tokens"]
            total_duration += result["duration"]
            
            results.append({
                "model": model_name,
                "difficulty": difficulty,
                "prompt": prompt,
                "output_tokens": result["output_tokens"],
                "duration": result["duration"],
                "tokens_per_second": result["tokens_per_second"],
                "gpu_util_before": result.get("gpu_before"),
                "gpu_util_after": result.get("gpu_after"),
                "cpu_util_before": result.get("cpu_before"),
                "cpu_util_after": result.get("cpu_util_after"),
                "response_preview": result["response"][:200]
            })
            
            # Update progress bar description
            if progress_bar:
                avg_test_time = sum(test_times) / len(test_times)
                remaining_tests = len(all_tests) - (idx + 1)
                eta_seconds = avg_test_time * remaining_tests
                
                if eta_seconds < 60:
                    eta_str = f"{eta_seconds:.0f}s"
                elif eta_seconds < 3600:
                    eta_str = f"{eta_seconds/60:.1f}m"
                else:
                    eta_str = f"{eta_seconds/3600:.1f}h"
                
                progress_bar.set_description(
                    f"Testing {model_name} [{difficulty.upper()}] - ETA: {eta_str}"
                )
                progress_bar.set_postfix({
                    'Speed': f"{result['tokens_per_second']:.1f} tok/s",
                    'Success': '✓'
                })
        
        else:
            # Update progress bar for failed test
            if progress_bar:
                progress_bar.set_postfix({
                    'Error': result.get('error', 'Unknown')[:20],
                    'Success': '✗'
                })
        
        # Update progress bar
        if progress_bar:
            progress_bar.update(1)
    
    # Calculate average speed
    avg_speed = total_tokens / total_duration if total_duration > 0 else 0
    
    return {
        "model": model_name,
        "total_tokens": total_tokens,
        "total_duration": total_duration,
        "average_speed": avg_speed,
        "test_count": len(results),
        "details": results
    }


def export_results_to_csv(results: List[Dict[str, Any]], output_file: Path) -> None:
    """Export detailed results to CSV file."""
    if not results:
        print("No results to export")
        return
    
    rows = []
    for model_result in results:
        for detail in model_result["details"]:
            rows.append({
                "model": detail["model"],
                "difficulty": detail["difficulty"],
                "prompt": detail["prompt"],
                "output_tokens": detail["output_tokens"],
                "duration": f"{detail['duration']:.2f}",
                "tokens_per_second": f"{detail['tokens_per_second']:.2f}",
                "cpu_util_before": detail.get("cpu_util_before"),
                "cpu_util_after": detail.get("cpu_util_after"),
                "response_preview": detail["response_preview"]
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\nDetailed results exported to: {output_file}")


def generate_summary_table(results: List[Dict[str, Any]], output_file: Path) -> None:
    """Generate simplified summary table"""
    if not results:
        return
    
    summary_data = []
    
    for model_result in results:
        model_name = model_result["model"]
        
        # Only keep overall summary data
        summary_data.append({
            "model": model_name,
            "output_tokens": model_result["total_tokens"],
            "total_duration": f"{model_result['total_duration']:.2f}",
            "tokens_per_second": f"{model_result['average_speed']:.2f}"
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Summary table exported to: {output_file}")
    
    # Print summary table
    print(f"\n{'='*80}")
    print("SPEED TEST SUMMARY")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    print(f"{'='*80}")


def get_available_models(base_url: str) -> List[str]:
    """Get available LLM models from Ollama."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        response.raise_for_status()
        data = response.json()
        
        models = []
        for model_info in data.get("models", []):
            model_name = model_info["name"]
            if not any(keyword in model_name.lower() 
                      for keyword in ["embedding", "bge", "bert"]):
                models.append(model_name)
        
        return models
    except Exception as e:
        print(f"Error getting models: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Model speed test tool with resource monitoring"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434",
        help="Base URL of Ollama service"
    )
    parser.add_argument(
        "--prompts-file",
        default="test_prompts.txt",
        help="Path to test prompts file"
    )
    parser.add_argument(
        "--model",
        help="Specific model to test (test all if not specified)"
    )
    parser.add_argument(
        "--output-dir",
        default="speed_test_results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load test prompts
    print("Loading test prompts...")
    prompts = load_test_prompts(args.prompts_file)
    
    total_prompts = sum(len(p) for p in prompts.values())
    print(f"Loaded {total_prompts} test prompts:")
    for diff, prompt_list in prompts.items():
        print(f"  - {diff}: {len(prompt_list)} prompts")
    
    # Get models to test
    if args.model:
        models = [args.model]
    else:
        print("\nDiscovering available models...")
        models = get_available_models(args.base_url)
        print(f"Found {len(models)} models: {', '.join(models)}")
    
    if not models:
        print("No models to test")
        return
    
    # Calculate total tests for progress bar
    total_tests = len(models) * total_prompts
    
    # Create progress bar
    progress_bar = tqdm(
        total=total_tests,
        desc="Initializing speed test...",
        unit="test",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
    
    # Run tests
    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        for model in models:
            result = run_speed_test(args.base_url, model, prompts, output_dir, progress_bar)
            all_results.append(result)
    finally:
        progress_bar.close()
    
    # Export results
    detail_file = output_dir / f"speed_test_details_{timestamp}.csv"
    summary_file = output_dir / f"speed_test_summary_{timestamp}.csv"
    json_file = output_dir / f"speed_test_results_{timestamp}.json"
    
    export_results_to_csv(all_results, detail_file)
    generate_summary_table(all_results, summary_file)
    
    # Export JSON
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "total_models": len(models),
            "results": all_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nAll results saved to: {output_dir}/")
    print(f"  - Detailed CSV: {detail_file.name}")
    print(f"  - Summary CSV: {summary_file.name}")
    print(f"  - JSON data: {json_file.name}")


if __name__ == "__main__":
    main()
