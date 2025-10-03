#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ollama Utilities Module
Ollama utilities module providing data processing, file operations and other utility functions
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import pandas as pd
except ImportError:
    print("Installing pandas for data processing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    import pandas as pd


def get_test_prompts() -> List[str]:
    """Return built-in test prompts (medium difficulty, ~400 token output)"""
    return [
        "Given two strings word1 and word2, calculate the minimum number of operations required to convert word1 to word2. Allowed operations include inserting a character, deleting a character, and replacing a character. Please provide a complete Python implementation including algorithm explanation, time complexity and space complexity analysis. Limit output to 400 tokens.",
        "Given a string containing only '(' and ')', find the length of the longest valid (well-formed and continuous) parentheses substring. Please provide a Python solution including algorithm explanation, code implementation, test cases, and complexity analysis. Limit output to 400 tokens.",
        "Given a 2D matrix consisting of 'X' and 'O', find all regions surrounded by 'X' and fill these regions' 'O' with 'X'. A surrounded region is one that is not connected to 'O' on the boundary. Please provide a complete Python implementation including algorithm explanation and complexity analysis. Limit output to 400 tokens.",
    ]


def safe_float(value: Any) -> float:
    """Convert arbitrary value to float"""
    try:
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        return float(str(value))
    except Exception:
        return 0.0


def export_results_to_csv(results: List[Dict[str, Any]], output_file: Path) -> None:
    """Export detailed results to CSV file"""
    if not results:
        print("No results to export")
        return
    
    rows: List[Dict[str, Any]] = []
    for model_result in results:
        for detail in model_result["details"]:
            rows.append(
                {
                    "model": detail["model"],
                    "prompt": detail["prompt"],
                    "output_tokens": detail["output_tokens"],
                    "duration": f"{detail['duration']:.2f}",
                    "tokens_per_second": f"{detail['tokens_per_second']:.2f}",
                    "cpu_util_before": detail.get("cpu_util_before"),
                    "cpu_util_after": detail.get("cpu_util_after"),
                    "response_preview": detail["response_preview"],
                }
            )
    
    frame = pd.DataFrame(rows)
    frame.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\nDetailed results exported to: {output_file}")


def generate_summary_table(results: List[Dict[str, Any]], output_file: Path) -> None:
    """Generate and save summary table for each model"""
    if not results:
        return
    
    summary = []
    for model_result in results:
        summary.append(
            {
                "model": model_result["model"],
                "output_tokens": model_result["total_tokens"],
                "total_duration": f"{model_result['total_duration']:.2f}",
                "tokens_per_second": f"{model_result['average_speed']:.2f}",
                "success_rate": f"{model_result['success_rate']:.2f}",
            }
        )
    
    frame = pd.DataFrame(summary)
    frame.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Summary table exported to: {output_file}")
    print(f"\n{'=' * 80}")
    print("SPEED TEST SUMMARY")
    print(f"{'=' * 80}")
    print(frame.to_string(index=False))
    print(f"{'=' * 80}")


def save_json_results(results: List[Dict[str, Any]], output_file: Path, timestamp: str, total_models: int) -> None:
    """Save results to JSON file"""
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "timestamp": timestamp,
                "total_models": total_models,
                "results": results,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


def create_timestamp() -> str:
    """Create timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_output_dir(output_dir: Path) -> None:
    """Ensure output directory exists"""
    output_dir.mkdir(exist_ok=True)

