#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Speed Test Tool with Integrated Analysis
模型速度测试工具（含资源监控与可视化分析）
"""

import argparse
import json
import math
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from batch_optimizer import BatchProcessor
from resource_manager import ResourceManager

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    print("Installing tqdm for progress bars...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

try:
    import seaborn as sns
except ImportError:  # pragma: no cover
    sns = None

DEFAULT_WEIGHTS: Dict[str, float] = {"speed": 0.5, "efficiency": 0.3, "stability": 0.2}
PLOT_FONTS = ["Microsoft YaHei", "SimHei", "DejaVu Sans", "Arial Unicode MS"]

if plt is not None:
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    plt.rcParams["font.sans-serif"] = PLOT_FONTS
    plt.rcParams["axes.unicode_minus"] = False

if sns is not None:
    sns.set_theme(style="whitegrid")


def load_test_prompts(file_path: str = "test_prompts.txt") -> Dict[str, List[str]]:
    """Load medium difficulty prompts only."""
    prompts: Dict[str, List[str]] = {"medium": []}
    with open(file_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "#" in line:
                line = line.split("#")[0].strip()
                if not line:
                    continue
            if line.startswith("[MEDIUM]"):
                text = line.replace("[MEDIUM]", "").strip()
                if text:
                    prompts["medium"].append(text)
    return prompts


def get_gpu_usage() -> Optional[Dict[str, Any]]:
    """Return GPU stats via nvidia-smi when available."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        entries = []
        for line in result.stdout.strip().split("\n"):
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 3:
                entries.append(
                    {
                        "gpu_util": float(parts[0]),
                        "memory_used": float(parts[1]),
                        "memory_total": float(parts[2]),
                    }
                )
        return {"gpus": entries} if entries else None
    except Exception as exc:  # pragma: no cover
        print(f"  [Warning] GPU monitoring unavailable: {exc}")
        return None


def get_cpu_usage() -> Optional[float]:
    """Return CPU usage percentage."""
    try:
        import psutil

        return psutil.cpu_percent(interval=None)
    except Exception as exc:  # pragma: no cover
        print(f"  [Warning] CPU monitoring unavailable: {exc}")
        return None


def test_model_with_prompt(
    base_url: str,
    model_name: str,
    prompt: str,
    timeout: int = 120,
    collect_resources: bool = True,
) -> Dict[str, Any]:
    """Run a single prompt against a model and collect metrics."""
    call_start = time.perf_counter()
    url = f"{base_url}/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 512},
    }

    timings: Dict[str, float] = {}

    gpu_before = None
    cpu_before = None
    if collect_resources:
        resource_before_start = time.perf_counter()
        gpu_before = get_gpu_usage()
        cpu_before = get_cpu_usage()
        timings["resource_before"] = time.perf_counter() - resource_before_start
    else:
        timings["resource_before"] = math.nan

    request_start = time.perf_counter()
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
    except requests.Timeout:
        timings["request"] = time.perf_counter() - request_start
        timings["resource_after"] = math.nan
        timings["wall"] = time.perf_counter() - call_start
        return {
            "success": False,
            "error": "Timeout",
            "duration": timeout,
            "timings": timings,
        }
    except Exception as exc:
        timings["request"] = time.perf_counter() - request_start
        timings["resource_after"] = math.nan
        timings["wall"] = time.perf_counter() - call_start
        return {
            "success": False,
            "error": str(exc),
            "duration": 0.0,
            "timings": timings,
        }

    duration = time.perf_counter() - request_start
    timings["request"] = duration

    data = response.json()
    output = data.get("response", "")
    tokens = data.get("eval_count", 0) or len(output.split())
    speed = tokens / duration if duration > 0 else 0.0

    gpu_after = None
    cpu_after = None
    if collect_resources:
        resource_after_start = time.perf_counter()
        gpu_after = get_gpu_usage()
        cpu_after = get_cpu_usage()
        timings["resource_after"] = time.perf_counter() - resource_after_start
    else:
        timings["resource_after"] = math.nan

    timings["wall"] = time.perf_counter() - call_start

    return {
        "success": True,
        "response": output,
        "output_tokens": tokens,
        "duration": duration,
        "tokens_per_second": speed,
        "gpu_util_before": gpu_before,
        "gpu_util_after": gpu_after,
        "cpu_util_before": cpu_before,
        "cpu_util_after": cpu_after,
        "timings": timings,
    }

def run_speed_test(
    base_url: str,
    model_name: str,
    prompts: Dict[str, List[str]],
    output_dir: Path,
    progress: Optional['tqdm'] = None,
    resource_sample_every: int = 1,
    resource_manager: Optional['ResourceManager'] = None,
    batch_processor: Optional['BatchProcessor'] = None,
) -> Dict[str, Any]:
    """Run speed test across all prompts for a model."""
    resource_manager = resource_manager or ResourceManager()
    batch_processor = batch_processor or BatchProcessor()

    results: List[Dict[str, Any]] = []
    total_tokens = 0
    total_duration = 0.0
    timings: List[float] = []
    profiling_records: List[Dict[str, Any]] = []

    batches = batch_processor.generate_batches(prompts)
    tests: List[Tuple[str, str]] = []
    for batch in batches:
        tests.extend(batch)

    for index, (difficulty, prompt) in enumerate(tests):
        concurrency_hint = resource_manager.get_optimal_concurrency()
        interval = resource_manager.collection_interval(resource_sample_every, concurrency_hint)
        collect_resources = interval > 0 and (index % interval == 0)
        start_time = time.time()
        outcome = test_model_with_prompt(
            base_url,
            model_name,
            prompt,
            collect_resources=collect_resources,
        )
        wall_duration = time.time() - start_time
        if math.isfinite(wall_duration):
            timings.append(wall_duration)

        if outcome["success"]:
            total_tokens += outcome["output_tokens"]
            total_duration += outcome["duration"]
            results.append(
                {
                    "model": model_name,
                    "difficulty": difficulty,
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

        timing_info = outcome.get("timings") or {}
        profiling_records.append(
            {
                "model": model_name,
                "difficulty": difficulty,
                "prompt": prompt,
                "success": outcome["success"],
                "error": outcome.get("error", ""),
                "resource_before": timing_info.get("resource_before", math.nan),
                "request": timing_info.get("request", outcome.get("duration", 0.0)),
                "resource_after": timing_info.get("resource_after", math.nan),
                "generation_duration": outcome.get("duration", 0.0),
                "wall_duration": timing_info.get("wall", wall_duration),
                "tokens": outcome.get("output_tokens", 0),
                "tokens_per_second": outcome.get("tokens_per_second", 0.0),
                "concurrency_hint": concurrency_hint,
            }
        )

        gpu_metrics = extract_gpu_metrics(outcome.get("gpu_util_after"))
        resource_manager.update_resource_history(
            {
                "gpu_util": gpu_metrics.get("gpu_util"),
                "cpu_util": safe_float(outcome.get("cpu_util_after")),
                "duration": outcome.get("duration"),
            }
        )

    avg_speed = total_tokens / total_duration if total_duration > 0 else 0.0
    return {
        "model": model_name,
        "total_tokens": total_tokens,
        "total_duration": total_duration,
        "average_speed": avg_speed,
        "test_count": len(results),
        "details": results,
        "profiling": profiling_records,
    }


def export_results_to_csv(results: List[Dict[str, Any]], output_file: Path) -> None:
    """Save detailed prompt-level results to CSV."""
    if not results:
        print("No results to export")
        return
    rows: List[Dict[str, Any]] = []
    for model_result in results:
        for detail in model_result["details"]:
            rows.append(
                {
                    "model": detail["model"],
                    "difficulty": detail["difficulty"],
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
    """Save per-model summary to CSV and print to console."""
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

def export_profiling_data(profiling: List[Dict[str, Any]], output_file: Path) -> None:
    """Export profiling metrics to CSV."""
    if not profiling:
        print("No profiling data collected.")
        return
    frame = pd.DataFrame(profiling)
    frame.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Profiling data exported to: {output_file}")

def print_profiling_summary(profiling: List[Dict[str, Any]]) -> None:
    """Display aggregate profiling statistics."""
    if not profiling:
        return
    frame = pd.DataFrame(profiling)
    numeric_columns = [
        column
        for column in [
            "resource_before",
            "request",
            "resource_after",
            "generation_duration",
            "wall_duration",
        ]
        if column in frame.columns
    ]
    if numeric_columns:
        print("\nProfiling averages (seconds):")
        averages = frame[numeric_columns].mean()
        for column, value in averages.items():
            print(f"  - {column}: {value:.4f}")
    if "success" in frame.columns and len(frame):
        success_rate = float(frame["success"].mean())
        print(f"  - success_rate: {success_rate:.3f}")

def get_available_models(base_url: str) -> List[str]:
    """Fetch model list from Ollama service."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        response.raise_for_status()
        payload = response.json()
        models: List[str] = []
        for item in payload.get("models", []):
            name = item["name"]
            if not any(word in name.lower() for word in ("embedding", "bge", "bert")):
                models.append(name)
        return models
    except Exception as exc:
        print(f"Error getting models: {exc}")
        return []


def safe_float(value: Any) -> float:
    """Convert arbitrary values to float."""
    try:
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        return float(str(value))
    except Exception:
        return 0.0


def extract_gpu_util(snapshot: Any) -> float:
    """Pull GPU utilization from collected snapshot."""
    if not snapshot:
        return 0.0
    if isinstance(snapshot, dict):
        if "gpus" in snapshot and snapshot["gpus"]:
            return safe_float(snapshot["gpus"][0].get("gpu_util"))
        return safe_float(snapshot.get("gpu_util"))
    return safe_float(snapshot)


def extract_gpu_metrics(snapshot: Any) -> Dict[str, float]:
    """Return GPU metrics (utilization, memory) from a snapshot."""
    metrics = {"gpu_util": 0.0, "memory_used": 0.0}
    if not snapshot:
        return metrics
    if isinstance(snapshot, dict):
        if "gpus" in snapshot and snapshot["gpus"]:
            info = snapshot["gpus"][0]
            metrics["gpu_util"] = safe_float(info.get("gpu_util"))
            metrics["memory_used"] = safe_float(info.get("memory_used"))
            return metrics
        metrics["gpu_util"] = safe_float(snapshot.get("gpu_util"))
        metrics["memory_used"] = safe_float(snapshot.get("memory_used"))
        return metrics
    metrics["gpu_util"] = extract_gpu_util(snapshot)
    return metrics


def normalize(series: pd.Series) -> pd.Series:
    """Min-max normalize numeric series."""
    if series.empty:
        return series
    filled = series.fillna(0.0)
    min_value = filled.min()
    max_value = filled.max()
    if max_value - min_value == 0:
        return pd.Series(1.0, index=series.index)
    return (filled - min_value) / (max_value - min_value)


def normalize_inverse(series: pd.Series) -> pd.Series:
    """Normalize where smaller values are better."""
    normalized = normalize(series)
    return 1.0 - normalized


def prepare_detail_dataframe(results: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
    """Flatten nested test results into a detail dataframe."""
    rows: List[Dict[str, Any]] = []
    for model_result in results:
        model = model_result.get("model", "")
        for detail in model_result.get("details", []):
            rows.append(
                {
                    "model": model,
                    "difficulty": detail.get("difficulty", ""),
                    "tokens_per_second": safe_float(detail.get("tokens_per_second")),
                    "duration": safe_float(detail.get("duration")),
                    "output_tokens": safe_float(detail.get("output_tokens")),
                    "cpu_util_before": safe_float(detail.get("cpu_util_before")),
                    "cpu_util_after": safe_float(detail.get("cpu_util_after")),
                    "gpu_util_after": extract_gpu_util(detail.get("gpu_util_after")),
                }
            )
    if not rows:
        return None
    detail_df = pd.DataFrame(rows)
    return detail_df


def compute_analysis_metrics(
    detail_df: pd.DataFrame,
    results: List[Dict[str, Any]],
    weights: Dict[str, float],
) -> pd.DataFrame:
    """Aggregate metrics per model and compute composite score."""
    summary_records: List[Dict[str, Any]] = []
    for model_result in results:
        summary_records.append(
            {
                "model": model_result["model"],
                "total_tokens": safe_float(model_result.get("total_tokens")),
                "total_duration": safe_float(model_result.get("total_duration")),
                "average_speed": safe_float(model_result.get("average_speed")),
                "test_count": model_result.get("test_count", 0),
            }
        )
    summary_df = pd.DataFrame(summary_records).set_index("model")

    grouped = detail_df.groupby("model")
    avg_speed = grouped["tokens_per_second"].mean()
    speed_std = grouped["tokens_per_second"].std().fillna(0.0)
    avg_gpu = grouped["gpu_util_after"].mean().fillna(0.0)
    avg_cpu = grouped["cpu_util_after"].mean().fillna(0.0)

    medium_speed = (
        detail_df[detail_df["difficulty"] == "medium"]
        .groupby("model")["tokens_per_second"]
        .mean()
        .reindex(avg_speed.index, fill_value=0.0)
    )

    metrics = pd.DataFrame(
        {
            "model": avg_speed.index,
            "avg_speed": avg_speed.values,
            "avg_gpu_util": avg_gpu.reindex(avg_speed.index).values,
            "avg_cpu_util": avg_cpu.reindex(avg_speed.index).values,
            "speed_std": speed_std.reindex(avg_speed.index).values,
            "medium_speed": medium_speed.values,
        }
    ).set_index("model")
    metrics = metrics.join(summary_df, how="left")
    metrics = metrics.fillna(0.0)

    scores = pd.DataFrame(index=metrics.index)
    scores["speed_score"] = normalize(metrics["avg_speed"])
    scores["efficiency_score"] = normalize(metrics["avg_speed"] / (1.0 + metrics["avg_gpu_util"]))
    scores["stability_score"] = normalize_inverse(metrics["speed_std"])

    normalized_weights = normalize_weights(weights)
    scores["overall_score"] = (
        scores["speed_score"] * normalized_weights.get("speed", DEFAULT_WEIGHTS["speed"])
        + scores["efficiency_score"] * normalized_weights.get("efficiency", DEFAULT_WEIGHTS["efficiency"])
        + scores["stability_score"] * normalized_weights.get("stability", DEFAULT_WEIGHTS["stability"])
    )

    combined = metrics.join(scores)
    combined = combined.reset_index()
    combined = combined.sort_values("overall_score", ascending=False).reset_index(drop=True)
    return combined


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Normalize custom weight dictionary."""
    positives = {key: max(0.0, float(value)) for key, value in weights.items()}
    total = sum(positives.values())
    if total <= 0:
        return DEFAULT_WEIGHTS
    return {key: value / total for key, value in positives.items()}


def prepare_plotting() -> bool:
    """Ensure matplotlib is available."""
    global plt, sns
    if plt is not None:
        return True
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "seaborn"])
        import matplotlib.pyplot as plt_module
        plt = plt_module
        import seaborn as sns_module
        sns = sns_module
        warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
        plt.rcParams["font.sans-serif"] = PLOT_FONTS
        plt.rcParams["axes.unicode_minus"] = False
        sns.set_theme(style="whitegrid")
        return True
    except Exception as exc:  # pragma: no cover
        print(f"[Warning] Unable to prepare plotting backend: {exc}")
        return False


def plot_speed_chart(metrics: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    """Create grouped bar chart for per-difficulty speed."""
    if not prepare_plotting():
        return None
    models = metrics["model"].tolist()
    if not models:
        return None
    speeds = metrics["medium_speed"].tolist()
    figure, axis = plt.subplots(figsize=(10, 6))
    x_positions = list(range(len(models)))
    axis.bar(x_positions, speeds, color="#4F81BD")
    axis.set_xlabel("模型")
    axis.set_ylabel("速度 (tokens/s)")
    axis.set_title("模型速度对比（Medium 问题）", fontweight="bold")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(models, rotation=30)
    figure.tight_layout()
    output_path = output_dir / "chart_speed_comparison.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_resource_chart(metrics: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    """Create resource utilization bar chart."""
    if not prepare_plotting():
        return None
    if metrics["avg_gpu_util"].sum() == 0 and metrics["avg_cpu_util"].sum() == 0:
        return None
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.bar(metrics["model"], metrics["avg_gpu_util"], label="GPU Util (%)", color="#FF6B6B")
    axis.bar(metrics["model"], metrics["avg_cpu_util"], bottom=metrics["avg_gpu_util"], label="CPU Util (%)", color="#4ECDC4", alpha=0.7)
    axis.set_ylabel("利用率 (%)")
    axis.set_title("资源使用情况", fontweight="bold")
    axis.tick_params(axis="x", rotation=30)
    axis.legend()
    figure.tight_layout()
    output_path = output_dir / "chart_resource_usage.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def dataframe_to_markdown(frame: pd.DataFrame, columns: List[str]) -> str:
    """Convert dataframe slice to markdown table string."""
    subset = frame[columns].copy()
    formatted = subset.round({col: 3 for col in columns if subset[col].dtype.kind in "fc"})
    headers = " | ".join(columns)
    separator = " | ".join(["---"] * len(columns))
    lines = [f"| {headers} |", f"| {separator} |"]
    for _, row in formatted.iterrows():
        line = " | ".join(str(row[col]) for col in columns)
        lines.append(f"| {line} |")
    return "\n".join(lines)


def generate_report(metrics: pd.DataFrame, charts: Dict[str, Optional[Path]], output_dir: Path) -> Path:
    """Create markdown report summarizing analysis."""
    top_models = metrics.head(3)
    lines = [
        "# Speed Test Analysis Report",
        "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "## Top Models",
    ]
    for index, row in top_models.iterrows():
        lines.append(
            f"{index + 1}. {row['model']} (Overall Score: {row['overall_score']:.3f}, Avg Speed: {row['avg_speed']:.1f} tok/s)"
        )
    lines.extend(["", "## Performance Overview", ""])
    lines.append(dataframe_to_markdown(metrics, [
        "model",
        "overall_score",
        "avg_speed",
        "avg_gpu_util",
        "avg_cpu_util",
        "test_count",
    ]))
    lines.extend(["", "## Medium 问题速度", ""])
    lines.append(dataframe_to_markdown(metrics, ["model", "medium_speed"]))
    lines.extend(["", "## Charts", ""])
    for label, path in charts.items():
        if path:
            lines.append(f"- {label}: ./{path.name}")
    report_path = output_dir / "analysis_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_analysis(
    results: List[Dict[str, Any]],
    analysis_dir: Path,
    weights: Dict[str, float],
) -> None:
    """Perform visualization and reporting based on speed test results."""
    detail_df = prepare_detail_dataframe(results)
    if detail_df is None:
        print("Skipping analysis: no detailed records available.")
        return
    metrics = compute_analysis_metrics(detail_df, results, weights)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = analysis_dir / "performance_metrics.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8")
    print(f"Performance metrics saved to: {metrics_path}")
    charts = {
        "Speed Comparison": plot_speed_chart(metrics, analysis_dir),
        "Resource Usage": plot_resource_chart(metrics, analysis_dir),
    }
    generated = [name for name, path in charts.items() if path]
    if generated:
        print(f"Generated charts: {', '.join(generated)}")
    else:
        print("No charts generated (plotting backend unavailable).")
    report_path = generate_report(metrics, charts, analysis_dir)
    print(f"Analysis report generated: {report_path}")
    print("\n=== Top 3 Models ===")
    for rank, row in metrics.head(3).iterrows():
        print(f"  {rank + 1}. {row['model']} (Overall Score: {row['overall_score']:.3f})")


def parse_weights(raw: Optional[str]) -> Dict[str, float]:
    """Parse custom weight configuration."""
    if not raw:
        return DEFAULT_WEIGHTS
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Weights must be a JSON object")
        return normalize_weights({str(key): float(value) for key, value in data.items()})
    except Exception as exc:
        print(f"[Warning] Invalid weights configuration: {exc}. Using defaults.")
        return DEFAULT_WEIGHTS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Model speed test tool with resource monitoring and integrated analysis",
    )
    parser.add_argument("--base-url", default="http://localhost:11434", help="Base URL of Ollama service")
    parser.add_argument("--prompts-file", default="test_prompts.txt", help="Path to test prompts file")
    parser.add_argument("--model", help="Specific model to test (test all if not specified)")
    parser.add_argument("--output-dir", default="speed_test_results", help="Output directory for raw results")
    parser.add_argument("--analysis-dir", default="analysis_results", help="Directory for analysis outputs")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip visualization and reporting")
    parser.add_argument("--weights", help="Custom analysis weights JSON string")
    parser.add_argument(
        "--resource-sample-every",
        type=int,
        default=1,
        help="Collect resource metrics every N prompts (0 disables resource sampling)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=4,
        help="Upper bound for adaptive concurrency hints",
    )
    parser.add_argument(
        "--min-concurrency",
        type=int,
        default=1,
        help="Lower bound for adaptive concurrency hints",
    )
    parser.add_argument(
        "--gpu-threshold",
        type=float,
        default=0.8,
        help="GPU utilization threshold for downscaling concurrency",
    )
    parser.add_argument(
        "--cpu-threshold",
        type=float,
        default=0.7,
        help="CPU utilization threshold for downscaling concurrency",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=6,
        help="Maximum number of prompts per execution batch",
    )
    parser.add_argument(
        "--min-batch-size",
        type=int,
        default=1,
        help="Minimum number of prompts per execution batch",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("Loading test prompts...")
    prompts = load_test_prompts(args.prompts_file)
    total_prompts = sum(len(items) for items in prompts.values())
    print(f"Loaded {total_prompts} test prompts:")
    for difficulty, items in prompts.items():
        print(f"  - {difficulty}: {len(items)} prompts")

    if args.model:
        models = [args.model]
    else:
        print("\nDiscovering available models...")
        models = get_available_models(args.base_url)
        print(f"Found {len(models)} models: {', '.join(models)}")

    if not models:
        print("No models to test")
        return

    total_tests = len(models) * total_prompts
    progress_bar = tqdm(
        total=total_tests,
        desc="Initializing speed test...",
        unit="test",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    all_results: List[Dict[str, Any]] = []
    profiling_rows: List[Dict[str, Any]] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        for model in models:
            progress_bar.set_description(f"Testing model: {model}")
            model_resource_manager = ResourceManager(
                gpu_threshold=args.gpu_threshold,
                cpu_threshold=args.cpu_threshold,
                min_concurrency=args.min_concurrency,
                max_concurrency=args.max_concurrency,
            )
            model_batch_processor = BatchProcessor(
                max_batch_size=args.max_batch_size,
                min_batch_size=args.min_batch_size,
            )
            result = run_speed_test(
                args.base_url,
                model,
                prompts,
                output_dir,
                progress_bar,
                resource_sample_every=args.resource_sample_every,
                resource_manager=model_resource_manager,
                batch_processor=model_batch_processor,
            )
            profiling_rows.extend(result.pop("profiling", []))
            all_results.append(result)
    finally:
        progress_bar.close()

    detail_file = output_dir / f"speed_test_details_{timestamp}.csv"
    summary_file = output_dir / f"speed_test_summary_{timestamp}.csv"
    profiling_file = output_dir / f"speed_test_profiling_{timestamp}.csv"
    json_file = output_dir / f"speed_test_results_{timestamp}.json"

    export_results_to_csv(all_results, detail_file)
    generate_summary_table(all_results, summary_file)
    export_profiling_data(profiling_rows, profiling_file)

    with open(json_file, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "timestamp": timestamp,
                "total_models": len(models),
                "results": all_results,
                "profiling": profiling_rows,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nAll results saved to: {output_dir}/")
    print(f"  - Detailed CSV: {detail_file.name}")
    print(f"  - Summary CSV: {summary_file.name}")
    print(f"  - JSON data: {json_file.name}")
    if profiling_rows:
        print(f"  - Profiling CSV: {profiling_file.name}")
    print_profiling_summary(profiling_rows)

    if args.skip_analysis:
        print("Analysis skipped by user request.")
        return

    analysis_dir = Path(args.analysis_dir)
    weights = parse_weights(args.weights)
    run_analysis(all_results, analysis_dir, weights)




if __name__ == '__main__':
    main()

