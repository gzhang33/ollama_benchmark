#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization Module for Speed Test Analysis
Speed test analysis visualization module
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# 尝试导入可视化库
try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

try:
    import seaborn as sns
except ImportError:  # pragma: no cover
    sns = None

# Font configuration - using default English font configuration
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
FONT_CONFIGURED = True

# Configure seaborn
if sns is not None:
    sns.set_theme(style="whitegrid")


class VisualizationManager:
    """Visualization manager responsible for handling all chart generation and report creation"""
    
    def __init__(self):
        self.plt = plt
        self.sns = sns
        self.plotting_available = self._prepare_plotting()
    
    def _prepare_plotting(self) -> bool:
        """Ensure matplotlib is available"""
        if self.plt is not None:
            return True
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "seaborn"])
            import matplotlib.pyplot as plt_module
            self.plt = plt_module
            import seaborn as sns_module
            self.sns = sns_module
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
            # Reconfigure fonts for English display
            self.plt.rcParams["font.family"] = "sans-serif"
            self.plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]
            self.plt.rcParams["axes.unicode_minus"] = False
            self.sns.set_theme(style="whitegrid")
            return True
        except Exception as exc:  # pragma: no cover
            print(f"[Warning] Unable to prepare plotting backend: {exc}")
            return False
    
    def plot_speed_chart(self, metrics: pd.DataFrame, output_dir: Path) -> Optional[Path]:
        """Create model speed comparison bar chart"""
        if not self.plotting_available:
            return None
        models = metrics["model"].tolist()
        if not models:
            return None
        speeds = metrics["avg_speed"].tolist()
        figure, axis = self.plt.subplots(figsize=(10, 6))
        x_positions = list(range(len(models)))
        axis.bar(x_positions, speeds, color="#4F81BD")
        axis.set_xlabel("Model")
        axis.set_ylabel("Speed (tokens/s)")
        axis.set_title("Model Speed Comparison", fontweight="bold")
        axis.set_xticks(x_positions)
        axis.set_xticklabels(models, rotation=30)
        figure.tight_layout()
        output_path = output_dir / "chart_speed_comparison.png"
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
        self.plt.close(figure)
        return output_path
    
    def plot_resource_chart(self, metrics: pd.DataFrame, output_dir: Path) -> Optional[Path]:
        """Create resource usage bar chart"""
        if not self.plotting_available:
            return None
        if metrics["avg_gpu_util"].sum() == 0 and metrics["avg_cpu_util"].sum() == 0:
            return None
        figure, axis = self.plt.subplots(figsize=(12, 6))
        axis.bar(metrics["model"], metrics["avg_gpu_util"], label="GPU Util (%)", color="#FF6B6B")
        axis.bar(metrics["model"], metrics["avg_cpu_util"], 
                bottom=metrics["avg_gpu_util"], label="CPU Util (%)", color="#4ECDC4", alpha=0.7)
        axis.set_ylabel("Utilization (%)")
        axis.set_title("Resource Usage", fontweight="bold")
        axis.tick_params(axis="x", rotation=30)
        axis.legend()
        figure.tight_layout()
        output_path = output_dir / "chart_resource_usage.png"
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
        self.plt.close(figure)
        return output_path
    
    def plot_efficiency_chart(self, metrics: pd.DataFrame, output_dir: Path) -> Optional[Path]:
        """Create efficiency comparison chart (speed/resource usage)"""
        if not self.plotting_available:
            return None
        figure, axis = self.plt.subplots(figsize=(10, 6))
        
        # Calculate efficiency metric (speed divided by resource usage)
        total_resource_util = metrics["avg_gpu_util"] + metrics["avg_cpu_util"]
        efficiency = metrics["avg_speed"] / (total_resource_util + 1)  # Avoid division by zero
        
        axis.bar(metrics["model"], efficiency, color="#9B59B6")
        axis.set_xlabel("Model")
        axis.set_ylabel("Efficiency Score")
        axis.set_title("Model Efficiency Comparison (Speed/Resource Usage)", fontweight="bold")
        axis.tick_params(axis="x", rotation=30)
        figure.tight_layout()
        output_path = output_dir / "chart_efficiency.png"
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
        self.plt.close(figure)
        return output_path


class ReportGenerator:
    """Report generator responsible for creating analysis reports"""
    
    def dataframe_to_markdown(self, frame: pd.DataFrame, columns: List[str]) -> str:
        """Convert DataFrame to Markdown table"""
        subset = frame[columns].copy()
        formatted = subset.round({col: 3 for col in columns if subset[col].dtype.kind in "fc"})
        headers = " | ".join(columns)
        separator = " | ".join(["---"] * len(columns))
        lines = [f"| {headers} |", f"| {separator} |"]
        for _, row in formatted.iterrows():
            line = " | ".join(str(row[col]) for col in columns)
            lines.append(f"| {line} |")
        return "\n".join(lines)
    
    def generate_report(self, metrics: pd.DataFrame, charts: Dict[str, Optional[Path]], output_dir: Path) -> Path:
        """Generate Markdown format analysis report"""
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
                f"{index + 1}. {row['model']} (Overall Score: {row['overall_score']:.3f}, "
                f"Avg Speed: {row['avg_speed']:.1f} tok/s)"
            )
        lines.extend(["", "## Performance Overview", ""])
        lines.append(self.dataframe_to_markdown(metrics, [
            "model",
            "overall_score",
            "avg_speed",
            "avg_gpu_util",
            "avg_cpu_util",
            "success_rate",
            "test_count",
        ]))
        lines.extend(["", "## Charts", ""])
        for label, path in charts.items():
            if path:
                lines.append(f"- {label}: ./{path.name}")
        report_path = output_dir / "analysis_report.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path


class AnalysisVisualizer:
    """Analysis visualizer integrating all visualization functionality"""
    
    def __init__(self):
        self.visualizer = VisualizationManager()
        self.report_generator = ReportGenerator()
    
    def run_analysis(self, results: List[Dict[str, Any]], analysis_dir: Path) -> None:
        """Execute complete analysis and visualization"""
        # Prepare data
        detail_df = self._prepare_detail_dataframe(results)
        if detail_df is None:
            print("Skipping analysis: no detailed records available.")
            return
        
        # Compute performance metrics
        metrics = self._compute_analysis_metrics(detail_df, results)
        
        # Create output directory
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Save performance metrics
        metrics_path = analysis_dir / "performance_metrics.csv"
        metrics.to_csv(metrics_path, index=False, encoding="utf-8")
        print(f"Performance metrics saved to: {metrics_path}")
        
        # Generate charts
        charts = {
            "Speed Comparison": self.visualizer.plot_speed_chart(metrics, analysis_dir),
            "Resource Usage": self.visualizer.plot_resource_chart(metrics, analysis_dir),
            "Efficiency": self.visualizer.plot_efficiency_chart(metrics, analysis_dir),
        }
        
        # Report generated charts
        generated = [name for name, path in charts.items() if path]
        if generated:
            print(f"Generated charts: {', '.join(generated)}")
        else:
            print("No charts generated (plotting backend unavailable).")
        
        # Generate report
        report_path = self.report_generator.generate_report(metrics, charts, analysis_dir)
        print(f"Analysis report generated: {report_path}")
        
        # Show top 3 models
        print("\n=== Top 3 Models ===")
        for rank, row in metrics.head(3).iterrows():
            print(f"  {rank + 1}. {row['model']} (Overall Score: {row['overall_score']:.3f})")
    
    def _prepare_detail_dataframe(self, results: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
        """Prepare detailed data DataFrame"""
        rows: List[Dict[str, Any]] = []
        for model_result in results:
            model = model_result.get("model", "")
            for detail in model_result.get("details", []):
                rows.append(
                    {
                        "model": model,
                        "tokens_per_second": self._safe_float(detail.get("tokens_per_second")),
                        "duration": self._safe_float(detail.get("duration")),
                        "output_tokens": self._safe_float(detail.get("output_tokens")),
                        "cpu_util_before": self._safe_float(detail.get("cpu_util_before")),
                        "cpu_util_after": self._safe_float(detail.get("cpu_util_after")),
                        "gpu_util_after": self._extract_gpu_util(detail.get("gpu_util_after")),
                    }
                )
        if not rows:
            return None
        return pd.DataFrame(rows)
    
    def _compute_analysis_metrics(self, detail_df: pd.DataFrame, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Compute analysis metrics"""
        # Prepare summary records
        summary_records: List[Dict[str, Any]] = []
        for model_result in results:
            summary_records.append(
                {
                    "model": model_result["model"],
                    "total_tokens": self._safe_float(model_result.get("total_tokens")),
                    "total_duration": self._safe_float(model_result.get("total_duration")),
                    "average_speed": self._safe_float(model_result.get("average_speed")),
                    "success_rate": self._safe_float(model_result.get("success_rate")),
                    "test_count": model_result.get("test_count", 0),
                }
            )
        summary_df = pd.DataFrame(summary_records).set_index("model")
        
        # 计算分组统计
        grouped = detail_df.groupby("model")
        avg_speed = grouped["tokens_per_second"].mean()
        speed_std = grouped["tokens_per_second"].std().fillna(0.0)
        avg_gpu = grouped["gpu_util_after"].mean().fillna(0.0)
        avg_cpu = grouped["cpu_util_after"].mean().fillna(0.0)
        
        # Create metrics DataFrame
        metrics = pd.DataFrame(
            {
                "model": avg_speed.index,
                "avg_speed": avg_speed.values,
                "avg_gpu_util": avg_gpu.reindex(avg_speed.index).values,
                "avg_cpu_util": avg_cpu.reindex(avg_speed.index).values,
                "speed_std": speed_std.reindex(avg_speed.index).values,
            }
        ).set_index("model")
        metrics = metrics.join(summary_df, how="left")
        metrics = metrics.fillna(0.0)
        
        # 计算评分
        metrics["speed_score"] = metrics["avg_speed"] / metrics["avg_speed"].max()
        metrics["success_score"] = metrics["success_rate"]
        metrics["overall_score"] = (metrics["speed_score"] * 0.7 + metrics["success_score"] * 0.3)
        
        # 排序并返回
        combined = metrics.reset_index()
        combined = combined.sort_values("overall_score", ascending=False).reset_index(drop=True)
        return combined
    
    def _safe_float(self, value: Any) -> float:
        """安全转换为float"""
        try:
            if value is None:
                return 0.0
            if isinstance(value, (int, float)):
                return float(value)
            return float(str(value))
        except Exception:
            return 0.0
    
    def _extract_gpu_util(self, snapshot: Any) -> float:
        """提取GPU使用率"""
        if not snapshot:
            return 0.0
        if isinstance(snapshot, dict):
            if "gpus" in snapshot and snapshot["gpus"]:
                return self._safe_float(snapshot["gpus"][0].get("gpu_util"))
            return self._safe_float(snapshot.get("gpu_util"))
        return self._safe_float(snapshot)


# Convenience functions
def create_visualizer() -> AnalysisVisualizer:
    """Create analysis visualizer instance"""
    return AnalysisVisualizer()


def run_visualization_analysis(results: List[Dict[str, Any]], analysis_dir: Path) -> None:
    """Convenience function to run visualization analysis"""
    visualizer = create_visualizer()
    visualizer.run_analysis(results, analysis_dir)


# Module exports
__all__ = [
    "VisualizationManager",
    "ReportGenerator", 
    "AnalysisVisualizer",
    "create_visualizer",
    "run_visualization_analysis"
]
