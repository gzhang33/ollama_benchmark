#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Speed Test Results Analyzer
速度测试结果分析器 - 专门分析 speed_test.py 的输出结果
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings


def check_dependencies() -> None:
    """Verify required third-party packages are available."""
    missing = []
    try:
        import pandas  # noqa: F401
    except ImportError:
        missing.append("pandas")
    try:
        import numpy  # noqa: F401
    except ImportError:
        missing.append("numpy")
    try:
        import matplotlib.pyplot  # noqa: F401
    except ImportError:
        missing.append("matplotlib")
    try:
        import seaborn  # noqa: F401
    except ImportError:
        print("Seaborn not available, installing...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])
            print("Seaborn installed successfully")
        except subprocess.CalledProcessError as exc:
            print(f"Failed to install seaborn: {exc}")
            print("Seaborn not available, charts will use matplotlib fallback")

    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Attempting installation...")
        import subprocess

        for package in missing:
            print(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            except subprocess.CalledProcessError as exc:
                print(f"Failed to install {package}: {exc}")
                print("Install required packages manually: pip install pandas numpy matplotlib seaborn tabulate")
                sys.exit(1)

    try:
        import tabulate  # noqa: F401
    except ImportError:
        print("Installing tabulate...")
        import subprocess

        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tabulate"])
        except subprocess.CalledProcessError as exc:
            print(f"Failed to install tabulate: {exc}")
            print("Tabulate not available, tables will use fallback formatting.")


check_dependencies()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tabulate import tabulate

# 尝试导入seaborn并设置主题
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", palette="husl")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Speed test 专用配置
DIFFICULTY_LEVELS = ["simple", "medium", "complex"]
DEFAULT_WEIGHTS = {"speed": 0.5, "efficiency": 0.3, "stability": 0.2}

# 性能指标权重
PERFORMANCE_WEIGHTS = {
    "speed": 0.4,      # 生成速度
    "efficiency": 0.3, # 效率（速度/资源使用）
    "stability": 0.2,  # 稳定性（速度方差）
    "resource": 0.1    # 资源使用效率
}


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Return weights normalized to sum to 1.0."""
    total = sum(value for value in weights.values() if value > 0)
    if total <= 0:
        return DEFAULT_WEIGHTS
    return {key: max(value, 0.0) / total for key, value in weights.items()}


def normalize_positive(series: pd.Series) -> pd.Series:
    """Normalize values by dividing by the max positive value."""
    series = series.astype(float)
    valid = series.replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return pd.Series(np.zeros(len(series)), index=series.index)
    peak = valid.max()
    if np.isclose(peak, 0.0):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return series.fillna(0.0) / peak


def normalize_inverse(series: pd.Series) -> pd.Series:
    """Normalize where lower values are better."""
    normalized = normalize_positive(series)
    inverted = 1.0 - normalized
    return inverted.clip(lower=0.0, upper=1.0)


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


class SpeedTestAnalyzer:
    """专门分析 speed_test.py 输出结果的分析器"""

    def __init__(
        self,
        input_path: Path,
        output_dir: Path,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.input_path = input_path
        self.output_dir = output_dir
        self.weights = normalize_weights(weights or DEFAULT_WEIGHTS)
        self.df: pd.DataFrame = pd.DataFrame()
        self.summary_df: pd.DataFrame = pd.DataFrame()

    def detect_input_format(self) -> str:
        """检测输入文件格式"""
        suffix = self.input_path.suffix.lower()
        if suffix in {".json", ".csv"}:
            return suffix.lstrip(".")
        with self.input_path.open("r", encoding="utf-8") as handle:
            sample = handle.read(2048).strip()
        if sample.startswith("{") or sample.startswith("["):
            return "json"
        if "," in sample:
            return "csv"
        raise ValueError("Unsupported input format. Provide JSON or CSV generated by speed_test.py.")

    def load_json_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载 speed_test.py 的 JSON 数据"""
        with self.input_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        results = payload.get("results", [])
        if not isinstance(results, list):
            raise ValueError("Invalid speed test JSON: missing 'results' list.")

        # 处理详细测试数据
        detail_rows: List[Dict[str, Any]] = []
        summary_rows: List[Dict[str, Any]] = []
        
        for model_result in results:
            model_name = model_result.get("model", "")
            total_tokens = model_result.get("total_tokens", 0)
            total_duration = model_result.get("total_duration", 0)
            average_speed = model_result.get("average_speed", 0)
            test_count = model_result.get("test_count", 0)
            
            # 汇总数据
            summary_rows.append({
                "model": model_name,
                "total_tokens": total_tokens,
                "total_duration": total_duration,
                "average_speed": average_speed,
                "test_count": test_count
            })
            
            # 详细数据
            details = model_result.get("details", [])
            for detail in details:
                # 提取 GPU 使用情况
                gpu_before = detail.get("gpu_util_before", {})
                gpu_after = detail.get("gpu_util_after", {})
                gpu_before_util = 0
                gpu_after_util = 0
                gpu_memory_used = 0
                
                if gpu_before and "gpus" in gpu_before:
                    gpu_before_util = gpu_before["gpus"][0].get("gpu_util", 0)
                if gpu_after and "gpus" in gpu_after:
                    gpu_after_util = gpu_after["gpus"][0].get("gpu_util", 0)
                    gpu_memory_used = gpu_after["gpus"][0].get("memory_used", 0)
                
                detail_rows.append({
                    "model": model_name,
                    "difficulty": detail.get("difficulty", ""),
                    "prompt": detail.get("prompt", ""),
                    "output_tokens": detail.get("output_tokens", 0),
                    "duration": detail.get("duration", 0),
                    "tokens_per_second": detail.get("tokens_per_second", 0),
                    "cpu_util_before": detail.get("cpu_util_before", 0),
                    "cpu_util_after": detail.get("cpu_util_after", 0),
                    "gpu_util_before": gpu_before_util,
                    "gpu_util_after": gpu_after_util,
                    "gpu_memory_used": gpu_memory_used,
                    "response_preview": detail.get("response_preview", "")
                })

        detail_df = pd.DataFrame(detail_rows)
        summary_df = pd.DataFrame(summary_rows)
        
        if detail_df.empty:
            raise ValueError("No detail entries found in speed test JSON.")
        if summary_df.empty:
            raise ValueError("No summary entries found in speed test JSON.")
            
        return detail_df, summary_df

    def load_csv_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载 CSV 数据"""
        df = pd.read_csv(self.input_path)
        
        # 检查是否为汇总 CSV
        if "difficulty" in df.columns and "OVERALL" in df["difficulty"].values:
            # 这是汇总 CSV
            summary_df = df[df["difficulty"] == "OVERALL"].copy()
            detail_df = df[df["difficulty"] != "OVERALL"].copy()
        else:
            # 假设这是详细 CSV
            detail_df = df.copy()
            # 创建汇总数据
            summary_data = []
            for model in df["model"].unique():
                model_data = df[df["model"] == model]
                summary_data.append({
                    "model": model,
                    "total_tokens": model_data["output_tokens"].sum(),
                    "total_duration": model_data["total_duration"].astype(float).sum(),
                    "average_speed": model_data["tokens_per_second"].astype(float).mean(),
                    "test_count": len(model_data)
                })
            summary_df = pd.DataFrame(summary_data)
        
        return detail_df, summary_df

    def load_data(self) -> None:
        """加载数据"""
        file_type = self.detect_input_format()
        if file_type == "json":
            self.df, self.summary_df = self.load_json_data()
        elif file_type == "csv":
            self.df, self.summary_df = self.load_csv_data()
        else:
            raise ValueError(f"Unsupported input type: {file_type}")

        # 数据清洗和类型转换
        numeric_cols = ["output_tokens", "duration", "tokens_per_second", 
                       "cpu_util_before", "cpu_util_after", "gpu_util_before", "gpu_util_after"]
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

    def compute_performance_metrics(self) -> pd.DataFrame:
        """计算性能指标"""
        # 按模型分组计算指标
        model_metrics = []
        
        for model in self.summary_df["model"].unique():
            model_detail = self.df[self.df["model"] == model]
            model_summary = self.summary_df[self.summary_df["model"] == model].iloc[0]
            
            # 基础指标
            avg_speed = model_summary["average_speed"]
            total_duration = model_summary["total_duration"]
            total_tokens = model_summary["total_tokens"]
            
            # 按难度级别的性能
            speed_by_difficulty = {}
            for difficulty in DIFFICULTY_LEVELS:
                diff_data = model_detail[model_detail["difficulty"] == difficulty]
                if not diff_data.empty:
                    speed_by_difficulty[difficulty] = diff_data["tokens_per_second"].mean()
                else:
                    speed_by_difficulty[difficulty] = 0
            
            # 稳定性指标（速度方差）
            speed_variance = model_detail["tokens_per_second"].std() / model_detail["tokens_per_second"].mean() if model_detail["tokens_per_second"].mean() > 0 else 0
            
            # 资源使用指标
            avg_cpu_before = model_detail["cpu_util_before"].mean() if "cpu_util_before" in model_detail.columns else 0
            avg_cpu_after = model_detail["cpu_util_after"].mean() if "cpu_util_after" in model_detail.columns else 0
            avg_gpu_util = model_detail["gpu_util_after"].mean() if "gpu_util_after" in model_detail.columns else 0
            
            # 效率指标（速度/资源使用）
            efficiency = avg_speed / max(avg_gpu_util, 1) if avg_gpu_util > 0 else avg_speed
            
            model_metrics.append({
                "model": model,
                "average_speed": avg_speed,
                "total_duration": total_duration,
                "total_tokens": total_tokens,
                "speed_variance": speed_variance,
                "efficiency": efficiency,
                "avg_cpu_util": (avg_cpu_before + avg_cpu_after) / 2,
                "avg_gpu_util": avg_gpu_util,
                "simple_speed": speed_by_difficulty.get("simple", 0),
                "medium_speed": speed_by_difficulty.get("medium", 0),
                "complex_speed": speed_by_difficulty.get("complex", 0),
                "test_count": len(model_detail)
            })
        
        metrics_df = pd.DataFrame(model_metrics)
        
        # 计算标准化分数
        metrics_df["speed_score"] = normalize_positive(metrics_df["average_speed"])
        metrics_df["efficiency_score"] = normalize_positive(metrics_df["efficiency"])
        metrics_df["stability_score"] = normalize_inverse(metrics_df["speed_variance"])
        metrics_df["resource_score"] = normalize_inverse(metrics_df["avg_gpu_util"])
        
        # 计算综合分数
        metrics_df["overall_score"] = (
            PERFORMANCE_WEIGHTS["speed"] * metrics_df["speed_score"] +
            PERFORMANCE_WEIGHTS["efficiency"] * metrics_df["efficiency_score"] +
            PERFORMANCE_WEIGHTS["stability"] * metrics_df["stability_score"] +
            PERFORMANCE_WEIGHTS["resource"] * metrics_df["resource_score"]
        )
        
        return metrics_df.sort_values("overall_score", ascending=False).reset_index(drop=True)

    def generate_speed_comparison_chart(self, df: pd.DataFrame) -> Optional[Path]:
        """生成速度对比图表"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        models = df["model"].tolist()
        simple_speeds = df["simple_speed"].tolist()
        medium_speeds = df["medium_speed"].tolist()
        complex_speeds = df["complex_speed"].tolist()
        
        x = np.arange(len(models))
        width = 0.25
        
        if HAS_SEABORN:
            # 准备数据用于 seaborn
            chart_data = []
            for i, model in enumerate(models):
                chart_data.extend([
                    {"model": model, "difficulty": "Simple", "speed": simple_speeds[i]},
                    {"model": model, "difficulty": "Medium", "speed": medium_speeds[i]},
                    {"model": model, "difficulty": "Complex", "speed": complex_speeds[i]}
                ])
            
            chart_df = pd.DataFrame(chart_data)
            sns.barplot(data=chart_df, x="model", y="speed", hue="difficulty", ax=ax)
            ax.set_title("模型速度对比（按难度级别）", fontsize=14, fontweight='bold')
        else:
            ax.bar(x - width, simple_speeds, width, label='Simple', color='#2E8B57')
            ax.bar(x, medium_speeds, width, label='Medium', color='#FF8C00')
            ax.bar(x + width, complex_speeds, width, label='Complex', color='#DC143C')
            ax.set_title("Model Speed Comparison by Difficulty")
            ax.legend()
        
        ax.set_xlabel("模型", fontsize=12)
        ax.set_ylabel("速度 (tokens/s)", fontsize=12)
        ax.tick_params(axis='x', rotation=30)
        
        fig.tight_layout()
        output_path = self.output_dir / "chart_speed_comparison.png"
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        return output_path

    def generate_efficiency_chart(self, df: pd.DataFrame) -> Optional[Path]:
        """生成效率对比图表"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 效率分数对比
        if HAS_SEABORN:
            sns.barplot(data=df, x="model", y="efficiency_score", ax=ax1, palette="viridis")
            sns.barplot(data=df, x="model", y="speed_score", ax=ax2, palette="plasma")
        else:
            ax1.bar(df["model"], df["efficiency_score"], color='#4F81BD')
            ax2.bar(df["model"], df["speed_score"], color='#9BBB59')
        
        ax1.set_title("效率分数对比", fontsize=12, fontweight='bold')
        ax1.set_xlabel("模型", fontsize=10)
        ax1.set_ylabel("效率分数", fontsize=10)
        ax1.tick_params(axis='x', rotation=30)
        
        ax2.set_title("速度分数对比", fontsize=12, fontweight='bold')
        ax2.set_xlabel("模型", fontsize=10)
        ax2.set_ylabel("速度分数", fontsize=10)
        ax2.tick_params(axis='x', rotation=30)
        
        fig.tight_layout()
        output_path = self.output_dir / "chart_efficiency.png"
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        return output_path

    def generate_resource_usage_chart(self, df: pd.DataFrame) -> Optional[Path]:
        """生成资源使用对比图表"""
        if "avg_gpu_util" not in df.columns or df["avg_gpu_util"].sum() == 0:
            return None
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        if HAS_SEABORN:
            sns.barplot(data=df, x="model", y="avg_gpu_util", ax=ax1, palette="rocket")
            sns.barplot(data=df, x="model", y="avg_cpu_util", ax=ax2, palette="mako")
        else:
            ax1.bar(df["model"], df["avg_gpu_util"], color='#FF6B6B')
            ax2.bar(df["model"], df["avg_cpu_util"], color='#4ECDC4')
        
        ax1.set_title("GPU 使用率对比", fontsize=12, fontweight='bold')
        ax1.set_xlabel("模型", fontsize=10)
        ax1.set_ylabel("GPU 使用率 (%)", fontsize=10)
        ax1.tick_params(axis='x', rotation=30)
        
        ax2.set_title("CPU 使用率对比", fontsize=12, fontweight='bold')
        ax2.set_xlabel("模型", fontsize=10)
        ax2.set_ylabel("CPU 使用率 (%)", fontsize=10)
        ax2.tick_params(axis='x', rotation=30)
        
        fig.tight_layout()
        output_path = self.output_dir / "chart_resource_usage.png"
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        return output_path

    def generate_charts(self, df: pd.DataFrame) -> Dict[str, Optional[Path]]:
        """生成所有图表"""
        charts: Dict[str, Optional[Path]] = {}
        
        charts["speed_comparison"] = self.generate_speed_comparison_chart(df)
        charts["efficiency"] = self.generate_efficiency_chart(df)
        charts["resource_usage"] = self.generate_resource_usage_chart(df)
        
        return charts

    def generate_insights(self, df: pd.DataFrame) -> List[str]:
        """生成性能洞察"""
        insights = []
        
        # 速度分析
        fastest_model = df.loc[df["average_speed"].idxmax()]
        slowest_model = df.loc[df["average_speed"].idxmin()]
        speed_range = df["average_speed"].max() - df["average_speed"].min()
        
        insights.append(f"**速度分析**:")
        insights.append(f"- 最快模型: {fastest_model['model']} ({fastest_model['average_speed']:.1f} tokens/s)")
        insights.append(f"- 最慢模型: {slowest_model['model']} ({slowest_model['average_speed']:.1f} tokens/s)")
        insights.append(f"- 速度范围: {speed_range:.1f} tokens/s")
        
        # 效率分析
        most_efficient = df.loc[df["efficiency_score"].idxmax()]
        insights.append(f"- 最效率模型: {most_efficient['model']} (效率分数: {most_efficient['efficiency_score']:.3f})")
        
        # 稳定性分析
        most_stable = df.loc[df["stability_score"].idxmax()]
        insights.append(f"- 最稳定模型: {most_stable['model']} (稳定性分数: {most_stable['stability_score']:.3f})")
        
        # 难度级别分析
        insights.append(f"\n**难度级别表现**:")
        for difficulty in DIFFICULTY_LEVELS:
            col_name = f"{difficulty}_speed"
            if col_name in df.columns:
                best_model = df.loc[df[col_name].idxmax()]
                insights.append(f"- {difficulty.title()} 最佳: {best_model['model']} ({best_model[col_name]:.1f} tokens/s)")
        
        return insights

    def generate_report(self, df: pd.DataFrame, charts: Dict[str, Optional[Path]]) -> Path:
        """生成分析报告"""
        report_path = self.output_dir / "speed_test_analysis_report.md"
        
        # 生成洞察
        insights = self.generate_insights(df)
        
        # 创建排名表
        ranking_table = df[["model", "average_speed", "efficiency_score", "stability_score", "overall_score"]].copy()
        ranking_table = ranking_table.round(3)
        ranking_table.columns = ["模型", "平均速度", "效率分数", "稳定性分数", "综合分数"]
        
        # 生成报告内容
        lines = [
            "# Speed Test 分析报告",
            "",
            f"- 生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}",
            f"- 数据来源: {self.input_path}",
            f"- 测试模型数: {len(df)}",
            f"- 总测试次数: {df['test_count'].sum()}",
            "",
            "## 模型排名",
            "",
            tabulate(ranking_table.values.tolist(), headers=ranking_table.columns.tolist(), tablefmt="github"),
            "",
            "## 性能洞察",
        ]
        
        lines.extend(insights)
        lines.extend([
            "",
            "## 详细性能指标",
            "",
            "### 速度表现 (tokens/s)",
        ])
        
        # 添加速度详细表
        speed_table = df[["model", "simple_speed", "medium_speed", "complex_speed", "average_speed"]].copy()
        speed_table = speed_table.round(1)
        speed_table.columns = ["模型", "简单问题", "中等问题", "复杂问题", "平均速度"]
        lines.append(tabulate(speed_table.values.tolist(), headers=speed_table.columns.tolist(), tablefmt="github"))
        
        # 资源使用表
        if "avg_gpu_util" in df.columns and df["avg_gpu_util"].sum() > 0:
            lines.extend([
                "",
                "### 资源使用情况",
                "",
            ])
            resource_table = df[["model", "avg_gpu_util", "avg_cpu_util"]].copy()
            resource_table = resource_table.round(1)
            resource_table.columns = ["模型", "GPU使用率 (%)", "CPU使用率 (%)"]
            lines.append(tabulate(resource_table.values.tolist(), headers=resource_table.columns.tolist(), tablefmt="github"))
        
        # 图表部分
        lines.extend([
            "",
            "## 可视化图表",
            "",
        ])
        
        for chart_name, chart_path in charts.items():
            if chart_path:
                readable_name = {
                    "speed_comparison": "速度对比图",
                    "efficiency": "效率对比图",
                    "resource_usage": "资源使用图"
                }.get(chart_name, chart_name)
                lines.append(f"- [{readable_name}](./{chart_path.name})")
        
        lines.extend([
            "",
            "---",
            f"*报告由 Speed Test 分析器生成 | {datetime.now():%Y-%m-%d %H:%M:%S}*",
            "",
            "**使用说明**: 本报告基于 speed_test.py 的测试结果生成，提供模型性能的综合分析。",
            "建议根据具体应用场景选择最适合的模型。"
        ])
        
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path

    def run_analysis(self) -> None:
        """执行完整的分析流程"""
        try:
            print("Starting Speed Test Analysis...")
            print(f"Input file: {self.input_path}")
            print(f"Output directory: {self.output_dir}")

            # 加载数据
            print("Loading data...")
            self.load_data()

            if self.df.empty:
                print("Error: No valid data found")
                return

            print(f"Successfully loaded {len(self.df)} test records from {len(self.summary_df)} models")

            # 计算性能指标
            print("Computing performance metrics...")
            metrics_df = self.compute_performance_metrics()

            # 创建输出目录
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # 保存详细结果
            results_csv = self.output_dir / "performance_metrics.csv"
            metrics_df.to_csv(results_csv, index=False, encoding="utf-8")
            print(f"Performance metrics saved to: {results_csv}")

            # 生成图表
            print("Generating visualization charts...")
            charts = self.generate_charts(metrics_df)

            # 显示生成的图表
            generated_charts = [name for name, path in charts.items() if path]
            if generated_charts:
                print(f"Generated charts: {', '.join(generated_charts)}")
            else:
                print("No charts generated")

            # 生成报告
            print("Generating analysis report...")
            report_path = self.generate_report(metrics_df, charts)
            print(f"Analysis report generated: {report_path}")

            # 显示推荐结果
            print("\n=== Top 3 Models ===")
            for index, row in metrics_df.head(3).iterrows():
                print(f"  {index + 1}. {row['model']} (Overall Score: {row['overall_score']:.3f})")

            print("\n[SUCCESS] Analysis completed!")
            print("Output files:")
            print(f"  - Performance metrics: {results_csv}")
            print(f"  - Analysis report: {report_path}")
            print(f"  - Chart files: {self.output_dir}/chart_*.png")

        except Exception as e:
            print(f"Error during analysis: {e}")
            print("Please check input file format and dependencies")
            raise


def auto_detect_input() -> Path:
    """自动检测最新的速度测试结果文件"""
    speed_test_dir = Path("speed_test_results")
    candidates: List[Path] = []
    
    if speed_test_dir.exists():
        candidates.extend(speed_test_dir.glob("speed_test_results_*.json"))
        candidates.extend(speed_test_dir.glob("speed_test_summary_*.csv"))
    
    if not candidates:
        raise FileNotFoundError("No speed test files found. Use --input to specify a file.")
    
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    print(f"Auto-detected input file: {candidates[0]}")
    return candidates[0]


def main() -> None:
    """CLI 入口点"""
    parser = argparse.ArgumentParser(description="Speed Test Results Analyzer")
    parser.add_argument("--input", "-i", help="Path to speed test JSON or CSV file")
    parser.add_argument("--out", "-o", default="analysis_results", help="Directory for generated charts and tables")
    parser.add_argument(
        "--weights",
        help="Custom weights JSON, e.g. '{\"speed\":0.6,\"efficiency\":0.3,\"stability\":0.1}'",
    )

    args = parser.parse_args()

    try:
        if args.weights:
            weight_overrides = json.loads(args.weights)
            if not isinstance(weight_overrides, dict):
                raise ValueError
        else:
            weight_overrides = None
    except ValueError:
        print("Invalid weight configuration. Falling back to defaults.")
        weight_overrides = None

    if args.input:
        input_path = Path(args.input)
    else:
        input_path = auto_detect_input()

    analyzer = SpeedTestAnalyzer(
        input_path=input_path, 
        output_dir=Path(args.out), 
        weights=weight_overrides
    )
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
