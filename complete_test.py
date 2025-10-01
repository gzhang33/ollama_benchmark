#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Test and Analysis Workflow
完整测试和分析工作流程
"""

import subprocess
import sys
import time
from pathlib import Path

def run_command(cmd, description):
    """运行命令并显示进度"""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"Command failed: {cmd}")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print("Output:")
            print(e.stdout)
        if e.stderr:
            print("Error output:")
            print(e.stderr)
        return False

def main():
    """主工作流程"""
    print("Ollama Model Performance Test and Analysis Workflow")
    print("=" * 60)
    
    # 确保test_result目录存在
    test_result_dir = Path("test_result")
    test_result_dir.mkdir(exist_ok=True)
    
    # 步骤1: 运行速度测试
    print("\n1. Running Ollama speed test...")
    success = run_command("python speed_test.py", "Speed Test")
    
    if not success:
        print("Performance test failed. Please check Ollama service.")
        print("Make sure Ollama is running: ollama serve")
        sys.exit(1)
    
    # 检查速度测试结果文件是否生成
    speed_test_dir = Path("speed_test_results")
    if speed_test_dir.exists():
        # 查找最新的结果文件
        result_files = list(speed_test_dir.glob("speed_test_results_*.json"))
        if result_files:
            latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
            print(f"Speed test results saved to: {latest_file}")
        else:
            print("Warning: No speed test result files found")
    else:
        print("Warning: Speed test results directory not found")
    
    # 步骤2: 运行分析
    print("\n2. Running performance analysis...")
    success = run_command("python analyzer.py", "Performance Analysis")
    
    if not success:
        print("Analysis failed. Please check the analyzer.py script.")
        sys.exit(1)
    
    # 检查分析结果
    analysis_dir = Path("analysis_results")
    if analysis_dir.exists():
        report_file = analysis_dir / "analysis_report.md"
        if report_file.exists():
            print(f"Analysis report generated: {report_file}")
        
        print("\nGenerated files:")
        for file in analysis_dir.iterdir():
            print(f"  - {file.name}")
    
    print("\n" + "="*60)
    print("Workflow completed successfully!")
    print("="*60)
    print("Files generated:")
    print(f"  - Speed test results: speed_test_results/")
    print(f"  - Analysis report: {analysis_dir}/analysis_report.md")
    print(f"  - Performance charts: {analysis_dir}/chart_*.png")
    print(f"  - Detailed scores: {analysis_dir}/scored_results.csv")
    
    print("\nNext steps:")
    print("1. View the analysis report for model recommendations")
    print("2. Check the performance charts for visual comparisons")
    print("3. Use scored_results.csv for detailed analysis")
    print("4. Re-run this script anytime to update results")

if __name__ == "__main__":
    main()

