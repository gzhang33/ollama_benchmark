#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Output Generation Performance Test Tool
专门测试输出生成时间的性能测试工具，参考 ollama --verbose 的输出生成时间计算
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

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.direct_ollama_tester import DirectOllamaTester
from src.ollama_utils import (
    get_test_prompts,
    export_results_to_csv,
    generate_summary_table,
    save_json_results,
    create_timestamp,
    ensure_output_dir
)



# Test prompts moved to ollama_utils.py

# Resource monitoring moved to resource_monitor.py


def test_model_output_generation(
    tester: DirectOllamaTester,
    model_name: str,
    prompt: str,
) -> Dict[str, Any]:
    """
    测试单个模型的输出生成时间（使用直接方法）
    
    Args:
        tester: 直接输出生成时间测试器
        model_name: 模型名称
        prompt: 测试提示
        
    Returns:
        包含输出生成时间等详细信息的字典
    """
    result = tester.test_output_generation_time(model_name, prompt, num_predict=400)
    
    # 重命名字段以保持兼容性
    if result["success"]:
        result["output_tokens"] = result["eval_count"]
        result["duration"] = result["eval_duration"]
        result["tokens_per_second"] = result["eval_rate"]
    
    return result


def run_output_generation_test(
    tester: DirectOllamaTester,
    model_name: str,
    prompts: List[str],
    output_dir: Path,
    progress: Optional['tqdm'] = None,
) -> Dict[str, Any]:
    """
    运行输出生成时间测试
    
    Args:
        tester: 输出生成时间测试器
        model_name: 模型名称
        prompts: 测试提示列表
        output_dir: 输出目录
        progress: 进度条
        
    Returns:
        测试结果
    """
    results: List[Dict[str, Any]] = []
    total_eval_tokens = 0
    total_eval_duration = 0.0
    successful_tests = 0

    for prompt in prompts:
        outcome = test_model_output_generation(tester, model_name, prompt)

        if outcome["success"]:
            successful_tests += 1
            total_eval_tokens += outcome["eval_count"]
            total_eval_duration += outcome["eval_duration"]
            results.append(
                {
                    "model": model_name,
                    "prompt": prompt,
                    "eval_count": outcome["eval_count"],
                    "eval_duration": outcome["eval_duration"],
                    "eval_rate": outcome["eval_rate"],
                    "response_preview": outcome["response"][:200],
                    # 兼容性字段
                    "output_tokens": outcome["eval_count"],
                    "duration": outcome["eval_duration"],
                    "tokens_per_second": outcome["eval_rate"],
                }
            )

        if progress:
            progress.update(1)

    avg_eval_rate = total_eval_tokens / total_eval_duration if total_eval_duration > 0 else 0.0
    success_rate = successful_tests / len(prompts) if prompts else 0.0

    return {
        "model": model_name,
        "total_eval_tokens": total_eval_tokens,
        "total_eval_duration": total_eval_duration,
        "average_eval_rate": avg_eval_rate,
        "success_rate": success_rate,
        "test_count": len(results),
        "details": results,
        # 兼容性字段
        "total_tokens": total_eval_tokens,
        "total_duration": total_eval_duration,
        "average_speed": avg_eval_rate,
    }


# CSV export and summary functions moved to ollama_utils.py
# Model listing function moved to ollama_client.py
# Resource monitoring functions moved to resource_monitor.py






def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Model Output Generation Performance Test Tool - 专门测试输出生成时间",
    )
    parser.add_argument("--ollama-path", default="ollama", help="Path to ollama executable")
    parser.add_argument("--model", help="Specific model to test (test all if not specified)")
    parser.add_argument("--output-dir", default="data/output_generation_results", help="Output directory for results")
    parser.add_argument("--analysis-dir", default="data/analysis_results", help="Directory for analysis outputs")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip visualization and reporting")

    args = parser.parse_args()

    # 初始化测试器和工具（使用直接方法）
    tester = DirectOllamaTester(ollama_path=args.ollama_path)
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)

    print("Loading test prompts...")
    prompts = get_test_prompts()
    print(f"Loaded {len(prompts)} test prompts")

    if args.model:
        models = [args.model]
    else:
        print("\nDiscovering available models...")
        # 使用简单的模型发现方法
        from src.ollama_client import OllamaClient
        client = OllamaClient(args.base_url)
        models = client.get_available_models()
        print(f"Found {len(models)} models: {', '.join(models)}")

    if not models:
        print("No models to test")
        return

    total_tests = len(models) * len(prompts)
    progress_bar = tqdm(
        total=total_tests,
        desc="Initializing output generation test...",
        unit="test",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    all_results: List[Dict[str, Any]] = []
    timestamp = create_timestamp()

    try:
        for model in models:
            progress_bar.set_description(f"Testing model: {model}")
            result = run_output_generation_test(
                tester,
                model,
                prompts,
                output_dir,
                progress_bar,
            )
            all_results.append(result)
    finally:
        progress_bar.close()

    # 生成输出文件
    detail_file = output_dir / f"output_generation_details_{timestamp}.csv"
    summary_file = output_dir / f"output_generation_summary_{timestamp}.csv"
    json_file = output_dir / f"output_generation_results_{timestamp}.json"

    # 导出结果
    export_results_to_csv(all_results, detail_file)
    generate_summary_table(all_results, summary_file)
    save_json_results(all_results, json_file, timestamp, len(models))

    print(f"\nAll results saved to: {output_dir}/")
    print(f"  - Detailed CSV: {detail_file.name}")
    print(f"  - Summary CSV: {summary_file.name}")
    print(f"  - JSON data: {json_file.name}")
    
    # 显示关键输出生成时间指标
    print(f"\n{'=' * 80}")
    print("OUTPUT GENERATION PERFORMANCE SUMMARY")
    print(f"{'=' * 80}")
    for result in all_results:
        print(f"Model: {result['model']}")
        print(f"  - Average Output Generation Speed: {result['average_eval_rate']:.2f} tokens/s")
        print(f"  - Total Output Tokens: {result['total_eval_tokens']}")
        print(f"  - Total Output Generation Time: {result['total_eval_duration']:.2f}s")
        print(f"  - Success Rate: {result['success_rate']:.2f}")
        print()
    
    if not args.skip_analysis:
        print("Analysis features available in the original performance.py")


if __name__ == '__main__':
    main()