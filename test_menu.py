#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Testing Menu
模型测试工具菜单
"""

import subprocess
import sys
from pathlib import Path


def print_menu():
    """Display main menu."""
    print("\n" + "="*80)
    print(" " * 25 + "模型测试工具集")
    print("="*80)
    print("\n请选择要运行的测试工具：\n")
    print("  [1] 交互式并行测试（interactive_test.py）")
    print("      - 单界面输入提示词")
    print("      - 所有模型并行回答")
    print("      - 结果对比展示")
    print()
    print("  [2] 速度测试（speed_test.py）")
    print("      - 标准化问题集测试")
    print("      - 准确速度计算")
    print("      - 资源监控（GPU/CPU）")
    print("      - 详细报表生成")
    print()
    print("  [3] 分析工具（analyzer.py）")
    print("      - 分析测试结果")
    print("      - 生成可视化图表")
    print()
    print("  [4] 完整测试流程（complete_test.py）")
    print("      - 速度测试 + 分析报告")
    print()
    print("  [0] 退出")
    print()
    print("="*80)


def run_command(cmd: str) -> bool:
    """Run a command and return success status."""
    try:
        result = subprocess.run(cmd, shell=True)
        return result.returncode == 0
    except Exception as e:
        print(f"\n错误: {e}")
        return False


def interactive_test():
    """Run interactive test."""
    print("\n启动交互式并行测试...")
    print("提示：输入 'quit' 或 'exit' 退出交互模式\n")
    
    cmd = "python interactive_test.py"
    run_command(cmd)


def speed_test():
    """Run speed test with options."""
    print("\n速度测试选项：")
    print("  [1] 测试所有模型（完整测试）")
    print("  [2] 测试指定模型")
    print("  [0] 返回主菜单")
    
    choice = input("\n请选择 [0-2]: ").strip()
    
    if choice == "1":
        print("\n启动所有模型速度测试...")
        run_command("python speed_test.py")
    elif choice == "2":
        model = input("\n请输入模型名称（例如 gemma3:4b）: ").strip()
        if model:
            print(f"\n启动模型 {model} 速度测试...")
            run_command(f'python speed_test.py --model "{model}"')
    elif choice == "0":
        return
    else:
        print("\n无效选择")




def analyzer():
    """Run analyzer."""
    print("\n启动分析工具...")
    run_command("python analyzer.py")


def complete_test():
    """Run complete test workflow."""
    print("\n启动完整测试流程...")
    print("包括：性能测试 → 分析报告")
    confirm = input("确认继续？(y/n): ").strip().lower()
    if confirm == 'y':
        run_command("python complete_test.py")


def check_dependencies():
    """Check if required files exist."""
    required_files = [
        "interactive_test.py",
        "speed_test.py",
        "test_prompts.txt",
        "analyzer.py",
        "complete_test.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("\n警告：以下文件缺失：")
        for file in missing_files:
            print(f"  - {file}")
        print("\n部分功能可能不可用")
        return False
    
    return True


def main():
    """Main menu loop."""
    print("\n正在检查环境...")
    check_dependencies()
    
    while True:
        print_menu()
        
        choice = input("请选择 [0-4]: ").strip()
        
        if choice == "1":
            interactive_test()
        elif choice == "2":
            speed_test()
        elif choice == "3":
            analyzer()
        elif choice == "4":
            complete_test()
        elif choice == "0":
            print("\n感谢使用！再见！\n")
            sys.exit(0)
        else:
            print("\n无效选择，请重试")
        
        input("\n按 Enter 继续...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断，退出程序")
        sys.exit(0)

