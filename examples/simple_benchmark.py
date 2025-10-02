#!/usr/bin/env python3
"""
Simple example of using the Ollama benchmark programmatically.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ollama_benchmark import OllamaBenchmark, BenchmarkConfig

def main():
    """Run a simple benchmark example."""
    
    # Create a simple configuration
    config = BenchmarkConfig(
        ollama_url="http://localhost:11434",
        models=["llama2"],  # You can add more models here
        prompts=[
            "Hello, how are you?",
            "What is the capital of France?",
            "Write a haiku about programming."
        ],
        num_runs=2,
        save_responses=True
    )
    
    # Create and run benchmark
    benchmark = OllamaBenchmark(config)
    
    print("Running simple Ollama benchmark example...")
    print("-" * 50)
    
    benchmark.run_benchmark()
    benchmark.generate_report()
    
    # Save results
    benchmark.save_results("simple_benchmark_results.json")
    
    print("\nExample completed! Check simple_benchmark_results.json for detailed results.")

if __name__ == "__main__":
    main()