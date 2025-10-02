#!/usr/bin/env python3
"""
Demo script showing what the Ollama benchmark output would look like.
This simulates the tool running with actual Ollama instances.
"""

from colorama import init, Fore, Style
from tabulate import tabulate
import time

# Initialize colorama for colored output
init(autoreset=True)

def demo_benchmark_output():
    """Show what the benchmark tool output would look like."""
    
    print(f"{Fore.BLUE}Ollama Benchmark Tool - Demo Output{Style.RESET_ALL}")
    print("="*60)
    print()
    
    # Simulate starting the benchmark
    print(f"{Fore.BLUE}Starting Ollama Benchmark...{Style.RESET_ALL}")
    print("Ollama URL: http://localhost:11434")
    print("Models to test: llama2, mistral")
    print("Number of runs per test: 3")
    print()
    
    # Simulate connection check
    print("Available models: llama2, mistral, codellama, phi")
    print()
    
    print("="*50)
    print("Running benchmarks...")
    print("="*50)
    
    # Simulate testing llama2
    print(f"\n{Fore.GREEN}Testing model: llama2{Style.RESET_ALL}")
    
    prompts = [
        "Hello, how are you today?",
        "Write a short poem about technology.",
        "Explain quantum computing in simple terms."
    ]
    
    # Mock results for llama2
    results_llama2 = [
        [(1.23, 15.4), (1.18, 16.2), (1.31, 14.8)],  # Hello prompt
        [(3.45, 12.8), (3.22, 13.5), (3.67, 12.1)],  # Poem prompt
        [(8.92, 18.7), (9.15, 18.2), (8.76, 19.1)]   # Quantum prompt
    ]
    
    test_num = 1
    for i, prompt in enumerate(prompts):
        print(f"  Prompt {i+1}: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
        
        run_times = []
        run_tps = []
        
        for j, (time_val, tps_val) in enumerate(results_llama2[i]):
            print(f"    Run {j+1}/3 ({test_num}/18)", end=" ... ")
            # Add a small delay to simulate processing
            time.sleep(0.1)
            print(f"{Fore.GREEN}✓{Style.RESET_ALL} {time_val:.2f}s, {tps_val:.1f} t/s")
            run_times.append(time_val)
            run_tps.append(tps_val)
            test_num += 1
        
        avg_time = sum(run_times) / len(run_times)
        avg_tps = sum(run_tps) / len(run_tps)
        print(f"    Average: {avg_time:.2f}s, {avg_tps:.1f} t/s")
    
    # Simulate testing mistral
    print(f"\n{Fore.GREEN}Testing model: mistral{Style.RESET_ALL}")
    
    # Mock results for mistral (generally faster)
    results_mistral = [
        [(0.95, 18.9), (0.89, 20.1), (1.02, 17.6)],  # Hello prompt
        [(2.34, 19.8), (2.18, 21.2), (2.45, 18.9)],  # Poem prompt
        [(6.78, 22.4), (6.92, 21.8), (6.54, 23.1)]   # Quantum prompt
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"  Prompt {i+1}: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
        
        run_times = []
        run_tps = []
        
        for j, (time_val, tps_val) in enumerate(results_mistral[i]):
            print(f"    Run {j+1}/3 ({test_num}/18)", end=" ... ")
            time.sleep(0.1)
            print(f"{Fore.GREEN}✓{Style.RESET_ALL} {time_val:.2f}s, {tps_val:.1f} t/s")
            run_times.append(time_val)
            run_tps.append(tps_val)
            test_num += 1
        
        avg_time = sum(run_times) / len(run_times)
        avg_tps = sum(run_tps) / len(run_tps)
        print(f"    Average: {avg_time:.2f}s, {avg_tps:.1f} t/s")

def demo_benchmark_report():
    """Show what the final benchmark report would look like."""
    
    print(f"\n{Fore.BLUE}Benchmark Report{Style.RESET_ALL}")
    print("="*50)
    print("Total tests: 18")
    print("Successful: 18")
    print("Failed: 0")
    
    print(f"\n{Fore.GREEN}Performance Summary:{Style.RESET_ALL}")
    
    # Create mock summary table
    table_data = [
        ["llama2", 9, "4.52s", "1.18s", "9.15s", "15.5", 2847],
        ["mistral", 9, "3.44s", "0.89s", "6.92s", "20.3", 3156]
    ]
    
    headers = ["Model", "Tests", "Avg Time", "Min Time", "Max Time", "Avg T/s", "Total Tokens"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    print(f"\n{Fore.CYAN}Key Insights:{Style.RESET_ALL}")
    print("• Mistral shows 24% faster average response time")
    print("• Mistral generates 31% more tokens per second on average")
    print("• Both models show consistent performance across runs")
    print("• Complex prompts (quantum computing) take 4-7x longer than simple greetings")

def demo_json_output():
    """Show what the JSON output would contain."""
    
    print(f"\n{Fore.YELLOW}Sample JSON Output Structure:{Style.RESET_ALL}")
    
    sample_json = """{
  "config": {
    "ollama_url": "http://localhost:11434",
    "models": ["llama2", "mistral"],
    "prompts": ["Hello, how are you today?", "..."],
    "num_runs": 3,
    "save_responses": false
  },
  "results": [
    {
      "model": "llama2",
      "prompt": "Hello, how are you today?",
      "response_time": 1.23,
      "tokens_generated": 19,
      "tokens_per_second": 15.4,
      "response_text": "",
      "success": true,
      "error_message": null
    }
  ],
  "summary": {
    "total_tests": 18,
    "successful_tests": 18,
    "failed_tests": 0,
    "average_response_time": 3.98,
    "average_tokens_per_second": 17.9,
    "total_tokens_generated": 6003
  }
}"""
    
    print(sample_json)

def main():
    """Run the complete demo."""
    
    print(f"{Fore.MAGENTA}🚀 OLLAMA BENCHMARK TOOL DEMONSTRATION 🚀{Style.RESET_ALL}")
    print()
    print("This demo shows what the tool output would look like when")
    print("benchmarking a real Ollama instance with multiple models.")
    print()
    
    demo_benchmark_output()
    demo_benchmark_report()
    demo_json_output()
    
    print(f"\n{Fore.GREEN}Demo completed! 🎉{Style.RESET_ALL}")
    print("\nTo use the actual tool:")
    print("1. Install and start Ollama: https://ollama.ai")
    print("2. Pull some models: ollama pull llama2")
    print("3. Run the benchmark: python ollama_benchmark.py")

if __name__ == "__main__":
    main()