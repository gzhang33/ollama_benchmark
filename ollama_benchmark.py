#!/usr/bin/env python3
"""
Ollama Benchmark Tool
Test your local Ollama performance with various models and prompts.
"""

import json
import time
import statistics
import requests
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
import click
from tabulate import tabulate
from colorama import init, Fore, Style
import yaml
import os

# Initialize colorama for cross-platform colored terminal text
init(autoreset=True)

@dataclass
class BenchmarkResult:
    """Data class to store benchmark results for a single test."""
    model: str
    prompt: str
    response_time: float  # seconds
    tokens_generated: int
    tokens_per_second: float
    response_text: str
    success: bool
    error_message: Optional[str] = None

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tests."""
    ollama_url: str = "http://localhost:11434"
    models: List[str] = None
    prompts: List[str] = None
    num_runs: int = 3
    output_format: str = "table"  # table, json, csv
    save_responses: bool = False
    
    def __post_init__(self):
        if self.models is None:
            self.models = ["llama2"]
        if self.prompts is None:
            self.prompts = [
                "Hello, how are you today?",
                "Write a short poem about technology.",
                "Explain quantum computing in simple terms."
            ]

class OllamaBenchmark:
    """Main benchmark class for testing Ollama performance."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[BenchmarkResult] = []
    
    def check_ollama_connection(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.config.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models from Ollama."""
        try:
            response = requests.get(f"{self.config.ollama_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
            return []
        except requests.exceptions.RequestException:
            return []
    
    def run_single_test(self, model: str, prompt: str) -> BenchmarkResult:
        """Run a single benchmark test with the given model and prompt."""
        start_time = time.time()
        
        try:
            # Prepare the request payload
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False
            }
            
            # Make the request to Ollama
            response = requests.post(
                f"{self.config.ollama_url}/api/generate",
                json=payload,
                timeout=120
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                response_text = data.get("response", "")
                
                # Estimate token count (rough approximation: ~4 chars per token)
                tokens_generated = len(response_text) // 4
                tokens_per_second = tokens_generated / response_time if response_time > 0 else 0
                
                return BenchmarkResult(
                    model=model,
                    prompt=prompt,
                    response_time=response_time,
                    tokens_generated=tokens_generated,
                    tokens_per_second=tokens_per_second,
                    response_text=response_text if self.config.save_responses else "",
                    success=True
                )
            else:
                return BenchmarkResult(
                    model=model,
                    prompt=prompt,
                    response_time=response_time,
                    tokens_generated=0,
                    tokens_per_second=0,
                    response_text="",
                    success=False,
                    error_message=f"HTTP {response.status_code}: {response.text}"
                )
                
        except requests.exceptions.RequestException as e:
            end_time = time.time()
            response_time = end_time - start_time
            
            return BenchmarkResult(
                model=model,
                prompt=prompt,
                response_time=response_time,
                tokens_generated=0,
                tokens_per_second=0,
                response_text="",
                success=False,
                error_message=str(e)
            )
    
    def run_benchmark(self) -> None:
        """Run the complete benchmark suite."""
        print(f"{Fore.BLUE}Starting Ollama Benchmark...{Style.RESET_ALL}")
        print(f"Ollama URL: {self.config.ollama_url}")
        print(f"Models to test: {', '.join(self.config.models)}")
        print(f"Number of runs per test: {self.config.num_runs}")
        print()
        
        # Check Ollama connection
        if not self.check_ollama_connection():
            print(f"{Fore.RED}Error: Cannot connect to Ollama at {self.config.ollama_url}{Style.RESET_ALL}")
            print("Please make sure Ollama is running and accessible.")
            return
        
        # Get available models
        available_models = self.get_available_models()
        print(f"Available models: {', '.join(available_models) if available_models else 'None'}")
        
        # Validate requested models
        missing_models = [m for m in self.config.models if m not in available_models]
        if missing_models and available_models:
            print(f"{Fore.YELLOW}Warning: Models not found: {', '.join(missing_models)}{Style.RESET_ALL}")
        
        print("\n" + "="*50)
        print("Running benchmarks...")
        print("="*50)
        
        total_tests = len(self.config.models) * len(self.config.prompts) * self.config.num_runs
        current_test = 0
        
        for model in self.config.models:
            print(f"\n{Fore.GREEN}Testing model: {model}{Style.RESET_ALL}")
            
            for prompt_idx, prompt in enumerate(self.config.prompts):
                print(f"  Prompt {prompt_idx + 1}: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
                
                # Run multiple iterations for statistical significance
                run_results = []
                for run in range(self.config.num_runs):
                    current_test += 1
                    print(f"    Run {run + 1}/{self.config.num_runs} ({current_test}/{total_tests})", end=" ... ")
                    
                    result = self.run_single_test(model, prompt)
                    run_results.append(result)
                    self.results.append(result)
                    
                    if result.success:
                        print(f"{Fore.GREEN}✓{Style.RESET_ALL} {result.response_time:.2f}s, {result.tokens_per_second:.1f} t/s")
                    else:
                        print(f"{Fore.RED}✗{Style.RESET_ALL} {result.error_message}")
                
                # Print summary for this prompt
                successful_runs = [r for r in run_results if r.success]
                if successful_runs:
                    avg_time = statistics.mean([r.response_time for r in successful_runs])
                    avg_tps = statistics.mean([r.tokens_per_second for r in successful_runs])
                    print(f"    Average: {avg_time:.2f}s, {avg_tps:.1f} t/s")
    
    def generate_report(self) -> None:
        """Generate and display benchmark report."""
        if not self.results:
            print("No benchmark results to report.")
            return
        
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        print(f"\n{Fore.BLUE}Benchmark Report{Style.RESET_ALL}")
        print("="*50)
        print(f"Total tests: {len(self.results)}")
        print(f"Successful: {len(successful_results)}")
        print(f"Failed: {len(failed_results)}")
        
        if successful_results:
            print(f"\n{Fore.GREEN}Performance Summary:{Style.RESET_ALL}")
            
            # Group results by model
            model_stats = {}
            for result in successful_results:
                if result.model not in model_stats:
                    model_stats[result.model] = []
                model_stats[result.model].append(result)
            
            # Create summary table
            table_data = []
            for model, results in model_stats.items():
                avg_time = statistics.mean([r.response_time for r in results])
                min_time = min([r.response_time for r in results])
                max_time = max([r.response_time for r in results])
                avg_tps = statistics.mean([r.tokens_per_second for r in results])
                total_tokens = sum([r.tokens_generated for r in results])
                
                table_data.append([
                    model,
                    len(results),
                    f"{avg_time:.2f}s",
                    f"{min_time:.2f}s",
                    f"{max_time:.2f}s",
                    f"{avg_tps:.1f}",
                    total_tokens
                ])
            
            headers = ["Model", "Tests", "Avg Time", "Min Time", "Max Time", "Avg T/s", "Total Tokens"]
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        if failed_results:
            print(f"\n{Fore.RED}Failed Tests:{Style.RESET_ALL}")
            for result in failed_results:
                print(f"  {result.model}: {result.error_message}")
    
    def save_results(self, filename: str) -> None:
        """Save benchmark results to file."""
        results_data = {
            "config": asdict(self.config),
            "results": [asdict(result) for result in self.results],
            "summary": self._generate_summary()
        }
        
        with open(filename, 'w') as f:
            if filename.endswith('.yaml') or filename.endswith('.yml'):
                yaml.dump(results_data, f, default_flow_style=False)
            else:
                json.dump(results_data, f, indent=2)
        
        print(f"Results saved to {filename}")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        successful_results = [r for r in self.results if r.success]
        
        if not successful_results:
            return {"total_tests": len(self.results), "successful_tests": 0}
        
        return {
            "total_tests": len(self.results),
            "successful_tests": len(successful_results),
            "failed_tests": len(self.results) - len(successful_results),
            "average_response_time": statistics.mean([r.response_time for r in successful_results]),
            "average_tokens_per_second": statistics.mean([r.tokens_per_second for r in successful_results]),
            "total_tokens_generated": sum([r.tokens_generated for r in successful_results])
        }

def load_config_file(config_path: str) -> BenchmarkConfig:
    """Load configuration from YAML or JSON file."""
    with open(config_path, 'r') as f:
        if config_path.endswith(('.yaml', '.yml')):
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    
    return BenchmarkConfig(**data)

@click.command()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file (JSON/YAML)')
@click.option('--url', default='http://localhost:11434', help='Ollama API URL')
@click.option('--models', '-m', multiple=True, help='Models to benchmark (can be specified multiple times)')
@click.option('--prompts', '-p', multiple=True, help='Prompts to test (can be specified multiple times)')
@click.option('--runs', '-r', default=3, type=int, help='Number of runs per test')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
@click.option('--save-responses', is_flag=True, help='Save model responses in results')
def main(config, url, models, prompts, runs, output, save_responses):
    """Ollama Benchmark Tool - Test your local Ollama performance."""
    
    # Load configuration
    if config:
        benchmark_config = load_config_file(config)
    else:
        benchmark_config = BenchmarkConfig(
            ollama_url=url,
            models=list(models) if models else None,
            prompts=list(prompts) if prompts else None,
            num_runs=runs,
            save_responses=save_responses
        )
    
    # Run benchmark
    benchmark = OllamaBenchmark(benchmark_config)
    benchmark.run_benchmark()
    benchmark.generate_report()
    
    # Save results if requested
    if output:
        benchmark.save_results(output)

if __name__ == "__main__":
    main()