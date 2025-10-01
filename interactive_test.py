#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Model Comparison Tool
交互式模型对比工具 - 单界面输入，所有模型并行回答
"""

import argparse
import concurrent.futures
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


def get_available_models(base_url: str) -> List[str]:
    """Get available LLM models from Ollama."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        response.raise_for_status()
        data = response.json()
        
        models = []
        for model_info in data.get("models", []):
            model_name = model_info["name"]
            if not any(keyword in model_name.lower() 
                      for keyword in ["embedding", "bge", "bert"]):
                models.append(model_name)
        
        return sorted(models)
    except Exception as e:
        print(f"Error getting models: {e}")
        return []


def query_model(
    base_url: str,
    model_name: str,
    prompt: str,
    timeout: int = 120
) -> Dict[str, Any]:
    """Query a single model and return response with timing."""
    url = f"{base_url}/api/generate"
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 512
        }
    }
    
    start_time = time.perf_counter()
    
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        data = response.json()
        response_text = data.get("response", "")
        
        # Token count
        eval_count = data.get("eval_count", 0)
        if eval_count == 0:
            eval_count = len(response_text.split())
        
        tokens_per_second = eval_count / duration if duration > 0 else 0
        
        return {
            "model": model_name,
            "success": True,
            "response": response_text,
            "duration": duration,
            "tokens": eval_count,
            "tokens_per_second": tokens_per_second
        }
        
    except requests.Timeout:
        return {
            "model": model_name,
            "success": False,
            "error": "Timeout",
            "duration": timeout
        }
    except Exception as e:
        return {
            "model": model_name,
            "success": False,
            "error": str(e),
            "duration": 0
        }


def parallel_query_all_models(
    base_url: str,
    models: List[str],
    prompt: str,
    max_workers: int = 5
) -> List[Dict[str, Any]]:
    """Query all models in parallel and return results."""
    print(f"\n{'='*80}")
    print(f"Querying {len(models)} models in parallel...")
    print(f"{'='*80}\n")
    
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_model = {
            executor.submit(query_model, base_url, model, prompt): model 
            for model in models
        }
        
        for future in concurrent.futures.as_completed(future_to_model):
            model = future_to_model[future]
            try:
                result = future.result()
                results.append(result)
                
                if result["success"]:
                    print(f"✓ {model}: Completed in {result['duration']:.2f}s "
                          f"({result['tokens_per_second']:.2f} tokens/s)")
                else:
                    print(f"✗ {model}: Failed - {result.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"✗ {model}: Exception - {e}")
                results.append({
                    "model": model,
                    "success": False,
                    "error": str(e)
                })
    
    # Sort by model name
    results.sort(key=lambda x: x["model"])
    
    return results


def display_results(results: List[Dict[str, Any]], prompt: str) -> None:
    """Display all model responses in a comparison format."""
    print(f"\n{'='*80}")
    print("MODEL RESPONSES COMPARISON")
    print(f"{'='*80}")
    print(f"\nPrompt: {prompt}")
    print(f"\n{'='*80}\n")
    
    successful_results = [r for r in results if r["success"]]
    failed_results = [r for r in results if not r["success"]]
    
    # Display successful responses
    for idx, result in enumerate(successful_results, 1):
        print(f"[{idx}] Model: {result['model']}")
        print(f"    Duration: {result['duration']:.2f}s")
        print(f"    Tokens: {result['tokens']}")
        print(f"    Speed: {result['tokens_per_second']:.2f} tokens/s")
        print(f"    Response:")
        print(f"    {'-'*76}")
        
        # Format response with indentation
        response_lines = result['response'].split('\n')
        for line in response_lines:
            print(f"    {line}")
        
        print(f"    {'-'*76}")
        print()
    
    # Display failed models
    if failed_results:
        print(f"\n{'='*80}")
        print("FAILED MODELS")
        print(f"{'='*80}\n")
        for result in failed_results:
            print(f"✗ {result['model']}: {result.get('error', 'Unknown error')}")
    
    # Display summary statistics
    if successful_results:
        print(f"\n{'='*80}")
        print("SUMMARY STATISTICS")
        print(f"{'='*80}\n")
        
        speeds = [r['tokens_per_second'] for r in successful_results]
        durations = [r['duration'] for r in successful_results]
        
        print(f"Total models tested: {len(results)}")
        print(f"Successful: {len(successful_results)}")
        print(f"Failed: {len(failed_results)}")
        print(f"\nSpeed statistics:")
        print(f"  Fastest: {max(speeds):.2f} tokens/s ({successful_results[speeds.index(max(speeds))]['model']})")
        print(f"  Slowest: {min(speeds):.2f} tokens/s ({successful_results[speeds.index(min(speeds))]['model']})")
        print(f"  Average: {sum(speeds)/len(speeds):.2f} tokens/s")
        print(f"\nDuration statistics:")
        print(f"  Fastest: {min(durations):.2f}s ({successful_results[durations.index(min(durations))]['model']})")
        print(f"  Slowest: {max(durations):.2f}s ({successful_results[durations.index(max(durations))]['model']})")
        print(f"  Average: {sum(durations)/len(durations):.2f}s")


def save_results_to_file(
    results: List[Dict[str, Any]],
    prompt: str,
    output_dir: Path
) -> None:
    """Save results to a markdown file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"interactive_test_{timestamp}.md"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# Interactive Model Comparison\n\n")
        f.write(f"**Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Prompt**: {prompt}\n\n")
        f.write(f"---\n\n")
        
        successful_results = [r for r in results if r["success"]]
        
        for idx, result in enumerate(successful_results, 1):
            f.write(f"## [{idx}] {result['model']}\n\n")
            f.write(f"- **Duration**: {result['duration']:.2f}s\n")
            f.write(f"- **Tokens**: {result['tokens']}\n")
            f.write(f"- **Speed**: {result['tokens_per_second']:.2f} tokens/s\n\n")
            f.write(f"**Response**:\n\n")
            f.write(f"```\n{result['response']}\n```\n\n")
            f.write(f"---\n\n")
        
        # Add summary
        if successful_results:
            speeds = [r['tokens_per_second'] for r in successful_results]
            durations = [r['duration'] for r in successful_results]
            
            f.write(f"## Summary Statistics\n\n")
            f.write(f"- **Total models**: {len(results)}\n")
            f.write(f"- **Successful**: {len(successful_results)}\n")
            f.write(f"- **Failed**: {len([r for r in results if not r['success']])}\n\n")
            f.write(f"### Speed Statistics\n\n")
            f.write(f"- Fastest: {max(speeds):.2f} tokens/s ({successful_results[speeds.index(max(speeds))]['model']})\n")
            f.write(f"- Slowest: {min(speeds):.2f} tokens/s ({successful_results[speeds.index(min(speeds))]['model']})\n")
            f.write(f"- Average: {sum(speeds)/len(speeds):.2f} tokens/s\n\n")
    
    print(f"\nResults saved to: {output_file}")


def interactive_mode(base_url: str, output_dir: Path) -> None:
    """Run in interactive mode with continuous prompts."""
    print(f"\n{'='*80}")
    print("INTERACTIVE MODEL COMPARISON MODE")
    print(f"{'='*80}\n")
    
    # Get available models
    print("Discovering available models...")
    models = get_available_models(base_url)
    
    if not models:
        print("No models available. Please check your Ollama service.")
        return
    
    print(f"\nFound {len(models)} models:")
    for idx, model in enumerate(models, 1):
        print(f"  {idx}. {model}")
    
    print(f"\n{'='*80}")
    print("Enter your prompts below. All models will respond in parallel.")
    print("Type 'quit' or 'exit' to stop.")
    print(f"{'='*80}\n")
    
    while True:
        try:
            prompt = input("\n💬 Your prompt: ").strip()
            
            if not prompt:
                continue
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("\nExiting interactive mode. Goodbye!")
                break
            
            # Query all models in parallel
            results = parallel_query_all_models(base_url, models, prompt)
            
            # Display results
            display_results(results, prompt)
            
            # Ask if user wants to save
            save = input("\n💾 Save results to file? (y/n): ").strip().lower()
            if save == 'y':
                save_results_to_file(results, prompt, output_dir)
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Interactive model comparison tool - test all models with a single prompt"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434",
        help="Base URL of Ollama service"
    )
    parser.add_argument(
        "--prompt",
        help="Single prompt to test (non-interactive mode)"
    )
    parser.add_argument(
        "--output-dir",
        default="interactive_test_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum parallel workers"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if args.prompt:
        # Single prompt mode
        print(f"Running single prompt test...")
        models = get_available_models(args.base_url)
        
        if not models:
            print("No models available")
            return
        
        print(f"Found {len(models)} models")
        results = parallel_query_all_models(args.base_url, models, args.prompt, args.max_workers)
        display_results(results, args.prompt)
        save_results_to_file(results, args.prompt, output_dir)
    else:
        # Interactive mode
        interactive_mode(args.base_url, output_dir)


if __name__ == "__main__":
    main()

