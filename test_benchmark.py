#!/usr/bin/env python3
"""
Basic tests for the Ollama benchmark functionality.
"""

import json
import tempfile
import os
from ollama_benchmark import OllamaBenchmark, BenchmarkConfig, BenchmarkResult

def test_config_creation():
    """Test configuration creation and defaults."""
    print("Testing configuration creation...")
    
    # Test default configuration
    config = BenchmarkConfig()
    assert config.ollama_url == "http://localhost:11434"
    assert config.models == ["llama2"]
    assert len(config.prompts) == 3
    assert config.num_runs == 3
    
    # Test custom configuration
    custom_config = BenchmarkConfig(
        ollama_url="http://custom:8080",
        models=["model1", "model2"],
        prompts=["test prompt"],
        num_runs=5
    )
    assert custom_config.ollama_url == "http://custom:8080"
    assert custom_config.models == ["model1", "model2"]
    assert custom_config.prompts == ["test prompt"]
    assert custom_config.num_runs == 5
    
    print("✓ Configuration tests passed")

def test_benchmark_result():
    """Test benchmark result data structure."""
    print("Testing BenchmarkResult...")
    
    result = BenchmarkResult(
        model="test-model",
        prompt="test prompt",
        response_time=1.5,
        tokens_generated=100,
        tokens_per_second=66.7,
        response_text="test response",
        success=True
    )
    
    assert result.model == "test-model"
    assert result.success == True
    assert result.tokens_generated == 100
    
    print("✓ BenchmarkResult tests passed")

def test_benchmark_initialization():
    """Test benchmark object initialization."""
    print("Testing OllamaBenchmark initialization...")
    
    config = BenchmarkConfig()
    benchmark = OllamaBenchmark(config)
    
    assert benchmark.config == config
    assert len(benchmark.results) == 0
    
    print("✓ OllamaBenchmark initialization tests passed")

def test_connection_check():
    """Test connection checking (will fail since Ollama is not running)."""
    print("Testing connection check...")
    
    config = BenchmarkConfig()
    benchmark = OllamaBenchmark(config)
    
    # This should return False since Ollama is not running
    connected = benchmark.check_ollama_connection()
    assert connected == False  # Expected behavior in test environment
    
    print("✓ Connection check tests passed (correctly detected no Ollama)")

def test_model_listing():
    """Test model listing (will return empty since Ollama is not running)."""
    print("Testing model listing...")
    
    config = BenchmarkConfig()
    benchmark = OllamaBenchmark(config)
    
    models = benchmark.get_available_models()
    assert isinstance(models, list)
    assert len(models) == 0  # Expected in test environment
    
    print("✓ Model listing tests passed")

def test_result_saving():
    """Test saving results to file."""
    print("Testing result saving...")
    
    config = BenchmarkConfig()
    benchmark = OllamaBenchmark(config)
    
    # Add a mock result
    result = BenchmarkResult(
        model="test-model",
        prompt="test prompt",
        response_time=1.5,
        tokens_generated=100,
        tokens_per_second=66.7,
        response_text="test response",
        success=True
    )
    benchmark.results.append(result)
    
    # Test JSON saving
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        json_filename = f.name
    
    try:
        benchmark.save_results(json_filename)
        
        # Verify the file was created and contains valid JSON
        with open(json_filename, 'r') as f:
            data = json.load(f)
        
        assert 'config' in data
        assert 'results' in data
        assert 'summary' in data
        assert len(data['results']) == 1
        
        print("✓ JSON saving tests passed")
    
    finally:
        os.unlink(json_filename)

def test_summary_generation():
    """Test summary statistics generation."""
    print("Testing summary generation...")
    
    config = BenchmarkConfig()
    benchmark = OllamaBenchmark(config)
    
    # Add mock results
    results = [
        BenchmarkResult("model1", "prompt1", 1.0, 50, 50.0, "response1", True),
        BenchmarkResult("model1", "prompt2", 2.0, 100, 50.0, "response2", True),
        BenchmarkResult("model1", "prompt3", 0.0, 0, 0.0, "", False, "error"),
    ]
    
    benchmark.results.extend(results)
    
    summary = benchmark._generate_summary()
    
    assert summary['total_tests'] == 3
    assert summary['successful_tests'] == 2
    assert summary['failed_tests'] == 1
    assert summary['average_response_time'] == 1.5
    assert summary['total_tokens_generated'] == 150
    
    print("✓ Summary generation tests passed")

def main():
    """Run all tests."""
    print("Running Ollama Benchmark Tests")
    print("=" * 40)
    
    tests = [
        test_config_creation,
        test_benchmark_result,
        test_benchmark_initialization,
        test_connection_check,
        test_model_listing,
        test_result_saving,
        test_summary_generation,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
    
    print("\n" + "=" * 40)
    print(f"Tests completed: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    exit(main())