# Quick Usage Guide

## Getting Started

### 1. Prerequisites
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama
ollama serve

# Pull some models
ollama pull llama2
ollama pull mistral
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Basic Benchmark
```bash
# Test default configuration
python ollama_benchmark.py

# Test specific models
python ollama_benchmark.py -m llama2 -m mistral

# Custom prompts
python ollama_benchmark.py -p "Hello world" -p "Write a poem"

# More runs for better statistics
python ollama_benchmark.py -r 5
```

### 4. Use Configuration File
```bash
# Edit config.yaml to your needs
python ollama_benchmark.py -c config.yaml

# Save results
python ollama_benchmark.py -c config.yaml -o results.json
```

### 5. Programmatic Usage
```python
from ollama_benchmark import OllamaBenchmark, BenchmarkConfig

config = BenchmarkConfig(
    models=["llama2", "mistral"],
    prompts=["Hello", "Write a poem"],
    num_runs=3
)

benchmark = OllamaBenchmark(config)
benchmark.run_benchmark()
benchmark.generate_report()
benchmark.save_results("results.json")
```

## Common Use Cases

### Compare Models
```bash
python ollama_benchmark.py -m llama2 -m mistral -m codellama -r 5
```

### Test Specific Scenarios
```bash
python ollama_benchmark.py \
  -p "Code: Write a Python function" \
  -p "Math: Solve 2x + 5 = 15" \
  -p "Creative: Write a haiku" \
  -r 3
```

### Monitor Performance Over Time
```bash
# Run daily benchmark
python ollama_benchmark.py -c config.yaml -o "results_$(date +%Y%m%d).json"
```

## Output Formats

- **Terminal**: Colored, formatted output
- **JSON**: `--output results.json`
- **YAML**: `--output results.yaml`

## Configuration Options

- `ollama_url`: Ollama server URL (default: http://localhost:11434)
- `models`: List of models to test
- `prompts`: List of test prompts
- `num_runs`: Number of runs per test (default: 3)
- `save_responses`: Save model responses (default: false)

## Tips

1. **Start Simple**: Begin with 1-2 models and basic prompts
2. **Multiple Runs**: Use 3+ runs for reliable statistics
3. **Monitor Resources**: Watch CPU/GPU usage during benchmarks
4. **Custom Prompts**: Test prompts relevant to your use case
5. **Save Results**: Keep historical data for comparison