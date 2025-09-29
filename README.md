# Ollama Benchmark

Test your local Ollama performance with comprehensive benchmarks across different models and prompts.

## Features

- 🚀 **Performance Testing**: Measure response times and tokens per second
- 📊 **Multiple Models**: Test multiple Ollama models simultaneously
- 🎯 **Custom Prompts**: Use predefined or custom prompts for testing
- 📈 **Statistical Analysis**: Multiple runs for accurate performance metrics
- 💾 **Export Results**: Save results in JSON or YAML format
- 🎨 **Beautiful Reports**: Colored terminal output with detailed tables
- ⚙️ **Configurable**: Use configuration files or command-line options

## Installation

### Prerequisites

1. **Ollama**: Make sure Ollama is installed and running on your system
   ```bash
   # Install Ollama (if not already installed)
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Start Ollama
   ollama serve
   
   # Pull some models for testing
   ollama pull llama2
   ollama pull codellama
   ollama pull mistral
   ```

2. **Python 3.8+**: Required for running the benchmark tool

### Install the benchmark tool

```bash
# Clone the repository
git clone https://github.com/gzhang33/ollama_benchmark.git
cd ollama_benchmark

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

## Quick Start

### Basic Usage

```bash
# Run with default settings
python ollama_benchmark.py

# Test specific models
python ollama_benchmark.py --models llama2 --models mistral

# Use custom prompts
python ollama_benchmark.py --prompts "Hello world" --prompts "Write a poem"

# More runs for better statistics
python ollama_benchmark.py --runs 5
```

### Using Configuration File

```bash
# Use the provided configuration template
python ollama_benchmark.py --config config.yaml

# Save results to file
python ollama_benchmark.py --config config.yaml --output results.json
```

## Configuration

The tool can be configured using a YAML or JSON configuration file. See `config.yaml` for an example:

```yaml
# Ollama server configuration
ollama_url: "http://localhost:11434"

# Models to benchmark
models:
  - "llama2"
  - "codellama" 
  - "mistral"

# Test prompts
prompts:
  - "Hello, how are you today?"
  - "Write a short poem about technology."
  - "Explain quantum computing in simple terms."

# Number of runs per test
num_runs: 3

# Save model responses
save_responses: false
```

## Command Line Options

```
Usage: ollama_benchmark.py [OPTIONS]

Options:
  -c, --config PATH       Configuration file (JSON/YAML)
  --url TEXT             Ollama API URL (default: http://localhost:11434)
  -m, --models TEXT      Models to benchmark (can be used multiple times)
  -p, --prompts TEXT     Prompts to test (can be used multiple times)
  -r, --runs INTEGER     Number of runs per test (default: 3)
  -o, --output PATH      Output file for results
  --save-responses       Save model responses in results
  --help                 Show this message and exit
```

## Example Output

```
Starting Ollama Benchmark...
Ollama URL: http://localhost:11434
Models to test: llama2, mistral
Number of runs per test: 3

Available models: llama2, mistral, codellama

==================================================
Running benchmarks...
==================================================

Testing model: llama2
  Prompt 1: Hello, how are you today?
    Run 1/3 (1/18) ... ✓ 1.23s, 15.4 t/s
    Run 2/3 (2/18) ... ✓ 1.18s, 16.2 t/s
    Run 3/3 (3/18) ... ✓ 1.31s, 14.8 t/s
    Average: 1.24s, 15.5 t/s

Benchmark Report
==================================================
Total tests: 18
Successful: 18
Failed: 0

Performance Summary:
┌─────────┬───────┬──────────┬──────────┬──────────┬─────────┬──────────────┐
│ Model   │ Tests │ Avg Time │ Min Time │ Max Time │ Avg T/s │ Total Tokens │
├─────────┼───────┼──────────┼──────────┼──────────┼─────────┼──────────────┤
│ llama2  │ 9     │ 2.45s    │ 1.18s    │ 4.12s    │ 18.3    │ 1247         │
│ mistral │ 9     │ 1.87s    │ 0.94s    │ 2.83s    │ 23.1    │ 1156         │
└─────────┴───────┴──────────┴──────────┴──────────┴─────────┴──────────────┘
```

## Output Formats

Results can be exported in multiple formats:

- **JSON**: Detailed results with all metrics
- **YAML**: Human-readable format with full data
- **Terminal Table**: Beautiful formatted output for immediate viewing

## Metrics Measured

- **Response Time**: Total time to generate response
- **Tokens per Second**: Generation speed (estimated)
- **Success Rate**: Percentage of successful requests
- **Token Count**: Total tokens generated (estimated)

## Use Cases

- **Model Comparison**: Compare performance across different models
- **Hardware Testing**: Test performance on different hardware configurations
- **Prompt Optimization**: Find optimal prompts for your use case
- **Performance Monitoring**: Track performance over time
- **Capacity Planning**: Understand throughput capabilities

## Troubleshooting

### Common Issues

1. **Connection Error**: Make sure Ollama is running
   ```bash
   ollama serve
   ```

2. **Model Not Found**: Pull the model first
   ```bash
   ollama pull <model-name>
   ```

3. **Timeout Issues**: Increase timeout for complex prompts or slower hardware

### Debug Mode

For debugging, you can check:
- Ollama logs: Check Ollama server logs for errors
- Network connectivity: Ensure the benchmark tool can reach Ollama
- Model availability: Verify models are downloaded and available

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
