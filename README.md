# Ollama Model Performance Testing Tool

A comprehensive toolkit for testing local Ollama model performance, helping you choose the most suitable model for local deployment.

## 🆕 Key Features

### Core Optimization Features
- **Interactive Parallel Testing**: Single interface input with all models responding in parallel for side-by-side comparison
- **Standardized Speed Testing**: 15 standard test questions with accurate speed calculation formula (Total Tokens ÷ Total Time)
- **Resource Monitoring**: Real-time GPU/CPU usage monitoring
- **Test Tool Menu**: User-friendly menu interface for quick tool selection

### Time Efficiency Improvements
- **Original Quality Test**: 49 samples × 90 seconds = 73 minutes/model
- **New Interactive Test**: 1 question × 30 seconds = 0.5 minutes (all models in parallel)
- **Time Saved: 99%+**

## 🔧 Installation Requirements

- Python 3.8+
- Ollama service installed and running (default `http://localhost:11434`)
- Dependencies installation:
```bash
pip install -r requirements.txt
```

## 📁 File Structure

```
ollama_test/
├── README.md                    # This file (complete usage guide)
├── ollama_client.py             # Ollama client module
├── ollama_utils.py              # Utility functions module
├── resource_monitor.py          # Resource monitoring module
├── font_config.py               # Font configuration module
├── visualization.py             # Visualization and analysis module
├── app.py                       # 🆕 Interactive parallel testing tool
├── speed_test.py                # 🆕 Speed testing tool (with resource monitoring)
├── requirements.txt             # Python dependencies
├── speed_test_results/          # 🆕 Speed test results directory
├── app_results/                 # 🆕 Interactive test results directory
└── analysis_results/            # Analysis reports directory
```

## 🚀 Quick Start

### 🆕 Method 1: Interactive Parallel Testing (Recommended for Model Comparison)
```bash
python app.py
```
**Core Features**:
- Single interface prompt input
- All models respond in parallel
- Side-by-side result comparison
- Real-time quality assessment of all model responses

**Usage Example**:
```bash
python app.py
💬 Your prompt: Explain what quantum computing is?
# Wait for all models to respond in parallel, view comparison results
```

### 🆕 Method 2: Speed Testing (Recommended for Performance Evaluation)
```bash
python speed_test.py
```
**Core Features**:
- 3 standardized test questions (medium difficulty, ~400 token output)
- Accurate speed calculation: Total Tokens ÷ Total Time
- GPU/CPU resource monitoring
- Detailed report generation (CSV + JSON)

**Test Question Set**:
- Algorithm implementation tasks
- Problem-solving challenges
- Technical analysis requests

**Usage Example**:
```bash
python speed_test.py                    # Test all models
python speed_test.py --model "gemma3:4b"  # Test specific model
```

## 📚 Tool Features

### 🆕 New Tools

#### 1. Interactive Parallel Testing (app.py)
**Core Advantages**:
- Replaces original quality testing, saves 99%+ time
- Real-time comparison of all model response quality
- Supports custom question testing

**Parameter Description**:
| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `--base-url` | http://localhost:11434 | Ollama service address |
| `--prompt` | None | Single test prompt |
| `--output-dir` | app_results | Result save directory |
| `--max-workers` | 5 | Parallel worker threads |

#### 2. Speed Testing (speed_test.py)
**Core Advantages**:
- Scientific speed calculation formula
- Standardized test question set
- Complete resource monitoring

**Speed Calculation Formula**:
```
Accurate Average Speed (tokens/s) = Total Output Tokens from All Questions ÷ Total Time for All Questions (seconds)
```

**Output Files**:
- `speed_test_details_[timestamp].csv` - Detailed test data
- `speed_test_summary_[timestamp].csv` - Summary statistics table
- `speed_test_results_[timestamp].json` - Complete JSON data

**Parameter Description**:
| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `--base-url` | http://localhost:11434 | Ollama service address |
| `--model` | None | Specific model to test |
| `--output-dir` | speed_test_results | Result save directory |
| `--analysis-dir` | analysis_results | Analysis output directory |
| `--skip-analysis` | False | Skip visualization and reporting |
| `--collect-resources` | True | Collect GPU/CPU usage metrics |

### Analysis Tools

#### Visualization and Analysis (visualization.py)
- **Intelligent Data Conversion**: Automatic JSON/CSV format recognition and conversion
- **Performance Ranking**: Model ranking based on comprehensive scoring
- **Visualization Charts**: Throughput, response time, radar chart comparisons
- **Scenario Recommendations**: Best model recommendations for different use cases
- **Detailed Reports**: Complete Markdown format analysis reports

## 🎯 Usage Scenario Guide

### Scenario 1: Quick Comparison of All Models' Response Quality
**Recommended Tool**: `app.py`
```bash
python app.py
💬 Your prompt: Explain what quantum computing is?
# View all models' response comparison
```

### Scenario 2: Systematic Model Speed Performance Evaluation
**Recommended Tool**: `speed_test.py`
```bash
python speed_test.py
# Run 3 standard tests, generate detailed speed report
```

### Scenario 3: Compare Specific Model Performance
**Recommended Tool**: `speed_test.py` with specific model
```bash
python speed_test.py --model "model1"
python speed_test.py --model "model2"
# Compare CSV results from both tests
```

### Scenario 4: Long-term Model Optimization Tracking
**Recommended Tool**: `speed_test.py` with regular runs
```bash
# Run weekly
python speed_test.py
# Historical data saved in speed_test_results/ directory
```

## 📊 Supported Models

This toolkit can **dynamically detect** and test all locally installed LLM models (automatically filters out embedding models):

- ✅ Dynamic model detection: Automatically discovers locally installed models
- ✅ Smart filtering: Automatically excludes embedding models (bge, bert, etc.)
- ✅ Supports all GGUF format models

**Currently Detected Models**:
- Qwen series (qwen2.5, qwen2.5-instruct)
- Gemma series (gemma3, gemma2)
- DeepSeek series (deepseek-r1)
- Llama series (llama3.1)
- Mistral series (mistral)
- Phi series (phi4-mini)
- And other locally installed LLM models

## 🧭 Usage Guide

1. Ensure Ollama is running locally:
```bash
ollama serve
```

2. Pull and install models to test:
```bash
ollama list
ollama pull <model_name>
```

3. Run tests and analysis (see "Quick Start" section).

## 📊 Tool Feature Comparison

| Feature | speed_test.py | app.py |
|---------|---------------|---------------------|
| Model Speed Testing | ✓✓✓ Professional | - |
| Interactive Comparison | - | ✓✓✓ Core Feature |
| Parallel Execution | - | ✓✓✓ Core Feature |
| Resource Monitoring | ✓✓✓ GPU+CPU | - |
| Standardized Question Set | ✓✓✓ 3 Standard Questions | - |
| Accurate Speed Formula | ✓✓✓ Total Tokens ÷ Total Time | - |
| Detailed Reports | CSV+JSON+Table | Markdown |
| Side-by-side Comparison | - | ✓✓✓ Core Feature |
| Test Time | 5-10 minutes | 0.5-2 minutes |

## 🧩 Metrics Explanation and Examples

### Key Metrics

- **tokens_per_second**: Generation speed, higher values mean faster
- **duration**: Total time to complete task
- **output_tokens**: Number of tokens generated
- **success_rate**: Percentage of successful test completions
- **gpu_util**: GPU utilization percentage
- **cpu_util**: CPU utilization percentage

### Interactive Test Output Example
```
================================================================================
MODEL RESPONSES COMPARISON
================================================================================

Prompt: Explain what quantum computing is?

================================================================================

[1] Model: gemma3:4b
    Duration: 3.45s
    Tokens: 128
    Speed: 37.10 tokens/s
    Response:
    ----------------------------------------------------------------------------
    Quantum computing is a new type of computation that utilizes quantum mechanical principles...
    ----------------------------------------------------------------------------

[2] Model: qwen2.5:14b
    Duration: 8.23s
    Tokens: 156
    Speed: 18.95 tokens/s
    Response:
    ----------------------------------------------------------------------------
    Quantum computing is a computational technology based on quantum bits...
    ----------------------------------------------------------------------------

================================================================================
SUMMARY STATISTICS
================================================================================

Total models tested: 12
Successful: 11
Failed: 1

Speed statistics:
  Fastest: 45.23 tokens/s (gemma3:4b)
  Slowest: 12.34 tokens/s (qwen2.5:14b)
  Average: 28.56 tokens/s
```

### Speed Test Output Example
```
================================================================================
Testing model: gemma3:4b
================================================================================

Testing 3 prompts...
  1/3: Given two strings word1 and word2, calculate the minimum number of operations...
    ✓ Tokens: 156, Duration: 4.12s, Speed: 37.86 tokens/s

  2/3: Given a string containing only '(' and ')', find the length...
    ✓ Tokens: 189, Duration: 5.23s, Speed: 36.14 tokens/s

  3/3: Given a 2D matrix consisting of 'X' and 'O', find all regions...
    ✓ Tokens: 203, Duration: 5.67s, Speed: 35.82 tokens/s

================================================================================
Summary for gemma3:4b:
  Total output tokens: 548
  Total duration: 15.02s
  Average speed: 36.48 tokens/s
================================================================================
```

## 🎯 Selection Recommendations (Reference)

1. **Daily Use**: Prioritize models with higher tokens_per_second and moderate memory usage
2. **High Quality Requirements**: Consider models with 13B+ parameters
3. **Speed Priority**: Choose lightweight 7B-level models
4. **Memory Constrained**: Choose quantized or lightweight optimized models

## 🛠️ Troubleshooting

If you encounter issues:

1. Ensure Ollama service is running: `ollama serve`
2. Check if models are installed: `ollama list`
3. Common issues and suggestions:
   - **Cannot connect to Ollama service**: Confirm service is started and check port (default 11434)
   - **Model not found**: First execute `ollama pull <model_name>`, then verify with `ollama list`
   - **Insufficient memory**: Close memory-consuming programs, use smaller or quantized models
   - **Test timeout**: Some models need more time, can increase script timeout threshold
   - **GPU monitoring unavailable**: If no NVIDIA GPU or nvidia-smi not installed, GPU monitoring will be skipped automatically
   - **Slow parallel testing**: Reduce parallel worker threads `--max-workers 3`

## ⚙️ Advanced Configuration Tips

- Adjust scoring weights in analysis modules to match business focus
- Add custom test cases, adjust generation length (e.g., `num_predict`), temperature, timeout, etc.
- All test results are saved in corresponding result directories for long-term tracking and analysis

## 🚀 Quick Command Reference

```bash
# Interactive testing (recommended for model comparison)
python app.py

# Speed testing (recommended for performance evaluation)
python speed_test.py

# Analysis and visualization
python speed_test.py  # Automatically generates analysis after testing
```

## 📋 Modular Architecture

This project uses a modular architecture for better maintainability and code reuse:

### Core Modules
- `ollama_client.py`: Ollama API client functionality
- `ollama_utils.py`: Data processing and utility functions
- `resource_monitor.py`: GPU/CPU monitoring
- `font_config.py`: Font configuration for visualization
- `visualization.py`: Chart generation and analysis

### Benefits
- **Code Reusability**: Shared functionality across tools
- **Maintainability**: Clear separation of concerns
- **Extensibility**: Easy to add new features
- **Backward Compatibility**: Existing code continues to work

For detailed module information, see `MODULARIZATION_RULES.md`.

---