# Project Structure

## Directory Organization

```
ollama_test/
├── README.md                    # Complete usage guide and documentation
├── requirements.txt             # Python dependencies
├── test_prompts.txt            # Standardized test questions (15 prompts across 3 difficulty levels)
│
├── # Core Testing Tools
├── speed_test.py               # Standardized speed testing with resource monitoring
├── interactive_test.py         # GUI-based parallel model comparison
├── complete_test.py            # End-to-end workflow automation
├── analyzer.py                 # Performance analysis and visualization
│
├── # Results Directories (auto-created)
├── speed_test_results/         # Speed test outputs (CSV, JSON)
├── interactive_test_results/   # Interactive test outputs (Markdown)
├── analysis_results/           # Charts, reports, and metrics
│
└── # Python Environment
    ├── .venv/                  # Virtual environment
    └── __pycache__/           # Python bytecode cache
```

## File Naming Conventions

### Result Files
- **Speed Test**: `speed_test_results_YYYYMMDD_HHMMSS.json`
- **Speed Test Details**: `speed_test_details_YYYYMMDD_HHMMSS.csv`
- **Speed Test Summary**: `speed_test_summary_YYYYMMDD_HHMMSS.csv`
- **Interactive Test**: `interactive_test_YYYYMMDD_HHMMSS.md`
- **Analysis Report**: `speed_test_analysis_report.md`
- **Charts**: `chart_[type].png` (e.g., `chart_speed_comparison.png`)

### Data Flow
1. **Input**: `test_prompts.txt` → Testing tools
2. **Raw Data**: Testing tools → `*_results/` directories
3. **Analysis**: `analyzer.py` processes latest results → `analysis_results/`
4. **Output**: Charts (PNG) + Reports (Markdown) + Metrics (CSV)

## Code Organization Patterns

### Main Entry Points
- Each tool is executable as `python [tool_name].py`
- All tools support `--help` for usage information
- Common arguments: `--base-url`, `--model`, `--output-dir`

### Shared Utilities
- Model discovery: `get_available_models()`
- API communication: `query_model()`, `test_model_with_prompt()`
- Resource monitoring: `get_gpu_usage()`, `get_cpu_usage()`
- Data export: CSV/JSON/Markdown formatters

### Configuration
- Default Ollama URL: `http://localhost:11434`
- Default timeouts: 120 seconds per model query
- Auto-filtering: Excludes embedding models (bge, bert, etc.)
- Parallel workers: 5 concurrent threads (configurable)

## Development Guidelines

### Adding New Test Types
1. Create new tool following existing patterns
2. Use consistent argument parsing with `argparse`
3. Save results to dedicated subdirectory
4. Support both JSON and CSV output formats
5. Include progress bars with `tqdm`

### Extending Analysis
1. Modify `analyzer.py` to handle new data formats
2. Add new chart types in `generate_charts()` method
3. Update report generation in `generate_report()`
4. Maintain backward compatibility with existing result files