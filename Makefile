# Ollama Benchmark Makefile

.PHONY: install test clean lint benchmark help

# Default target
help:
	@echo "Ollama Benchmark - Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make test       - Run tests (basic syntax check)"
	@echo "  make lint       - Run code linting"
	@echo "  make benchmark  - Run default benchmark"
	@echo "  make example    - Run simple example"
	@echo "  make clean      - Clean temporary files"
	@echo "  make help       - Show this help message"

# Install dependencies
install:
	pip install -r requirements.txt

# Install in development mode
install-dev:
	pip install -e .

# Run basic syntax check and simple tests
test:
	python -m py_compile ollama_benchmark.py
	python -c "import ollama_benchmark; print('✓ Module imports successfully')"

# Run code linting (if flake8 is available)
lint:
	@command -v flake8 >/dev/null 2>&1 && flake8 ollama_benchmark.py --max-line-length=100 || echo "flake8 not available, skipping lint"

# Run default benchmark
benchmark:
	python ollama_benchmark.py --models llama2 --runs 2

# Run benchmark with config file
benchmark-config:
	python ollama_benchmark.py --config config.yaml

# Run simple example
example:
	cd examples && python simple_benchmark.py

# Clean temporary files
clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	rm -f *.json *.yaml simple_benchmark_results.* 2>/dev/null || true

# Check Ollama connection
check-ollama:
	@curl -s http://localhost:11434/api/tags >/dev/null && echo "✓ Ollama is running" || echo "✗ Ollama not accessible at localhost:11434"