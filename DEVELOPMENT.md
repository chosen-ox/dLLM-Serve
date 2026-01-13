# Development Guide

This guide provides detailed instructions for setting up a development environment and contributing to dLLM-Serve.

## Prerequisites

### Required

- **Python 3.10-3.12** - The project requires Python 3.10 or newer
- **CUDA 12.6+** - For GPU support (required for model inference)
- **Git** - For version control

### Optional

- **Node.js 20+** - If working with server-related tooling
- **Docker** - For containerized development (coming soon)

## Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/chosen-ox/dLLM-Serve.git
cd dLLM-Serve
```

### 2. Load Required Modules (HPC Environments)

If you're on an HPC cluster with module system:

```bash
module load CUDA/12.6.0
module load nodejs/20.13.1-GCCcore-13.3.0
```

### 3. Create Python Environment

Using conda (recommended):

```bash
conda create -n dllmserve python=3.11
conda activate dllmserve
```

Or using venv:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4. Install Dependencies

Install the package in development mode:

```bash
pip install -e .
```

Install development dependencies:

```bash
pip install pytest pytest-cov black ipython jupyter
```

### 5. Verify Installation

```bash
python -c "from dllmserve import LLM; print('Installation successful!')"
```

## Project Structure

```
dllm-serve/
├── dllmserve/              # Core library
│   ├── __init__.py         # Package initialization
│   ├── llm.py              # Main LLM interface
│   ├── config.py           # Model configuration
│   ├── sampling_params.py  # Sampling parameters
│   ├── engine/             # Inference engine
│   │   ├── llm_engine.py   # Engine orchestrator
│   │   ├── model_runner.py # Model execution
│   │   ├── scheduler.py    # Request scheduling
│   │   ├── block_manager.py # KV cache management
│   │   └── sequence.py     # Sequence data structures
│   ├── layers/             # Neural network layers
│   │   ├── attention.py    # Attention mechanisms
│   │   ├── linear.py       # Linear layers with Triton
│   │   ├── layernorm.py    # Layer normalization
│   │   └── rotary_embedding.py # RoPE embeddings
│   ├── models/             # Model implementations
│   │   ├── llada.py        # LLaDA diffusion model
│   │   └── dream.py        # Dream diffusion model
│   ├── sparse/             # Sparse attention
│   │   └── state.py        # Sparse configuration
│   └── utils/              # Utility functions
│       ├── context.py      # Context management
│       └── loader.py       # Model loading
│
├── server/                 # REST API server
│   ├── app.py              # FastAPI application
│   └── engine_worker.py    # Async request worker
│
├── tests/                  # Test suite
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── performance/        # Performance tests
│
├── examples/               # Usage examples
└── docs/                   # Documentation
```

## Running Tests

### All Tests

```bash
pytest tests/
```

### Unit Tests Only

```bash
pytest tests/unit/
```

### Integration Tests

```bash
pytest tests/integration/
```

### Performance Tests

```bash
pytest tests/performance/
```

### With Coverage Report

```bash
pytest --cov=dllmserve --cov-report=html tests/
# Open htmlcov/index.html to view coverage
```

### Specific Test File

```bash
pytest tests/unit/test_sparse_attention.py
```

### Verbose Output

```bash
pytest -v tests/
```

## Code Formatting

### Format Code

```bash
black dllmserve/ server/ tests/ examples/
```

### Check Formatting (without changes)

```bash
black --check dllmserve/ server/ tests/ examples/
```

### Configure Black

Black is configured in `pyproject.toml`:

```toml
[tool.black]
line-length = 100
target-version = ['py310', 'py311', 'py312']
```

## Running Examples

### Basic Usage

```bash
python examples/basic_usage.py
```

### LLaDA Diffusion Model

```bash
python examples/llada_generation.py
```

### Dream Model

```bash
python examples/dream_generation.py
```

## Running the Server

### Start the Server

```bash
# Set model path (optional)
export MODEL_PATH="./LLaDA-8B-Instruct"

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 1
```

### Test the Server

```bash
# In another terminal
python tests/integration/test_server.py
```

## Model Download

### Downloading Models

Use Hugging Face CLI to download models:

```bash
# Download LLaDA-8B-Instruct
huggingface-cli download --resume-download GSAI-ML/LLaDA-8B-Instruct \
  --local-dir ./LLaDA-8B-Instruct \
  --local-dir-use-symlinks False

# Download Dream-v0-Instruct-7B
huggingface-cli download --resume-download Dream-org/Dream-v0-Instruct-7B \
  --local-dir ./Dream-v0-Instruct-7B \
  --local-dir-use-symlinks False
```

### Model Paths

The project supports these diffusion models:
- **LLaDA-8B-Instruct**: Latent Language Diffusion Autoregressive model
- **Dream-v0-Instruct-7B**: Dream diffusion model with shifted prediction

You can override these by:
1. Setting environment variables
2. Passing model paths directly to `LLM(model_path=...)`
3. Editing example scripts

## Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### GPU Memory Issues

If you encounter GPU memory issues:

1. Reduce `max_model_len` in `LLM` initialization:
   ```python
   llm = LLM(model_path, max_model_len=2048)
   ```

2. Use tensor parallelism for multi-GPU:
   ```python
   llm = LLM(model_path, tensor_parallel_size=2)
   ```

3. Enable sparse attention to reduce memory:
   ```python
   sparse_config = SparseConfig(enabled=True, retention_ratio=0.5)
   ```

### Common Issues

**Import Errors:**
- Ensure you're in the correct environment: `conda activate dllmserve`
- Reinstall in development mode: `pip install -e .`

**CUDA Errors:**
- Check CUDA version: `nvcc --version`
- Ensure CUDA modules are loaded (HPC): `module load CUDA/12.6.0`
- Verify GPU is visible: `nvidia-smi`

**Test Failures:**
- Ensure all dependencies are installed
- Check if GPU is available and has enough memory
- Some tests require specific model files

## Development Workflow

### Creating a New Feature

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Develop your feature:
   - Write code in appropriate modules
   - Add tests to `tests/unit/` or `tests/integration/`
   - Add examples to `examples/` if applicable
   - Update documentation in `docs/`

3. Test your changes:
   ```bash
   pytest tests/
   black dllmserve/ server/ tests/ examples/
   ```

4. Commit and push:
   ```bash
   git add .
   git commit -m "feat: description of your feature"
   git push origin feature/your-feature-name
   ```

5. Create a Pull Request on GitHub

### Code Review Process

- All PRs require review before merging
- Address reviewer feedback
- Keep PRs focused and reasonably sized
- Update tests and documentation as needed

## Performance Optimization

### Profiling

For profiling memory and performance:

```python
import torch
print(f"GPU Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"GPU Memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

### Benchmarking Changes

Before and after making performance changes, run the performance test suite:

```bash
pytest tests/performance/ -v
```

## Documentation

### Building Documentation

(Coming soon - Sphinx/MkDocs integration)

### Writing Documentation

- Add docstrings to all public functions and classes
- Update relevant files in `docs/` when adding features
- Keep documentation in sync with code changes

## Environment Variables

Useful environment variables for development:

- `MODEL_PATH` - Default model path for examples and server
- `DLLMSERVE_DISABLE_ATEXIT` - Disable atexit hooks for server mode
- `CUDA_VISIBLE_DEVICES` - Specify which GPUs to use

## Additional Resources

- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [GitHub Issues](https://github.com/chosen-ox/dLLM-Serve/issues) - Bug reports and feature requests

## Questions?

If you encounter issues or have questions:
1. Check existing GitHub issues
2. Create a new issue with detailed information
3. Ask in pull request discussions

Happy coding!
