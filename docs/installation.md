# Installation Guide

This guide will help you install and set up dLLM-Serve for diffusion language model inference.

## Prerequisites

- Python 3.10 or 3.11
- CUDA 12.6 (for GPU acceleration)
- 16GB+ GPU memory (for 8B models)
- Linux system (recommended)

## Step 1: Create Conda Environment

```bash
# Create a new conda environment
conda create -n dllm python=3.11
conda activate dllm
```

## Step 2: Clone Repository

```bash
git clone https://github.com/chosen-ox/dLLM-Serve.git
cd dLLM-Serve
```

## Step 3: Install Dependencies

Install in development mode:

```bash
pip install -e . --no-build-isolation
```

This will install all required dependencies including:
- PyTorch 2.4+
- Transformers 4.51+
- Triton 3.0+ (for GPU kernels)
- FastAPI + Uvicorn (for server mode)

## Step 4: Load CUDA Modules (Cluster/HPC)

If you're on a cluster with module system:

```bash
module load CUDA/12.6.0
module load nodejs/20.13.1-GCCcore-13.3.0
```

## Step 5: Download Models

### LLaDA-8B-Instruct

```bash
huggingface-cli download --resume-download GSAI-ML/LLaDA-8B-Instruct \
  --local-dir ./LLaDA-8B-Instruct \
  --local-dir-use-symlinks False
```

### Dream-v0-Instruct-7B

```bash
huggingface-cli download --resume-download Dream-org/Dream-v0-Instruct-7B \
  --local-dir ./Dream-v0-Instruct-7B \
  --local-dir-use-symlinks False
```

## Step 6: Verify Installation

Run a quick test:

```bash
python examples/simple_test.py
```

## Troubleshooting

### CUDA Not Available

If you get "CUDA not available" errors:

1. Check CUDA installation:
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

2. Reinstall PyTorch with CUDA support:
```bash
pip install torch --upgrade --index-url https://download.pytorch.org/whl/cu121
```

### Out of Memory

For GPU memory constraints, reduce max sequence length:

```python
llm = LLM(model_path, max_model_len=2048)
```

### Import Errors

Make sure you installed in development mode:

```bash
pip uninstall dllm-serve
pip install -e . --no-build-isolation
```

## Next Steps

- See [Quick Start Tutorial](quickstart.md) for usage examples
- See [API Reference](api_reference.md) for complete API documentation
