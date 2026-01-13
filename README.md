# dLLM-Serve

A lightweight, high-performance diffusion LLM (dLLM) serving framework built in ~1,200 lines of Python. Fast inference with sparse attention support for diffusion language models.

## Features

- **Fast Inference** - Efficient inference for diffusion language models with offline and online mode support
- **Flash Attention** - GPU-optimized attention implementation for faster inference
- **Sparse Attention** - Head-level sparsity for memory-efficient inference
- **Diffusion Models** - Support for LLaDA and Dream diffusion models
- **REST API Server** - FastAPI-based server for production deployments
- **Clean Codebase** - Readable implementation focused on clarity and performance

## Quick Start

### Installation

```bash
# Create a new conda environment
conda create -n dllm python=3.11
conda activate dllm

# Clone the repository
git clone https://github.com/chosen-ox/dLLM-Serve.git
cd dllm-serve

# Install build dependencies for flash-attn
pip install numpy psutil ninja packaging wheel

# Install dLLM-Serve in development mode
pip install -e . --no-build-isolation
```

### Installation Troubleshooting

If you encounter issues during installation, try these steps:

**PyTorch Installation:**

Install PyTorch first with CUDA support for your system. Visit [pytorch.org](https://pytorch.org/get-started/locally/) for the correct command for your CUDA version.


**Flash Attention Installation Issues:**

Flash-attn requires compilation and may fail without proper dependencies. If you get build errors:

```bash
# Install build dependencies first
pip install numpy psutil ninja packaging wheel

# Then install flash-attn with explicit flags
pip install flash-attn --no-cache-dir --no-build-isolation

# Finally install the package
pip install -e . --no-build-isolation
```

### Download Models

Download the supported diffusion models from Hugging Face:

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

### Basic Usage

```python
from dllmserve import LLM, SamplingParams

# Initialize LLM with a diffusion model
llm = LLM("./LLaDA-8B-Instruct")

# Configure sampling for diffusion models
sampling_params = SamplingParams(
    temperature=0,
    gen_length=32,  # Generation length
    steps=32,         # Number of diffusion steps
)

# Generate (sparse_config defaults to None if not provided)
prompts = ["Tell me about machine learning?"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output['text'])
```

### Sparse Attention Usage

```python
from dllmserve import LLM, SamplingParams
from dllmserve.sparse.state import SparseConfig

# Configure sparse attention
sparse_config = SparseConfig(
    enabled=True,
    retention_ratio=0.5,  # Retain 50% of attention heads
    head_select=True       # Enable head-level selection
)

llm = LLM("./LLaDA-8B-Instruct")
sampling_params = SamplingParams(
    temperature=0,
    gen_length=128,
    steps=128,
)

prompts = ["Tell me about machine learning?"]
outputs = llm.generate(
    prompts,
    sampling_params,
    sparse_configs=sparse_config
)
```

### Server Deployment

```bash
# Set model path (optional, or edit server/app.py)
export MODEL_PATH="./LLaDA-8B-Instruct"

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 1
```

Then make requests to the API:

```python
import requests

# Submit a generation request
response = requests.post("http://localhost:8000/v1/generate", json={
    "prompt": "Tell me about machine learning",
    "temperature": 0.6,
    "gen_length": 128,
})

request_id = response.json()["request_ids"][0]

# Retrieve results (poll until ready)
result = requests.get(f"http://localhost:8000/v1/result/{request_id}")
print(result.json())
# Output: {"request_id": 0, "status": "finished", "text": "..."}

# Or submit multiple requests at once
response = requests.post("http://localhost:8000/v1/generate_batch", json={
    "prompts": ["Prompt 1", "Prompt 2", "Prompt 3"],
    "temperature": 0.0,
    "gen_length": 64,
})
request_ids = response.json()["request_ids"]
```

## Documentation

- **[Installation Guide](docs/installation.md)** - Detailed installation and setup
- **[Quick Start Tutorial](docs/quickstart.md)** - Step-by-step usage examples
- **[API Reference](docs/api_reference.md)** - Complete API documentation
- **[Sparse Attention](docs/sparse_attention.md)** - Guide to sparse attention features
- **[Diffusion Models](docs/diffusion_models.md)** - Using LLaDA and Dream models
- **[Server Deployment](docs/server_deployment.md)** - Production server setup
- **[Architecture](docs/architecture.md)** - System architecture overview

## Supported Models

### Diffusion Models
- **LLaDA-8B-Instruct** - Latent Language Diffusion Autoregressive model
- **Dream-v0-Instruct-7B** - Dream diffusion model with shifted prediction

See [examples/](examples/) for usage examples of each model type.

## Examples

The [examples/](examples/) directory contains usage examples:

- **[llada_generation.py](examples/llada_generation.py)** - LLaDA diffusion model usage
- **[dream_generation.py](examples/dream_generation.py)** - Dream model usage
- **[sparse_attention.py](examples/sparse_attention.py)** - Sparse attention examples

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Development

For development setup and guidelines, see [DEVELOPMENT.md](DEVELOPMENT.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- Yanglin Zhang
- Jiakun Fan

## Citation

If you use dLLM-Serve in your research, please cite:

```bibtex
@article{fan2025taming,
  title={Taming the Memory Footprint Crisis: System Design for Production Diffusion LLM Serving},
  author={Fan, Jiakun and Zhang, Yanglin and Li, Xiangchen and Nikolopoulos, Dimitrios S},
  journal={arXiv preprint arXiv:2512.17077},
  year={2025}
}
```

## Acknowledgments

This project was developed as part of research into efficient LLM inference with sparse attention mechanisms.
