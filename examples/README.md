# Examples

This directory contains usage examples for dLLM-Serve across different model types.

## Available Examples

### Basic Usage

**[basic_usage.py](basic_usage.py)** - Basic diffusion model text generation

Demonstrates:
- Initializing a diffusion LLM
- Configuring sampling parameters for diffusion models
- Applying chat templates for instruction-tuned models
- Generating text completions

```bash
python examples/basic_usage.py
```

### LLaDA Diffusion Model

**[llada_generation.py](llada_generation.py)** - LLaDA diffusion model usage

Demonstrates:
- Using diffusion models with dLLM-Serve
- Applying chat templates for instruction-tuned models
- Multi-step diffusion generation

```bash
python examples/llada_generation.py
```

### Dream Diffusion Model

**[dream_generation.py](dream_generation.py)** - Dream model with shifted prediction

Demonstrates:
- Dream-v0-Instruct-7B model usage
- Shifted prediction mechanism
- Remasking for diffusion models
- Model-specific configuration

```bash
python examples/dream_generation.py
```

### Simple Test

**[simple_test.py](simple_test.py)** - Quick functionality test

A minimal example for quickly testing that your installation works correctly.

```bash
python examples/simple_test.py
```

## Common Usage Patterns

### Sparse Attention

Add sparse attention to any diffusion model:

```python
from dllmserve import LLM, SamplingParams
from dllmserve.sparse.state import SparseConfig
from transformers import AutoTokenizer

# Configure sparse attention
sparse_config = SparseConfig(
    enabled=True,
    retention_ratio=0.5,  # Retain 50% of heads
    delay_step=1,         # Start from step 1
    default_block_len=32, # Block size for caching
    head_select=True      # Enable head-level selection
)

path = "./LLaDA-8B-Instruct"
llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

sampling_params = SamplingParams(
    temperature=0,
    gen_length=128,
    remasking="low_confidence",
)

outputs = llm.generate(
    prompts,
    sampling_params,
    sparse_configs=sparse_config
)
```

### Tensor Parallelism

For multi-GPU inference:

```python
llm = LLM(
    "./LLaDA-8B-Instruct",
    tensor_parallel_size=2  # Use 2 GPUs
)
```

### Memory Management

For limited GPU memory:

```python
llm = LLM(
    "./Dream-v0-Instruct-7B",
    max_model_len=2048  # Reduce max sequence length
)
```

## Model Downloads

### Downloading Models

Use the Hugging Face CLI to download supported diffusion models:

```bash
# LLaDA-8B-Instruct
huggingface-cli download --resume-download GSAI-ML/LLaDA-8B-Instruct \
  --local-dir ./LLaDA-8B-Instruct \
  --local-dir-use-symlinks False

# Dream-v0-Instruct-7B
huggingface-cli download --resume-download Dream-org/Dream-v0-Instruct-7B \
  --local-dir ./Dream-v0-Instruct-7B \
  --local-dir-use-symlinks False
```

### Model Paths

You can specify model paths in two ways:

1. **Direct path in code**:
   ```python
   llm = LLM("./LLaDA-8B-Instruct")
   ```

2. **Environment variable**:
   ```bash
   export MODEL_PATH="./LLaDA-8B-Instruct"
   python examples/basic_usage.py
   ```

## Troubleshooting

### CUDA Out of Memory

If you encounter OOM errors:

1. Reduce `max_model_len`:
   ```python
   llm = LLM(model_path, max_model_len=1024)
   ```

2. Use sparse attention:
   ```python
   sparse_config = SparseConfig(enabled=True, retention_ratio=0.5)
   ```

3. Use tensor parallelism across multiple GPUs

### Import Errors

Ensure dLLM-Serve is installed:

```bash
pip install -e .  # From repository root
```

### Model Not Found

Check that the model path is correct and the model files exist:

```bash
ls -la ./LLaDA-8B-Instruct/
```

## Next Steps

- See [../docs/](../docs/) for detailed documentation
- See [../benchmarks/](../benchmarks/) for performance benchmarking
- See [CONTRIBUTING.md](../CONTRIBUTING.md) to contribute your own examples
