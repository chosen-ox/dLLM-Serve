# API Reference

Complete API documentation for dLLM-Serve.

## Core Classes

### LLM

The main interface for model inference.

```python
from dllmserve import LLM
```

#### Constructor

```python
LLM(model, **kwargs)
```

**Parameters:**

- `model` (str): Path to model or HuggingFace model ID
- `enforce_eager` (bool): Disable CUDA graphs (default: False)
- `tensor_parallel_size` (int): Number of GPUs for tensor parallelism (default: 1)
- `max_model_len` (int): Maximum model sequence length (default: from config)
- `trust_remote_code` (bool): Trust remote code when loading (default: True)

**Example:**

```python
from dllmserve import LLM

llm = LLM(
    "./LLaDA-8B-Instruct",
    enforce_eager=True,
    tensor_parallel_size=1,
    max_model_len=4096,
)
```

#### Methods

##### generate()

Generate text completions for prompts.

```python
generate(
    prompts: list[str] | list[list[int]],
    sampling_params: SamplingParams | list[SamplingParams],
    sparse_configs: SparseConfig | list[SparseConfig] = None,
    use_tqdm: bool = True,
) -> list[dict]
```

**Parameters:**

- `prompts`: List of text prompts or tokenized prompts
- `sampling_params`: Single params or list of params (one per prompt)
- `sparse_configs`: Optional sparse config or list of configs (one per prompt)
- `use_tqdm`: Show progress bar (default: True)

**Returns:**

List of dictionaries with keys:
- `text`: Generated text string
- `token_ids`: List of generated token IDs

**Example:**

```python
from dllmserve import LLM, SamplingParams

llm = LLM("./LLaDA-8B-Instruct", enforce_eager=True)
params = SamplingParams(temperature=0.6, gen_length=64, steps=64, cfg_scale=0.0, remasking="low_confidence")

outputs = llm.generate(["Hello, world!"], params)
print(outputs[0]['text'])
```

### SamplingParams

Parameters for controlling text generation.

```python
from dllmserve import SamplingParams
```

#### Constructor

```python
SamplingParams(
    temperature: float = 1.0,
    max_tokens: int = 64,
    ignore_eos: bool = False,
    gen_length: int = 128,
    steps: int = 128,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | 1.0 | Sampling temperature (0=deterministic, higher=more creative) |
| `max_tokens` | int | 64 | Maximum tokens to generate (autoregressive only) |
| `ignore_eos` | bool | False | Ignore end-of-sequence token |
| `gen_length` | int | 128 | Generation length in tokens (diffusion models) |
| `steps` | int | 128 | Number of diffusion steps (diffusion models) |
| `cfg_scale` | float | 0.0 | Classifier-free guidance scale (diffusion models) |
| `remasking` | str | "low_confidence" | Remasking strategy: "low_confidence" or "random" |

**Validation:**

- `gen_length` and `steps` must be positive integers ≤ 8192
- `remasking` must be "low_confidence" or "random"

**Example:**

```python
from dllmserve import SamplingParams

# Diffusion model parameters
params = SamplingParams(
    temperature=0.6,
    gen_length=128,
    steps=128,
    cfg_scale=0.0,
    remasking="low_confidence",
)
```

### SparseConfig

Configuration for sparse attention (head-level sparsity).

```python
from dllmserve.sparse.state import SparseConfig
```

#### Constructor

```python
SparseConfig(
    enabled: bool = True,
    retention_ratio: float = 0.5,
    delay_step: int = 1,
    default_block_len: int = 32,
    head_select: bool = True,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | True | Enable sparse attention |
| `retention_ratio` | float | 0.5 | Fraction of attention heads to retain (0-1) |
| `delay_step` | int | 1 | Step before sparsity kicks in |
| `default_block_len` | int | 32 | Block size for caching |
| `head_select` | bool | True | Enable head-level selection |

**Example:**

```python
from dllmserve.sparse.state import SparseConfig

sparse_config = SparseConfig(
    enabled=True,
    retention_ratio=0.5,  # Use 50% of heads
    delay_step=1,
    default_block_len=32,
    head_select=True,
)
```

## Server API

### Endpoints

#### POST /v1/generate

Submit a single generation request.

**Request:**

```json
{
  "prompt": "Tell me about AI",
  "temperature": 0.6,
  "gen_length": 128
}
```

**Response:**

```json
{
  "request_ids": [0],
  "status": "submitted"
}
```

#### POST /v1/generate_batch

Submit multiple generation requests.

**Request:**

```json
{
  "prompts": ["Prompt 1", "Prompt 2"],
  "temperature": 0.0,
  "gen_length": 64
}
```

**Response:**

```json
{
  "request_ids": [0, 1],
  "status": "submitted"
}
```

#### GET /v1/result/{request_id}

Retrieve generation result.

**Response (running):**

```json
{
  "request_id": 0,
  "status": "running",
  "text": null
}
```

**Response (finished):**

```json
{
  "request_id": 0,
  "status": "finished",
  "text": "Generated text here..."
}
```

#### GET /v1/health

Health check endpoint.

**Response:**

```json
{
  "status": "ok"
}
```

## Usage Examples

### Basic Generation

```python
from dllmserve import LLM, SamplingParams

llm = LLM("./LLaDA-8B-Instruct", enforce_eager=True)
params = SamplingParams(temperature=0.6, gen_length=64, steps=64, cfg_scale=0.0, remasking="low_confidence")
outputs = llm.generate(["Hello, world!"], params)
```

### Batch Generation

```python
prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
params = SamplingParams(temperature=0.0, gen_length=32, steps=32, cfg_scale=0.0, remasking="low_confidence")
outputs = llm.generate(prompts, params)
```

### Per-Prompt Sampling Parameters

```python
params_list = [
    SamplingParams(temperature=0.0, gen_length=32, steps=32, cfg_scale=0.0, remasking="low_confidence"),
    SamplingParams(temperature=0.6, gen_length=64, steps=64, cfg_scale=0.0, remasking="low_confidence"),
]
outputs = llm.generate(prompts, params_list)
```

### Sparse Attention

```python
from dllmserve.sparse.state import SparseConfig

sparse_config = SparseConfig(enabled=True, retention_ratio=0.5)
outputs = llm.generate(prompts, params, sparse_configs=sparse_config)
```

### Multi-GPU Inference

```python
llm = LLM("./LLaDA-8B-Instruct", tensor_parallel_size=2)
```

## Return Values

The `generate()` method returns a list of dictionaries:

```python
[
    {
        "text": "Generated text string",
        "token_ids": [101, 102, 103, ...]  # Token IDs
    },
    ...
]
```

## Error Handling

### ValueError

Raised when sampling parameters are invalid:

```python
# This will raise ValueError
params = SamplingParams(gen_length=10000)  # Exceeds maximum
```

### CUDA Out of Memory

Reduce memory usage by:

1. Reducing `max_model_len`
2. Using sparse attention
3. Enabling tensor parallelism

```python
llm = LLM(
    model_path,
    max_model_len=2048,  # Reduce sequence length
    tensor_parallel_size=2,  # Use 2 GPUs
)
```

## Next Steps

- See [Sparse Attention Guide](sparse_attention.md) for advanced sparsity usage
- See [Server Deployment](server_deployment.md) for production setup
