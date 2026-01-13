# Quick Start Tutorial

This tutorial will walk you through using dLLM-Serve for diffusion language model inference.

## Basic Text Generation

### Step 1: Import Required Modules

```python
from dllmserve import LLM, SamplingParams
from transformers import AutoTokenizer
```

### Step 2: Initialize the Model

```python
path = "./LLaDA-8B-Instruct"
llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
```

### Step 3: Configure Sampling Parameters

```python
sampling_params = SamplingParams(
    temperature=0.6,              # Sampling temperature (0-1)
    gen_length=64,                # Number of tokens to generate
    steps=64,                     # Number of diffusion steps
    cfg_scale=0.0,                # Classifier-free guidance scale
    remasking="low_confidence",   # Remasking strategy
)
```

### Step 4: Format Prompts with Chat Template

```python
prompts_raw = ["Tell me about machine learning"]

# Apply chat template for instruct models
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": p}],
        tokenize=False,
        add_generation_prompt=True,
    )
    for p in prompts_raw
]
```

### Step 5: Generate Text

```python
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output['text'])
```

## Batch Generation

Generate text for multiple prompts efficiently:

```python
prompts_raw = [
    "What is the capital of France?",
    "Explain quantum computing in one sentence.",
    "Write a haiku about programming.",
]

# Format all prompts
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": p}],
        tokenize=False,
        add_generation_prompt=True,
    )
    for p in prompts_raw
]

# Generate
outputs = llm.generate(prompts, sampling_params)

for i, output in enumerate(outputs):
    print(f"Prompt {i+1}: {prompts_raw[i]}")
    print(f"Output: {output['text']}\n")
```

## Using Different Sampling Parameters

You can specify different sampling parameters for each prompt:

```python
# Create different sampling params for each prompt
list_of_sampling_params = [
    SamplingParams(temperature=0.0, gen_length=32, steps=32),
    SamplingParams(temperature=0.6, gen_length=64, steps=64),
    SamplingParams(temperature=0.8, gen_length=128, steps=128),
]

outputs = llm.generate(prompts, list_of_sampling_params)
```

## Advanced: Sparse Attention

Enable sparse attention for memory-efficient inference:

```python
from dllmserve.sparse.state import SparseConfig

sparse_config = SparseConfig(
    enabled=True,
    retention_ratio=0.5,      # Use 50% of attention heads
    delay_step=1,             # Start sparse attention from step 1
    default_block_len=32,     # Block size for caching
    head_select=True,         # Enable head-level selection
)

outputs = llm.generate(
    prompts,
    sampling_params,
    sparse_configs=sparse_config
)
```

## Complete Example

Here's a complete working example:

```python
from dllmserve import LLM, SamplingParams
from transformers import AutoTokenizer

def main():
    # Initialize
    path = "./LLaDA-8B-Instruct"
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    # Configure sampling
    sampling_params = SamplingParams(
        temperature=0.6,
        gen_length=64,
        steps=64,
        cfg_scale=0.0,
        remasking="low_confidence",
    )

    # Prepare prompts
    prompts_raw = ["What is artificial intelligence?"]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts_raw
    ]

    # Generate
    outputs = llm.generate(prompts, sampling_params)

    # Display results
    for output in outputs:
        print(output['text'])

if __name__ == "__main__":
    main()
```

## Sampling Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | 1.0 | Sampling temperature (0=deterministic, 1=creative) |
| `gen_length` | int | 128 | Number of tokens to generate |
| `steps` | int | 128 | Number of diffusion steps |
| `cfg_scale` | float | 0.0 | Classifier-free guidance scale |
| `remasking` | str | "low_confidence" | Remasking strategy: "low_confidence" or "random" |

## Common Use Cases

### 1. Factual Responses (Low Temperature)

```python
params = SamplingParams(
    temperature=0.0,
    gen_length=64,
    steps=64,
    cfg_scale=0.0,
    remasking="low_confidence",
)
```

### 2. Creative Writing (High Temperature)

```python
params = SamplingParams(
    temperature=0.8,
    gen_length=128,
    steps=128,
    cfg_scale=0.0,
    remasking="random",
)
```

### 3. Memory-Efficient (Sparse Attention)

```python
from dllmserve.sparse.state import SparseConfig

sparse_config = SparseConfig(
    enabled=True,
    retention_ratio=0.3,  # Use only 30% of heads
)

outputs = llm.generate(prompts, params, sparse_configs=sparse_config)
```

## Next Steps

- See [API Reference](api_reference.md) for complete API documentation
- See [Sparse Attention Guide](sparse_attention.md) for advanced sparse attention usage
- See [Diffusion Models Guide](diffusion_models.md) for model-specific information
