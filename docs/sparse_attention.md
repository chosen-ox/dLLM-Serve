# Sparse Attention Guide

This guide explains how to use sparse attention for memory-efficient inference in dLLM-Serve.

## Overview

Sparse attention in dLLM-Serve uses **head-level sparsity** to reduce memory usage and computation while maintaining generation quality. By selectively using only a subset of attention heads, you can:

- Reduce GPU memory usage
- Enable larger batch sizes
- Support longer sequences
- Maintain good generation quality

## Basic Usage

### Enabling Sparse Attention

```python
from dllmserve import LLM, SamplingParams
from dllmserve.sparse.state import SparseConfig

# Configure sparse attention
sparse_config = SparseConfig(
    enabled=True,
    retention_ratio=0.5,  # Use 50% of attention heads
)

# Use with generate()
llm = LLM("./LLaDA-8B-Instruct", enforce_eager=True)
params = SamplingParams(temperature=0.6, gen_length=64, steps=64, cfg_scale=0.0, remasking="low_confidence")
outputs = llm.generate(prompts, params, sparse_configs=sparse_config)
```

## SparseConfig Parameters

### enabled (bool)

Enable or disable sparse attention.

```python
sparse_config = SparseConfig(enabled=True)
```

### retention_ratio (float)

Fraction of attention heads to retain (0.0 to 1.0).

- Lower values: More memory savings, potential quality degradation
- Higher values: Less memory savings, better quality
- Recommended range: 0.3 to 0.7

```python
# Aggressive sparsity (30% of heads)
sparse_config = SparseConfig(retention_ratio=0.3)

# Moderate sparsity (50% of heads)
sparse_config = SparseConfig(retention_ratio=0.5)

# Conservative sparsity (70% of heads)
sparse_config = SparseConfig(retention_ratio=0.7)
```

### delay_step (int)

Step before sparse attention kicks in.

- Allows full attention for initial steps
- Gradual transition to sparse attention
- Recommended: 1 to 10

```python
# Start sparse from step 1
sparse_config = SparseConfig(delay_step=1)

# Use full attention for first 5 steps
sparse_config = SparseConfig(delay_step=5)
```

### default_block_len (int)

Block size for caching in sparse attention.

- Controls granularity of sparse computation
- Larger blocks: Less overhead, less flexibility
- Smaller blocks: More overhead, more flexibility
- Recommended: 16 to 64

```python
sparse_config = SparseConfig(default_block_len=32)
```

### head_select (bool)

Enable head-level selection mechanism.

```python
sparse_config = SparseConfig(head_select=True)
```

## Complete Example

```python
from dllmserve import LLM, SamplingParams
from dllmserve.sparse.state import SparseConfig
from transformers import AutoTokenizer

def main():
    # Initialize model
    path = "./LLaDA-8B-Instruct"
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    # Configure sparse attention
    sparse_config = SparseConfig(
        enabled=True,
        retention_ratio=0.5,      # Use 50% of attention heads
        delay_step=1,             # Start from step 1
        default_block_len=32,     # Block size 32
        head_select=True,         # Enable head selection
    )

    # Configure sampling
    sampling_params = SamplingParams(
        temperature=0.6,
        gen_length=128,
        steps=128,
        cfg_scale=0.0,
        remasking="low_confidence",
    )

    # Prepare prompts
    prompts_raw = ["Explain quantum computing in simple terms."]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts_raw
    ]

    # Generate with sparse attention
    outputs = llm.generate(
        prompts,
        sampling_params,
        sparse_configs=sparse_config
    )

    for output in outputs:
        print(output['text'])

if __name__ == "__main__":
    main()
```

## Per-Prompt Sparse Configs

You can specify different sparse configs for each prompt:

```python
# Different sparsity for different prompts
sparse_configs = [
    SparseConfig(retention_ratio=0.3),  # High sparsity
    SparseConfig(retention_ratio=0.7),  # Low sparsity
    SparseConfig(enabled=False),        # No sparsity
]

outputs = llm.generate(prompts, params, sparse_configs=sparse_configs)
```

## Performance vs Quality Trade-offs

### Memory Reduction

Approximate memory reduction by retention ratio:

```
Memory at 0.5 ratio ≈ 50% memory reduction
Memory at 0.3 ratio ≈ 70% memory reduction
```

### Quality Impact

Quality impact depends on the task:

- **Factual responses** (low temp): Minimal impact even at 0.3 ratio
- **Creative writing** (high temp): Moderate impact at 0.3 ratio
- **Code generation**: Moderate impact, recommended 0.5+ ratio

### Recommended Configurations

#### Maximum Quality (No Sparsity)

```python
sparse_config = SparseConfig(enabled=False)
```

#### Balanced (Recommended)

```python
sparse_config = SparseConfig(
    enabled=True,
    retention_ratio=0.5,
    delay_step=1,
    default_block_len=32,
    head_select=True,
)
```

#### Maximum Memory Savings

```python
sparse_config = SparseConfig(
    enabled=True,
    retention_ratio=0.3,
    delay_step=5,  # Full attention for first 5 steps
    default_block_len=32,
    head_select=True,
)
```

## Benchmarking

Compare performance with different sparsity settings:

```python
from dllmserve import LLM, SamplingParams
from dllmserve.sparse.state import SparseConfig
import time

prompts = ["Test prompt"] * 10
params = SamplingParams(temperature=0.0, gen_length=64, steps=64, cfg_scale=0.0, remasking="low_confidence")

ratios = [1.0, 0.7, 0.5, 0.3]

for ratio in ratios:
    sparse_config = SparseConfig(retention_ratio=ratio)
    llm = LLM("./LLaDA-8B-Instruct", enforce_eager=True)

    start = time.time()
    outputs = llm.generate(prompts, params, sparse_configs=sparse_config)
    elapsed = time.time() - start

    print(f"Ratio {ratio}: {elapsed:.2f}s, {len(prompts)/elapsed:.2f} req/s")
    del llm
```

## Best Practices

1. **Start with retention_ratio=0.5**: Good balance between memory and quality
2. **Use delay_step for complex tasks**: Allow full attention for initial steps
3. **Monitor GPU memory**: Adjust ratio based on available memory
4. **Test quality**: Validate output quality for your specific use case
5. **Batch with same config**: Group similar sparsity requirements together

## Troubleshooting

### Quality Degradation

If output quality is poor:

1. Increase `retention_ratio` (try 0.7)
2. Increase `delay_step` (try 5-10)
3. Disable sparsity for comparison

### No Memory Savings

If you don't see memory reduction:

1. Ensure `enabled=True`
2. Check `retention_ratio` is less than 1.0
3. Verify you're not memory-bound elsewhere

### Slow Inference

If inference is slow with sparsity:

1. Increase `default_block_len` (try 64)
2. Reduce `retention_ratio` (try 0.3)
3. Ensure `head_select=True`

## Advanced Usage

### Dynamic Sparsity

Adjust sparsity based on sequence length:

```python
def get_sparse_config(seq_len):
    if seq_len < 128:
        return SparseConfig(enabled=False)  # Short sequences: no sparsity
    elif seq_len < 512:
        return SparseConfig(retention_ratio=0.7)  # Medium: 70% heads
    else:
        return SparseConfig(retention_ratio=0.5)  # Long: 50% heads
```

### Layer-wise Sparsity (Experimental)

Different sparsity for different layers:

```python
# This is an experimental feature
sparse_config = SparseConfig(
    enabled=True,
    retention_ratio=0.5,
    layer_wise=True,  # Enable layer-wise sparsity
)
```

## Next Steps

- See [API Reference](api_reference.md) for complete SparseConfig documentation
- See [Diffusion Models Guide](diffusion_models.md) for model-specific usage
