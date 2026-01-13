# Diffusion Models Guide

This guide covers using diffusion language models (LLaDA and Dream) with dLLM-Serve.

## Overview

dLLM-Serve supports two diffusion language model families:

1. **LLaDA** - Latent Language Diffusion Autoregressive models
2. **Dream** - Dream diffusion models with shifted prediction

Diffusion models offer several advantages:
- Better controllability through sampling parameters
- Natural support for classifier-free guidance (CFG)
- Remasking strategies for quality control
- Competitive performance with autoregressive models

## LLaDA Models

### Model: LLaDA-8B-Instruct

LLaDA uses direct prediction (hidden_states[i] → token[i]).

#### Download

```bash
huggingface-cli download --resume-download GSAI-ML/LLaDA-8B-Instruct \
  --local-dir ./LLaDA-8B-Instruct \
  --local-dir-use-symlinks False
```

#### Basic Usage

```python
from dllmserve import LLM, SamplingParams
from transformers import AutoTokenizer

path = "./LLaDA-8B-Instruct"
llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

sampling_params = SamplingParams(
    temperature=0.6,
    gen_length=64,
    steps=64,
    cfg_scale=0.0,
    remasking="low_confidence",
)

prompts_raw = ["Explain machine learning."]
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": p}],
        tokenize=False,
        add_generation_prompt=True,
    )
    for p in prompts_raw
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output['text'])
```

#### LLaDA Characteristics

- **Prediction**: Direct (h[i] → token[i])
- **QKV Bias**: `bias=False`
- **Config**: `d_model`, `n_heads`
- **Positional**: `max_sequence_length`
- **Mask Token**: 126336
- **Pad Token**: 126081

### Recommended Sampling Parameters for LLaDA

#### Factual Responses

```python
SamplingParams(
    temperature=0.0,
    gen_length=64,
    steps=64,
    cfg_scale=0.0,
    remasking="low_confidence",
)
```

#### Creative Writing

```python
SamplingParams(
    temperature=0.7,
    gen_length=128,
    steps=128,
    cfg_scale=0.5,
    remasking="random",
)
```

#### High Quality with CFG

```python
SamplingParams(
    temperature=0.0,
    gen_length=64,
    steps=64,
    cfg_scale=1.0,  # Enable CFG
    remasking="low_confidence",
)
```

## Dream Models

### Model: Dream-v0-Instruct-7B

Dream uses shifted prediction (hidden_states[i] → token[i+1]).

#### Download

```bash
huggingface-cli download --resume-download Dream-org/Dream-v0-Instruct-7B \
  --local-dir ./Dream-v0-Instruct-7B \
  --local-dir-use-symlinks False
```

#### Basic Usage

```python
from dllmserve import LLM, SamplingParams
from transformers import AutoTokenizer

path = "./Dream-v0-Instruct-7B"
llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

sampling_params = SamplingParams(
    temperature=0.0,
    gen_length=64,
    steps=64,
    cfg_scale=0.0,
    remasking="low_confidence",
)

prompts_raw = ["What is AI?"]
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": p}],
        tokenize=False,
        add_generation_prompt=True,
    )
    for p in prompts_raw
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output['text'])
```

#### Dream Characteristics

- **Prediction**: Shifted (h[i] → token[i+1])
- **QKV Bias**: `bias=True`
- **Config**: `hidden_size`, `num_attention_heads`
- **Positional**: `max_position_embeddings`
- **Mask Token**: 151666
- **Pad Token**: 151643

### Recommended Sampling Parameters for Dream

#### Default (Balanced)

```python
SamplingParams(
    temperature=0.0,
    gen_length=64,
    steps=64,
    cfg_scale=0.0,
    remasking="low_confidence",
)
```

#### High Quality

```python
SamplingParams(
    temperature=0.0,
    gen_length=128,
    steps=128,
    cfg_scale=0.0,
    remasking="low_confidence",
)
```

#### Creative

```python
SamplingParams(
    temperature=0.6,
    gen_length=64,
    steps=64,
    cfg_scale=0.0,
    remasking="random",
)
```

## Comparison: LLaDA vs Dream

| Feature | LLaDA | Dream |
|---------|-------|-------|
| **Prediction** | Direct (h[i] → token[i]) | Shifted (h[i] → token[i+1]) |
| **QKV Bias** | `bias=False` | `bias=True` |
| **Config Format** | `d_model`, `n_heads` | `hidden_size`, `num_attention_heads` |
| **Positional Embeddings** | `max_sequence_length` | `max_position_embeddings` |
| **Mask Token ID** | 126336 | 151666 |
| **Pad Token ID** | 126081 | 151643 |
| **Use Case** | General purpose | Instruction following |

## Sampling Parameters

### temperature (float)

Controls randomness in generation.
- **0.0**: Deterministic, greedy sampling
- **0.5-0.7**: Balanced creativity
- **0.8-1.0**: Highly creative, diverse

### gen_length (int)

Number of tokens to generate.
- Typical range: 32-512
- Must match `steps` for best results

### steps (int)

Number of diffusion steps.
- More steps: Better quality, slower
- Fewer steps: Faster, lower quality
- Recommended: Equal to `gen_length`

### cfg_scale (float)

Classifier-free guidance scale.
- **0.0**: No CFG (default)
- **0.5-1.0**: Light guidance
- **1.5-2.0**: Strong guidance
- Higher values can improve quality but may reduce diversity

### remasking (str)

Remasking strategy for diffusion process.
- **"low_confidence"**: Remask low-confidence tokens (recommended for factual responses)
- **"random"**: Random remasking (recommended for creative writing)

## Complete Examples

### LLaDA Example

```python
from dllmserve import LLM, SamplingParams
from dllmserve.sparse.state import SparseConfig
from transformers import AutoTokenizer

path = "./LLaDA-8B-Instruct"
llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

# Configure sampling
sampling_params = SamplingParams(
    temperature=0.6,
    gen_length=128,
    steps=128,
    cfg_scale=0.0,
    remasking="low_confidence",
)

# Configure sparse attention (optional)
sparse_config = SparseConfig(
    enabled=True,
    retention_ratio=0.5,
    delay_step=1,
    default_block_len=32,
    head_select=True,
)

# Prepare prompts
prompts_raw = [
    "What is the difference between AI and machine learning?",
    "Write a short poem about programming.",
]

prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": p}],
        tokenize=False,
        add_generation_prompt=True,
    )
    for p in prompts_raw
]

# Generate
outputs = llm.generate(
    prompts,
    sampling_params,
    sparse_configs=sparse_config
)

# Display results
for i, output in enumerate(outputs):
    print(f"\nPrompt: {prompts_raw[i]}")
    print(f"Output: {output['text']}\n")
```

### Dream Example

```python
from dllmserve import LLM, SamplingParams
from transformers import AutoTokenizer

path = "./Dream-v0-Instruct-7B"
llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

# Different sampling params for each prompt
list_of_sampling_params = [
    SamplingParams(temperature=0.0, gen_length=64, steps=64, cfg_scale=0.0, remasking="low_confidence"),
    SamplingParams(temperature=0.0, gen_length=64, steps=64, cfg_scale=0.0, remasking="random"),
    SamplingParams(temperature=0.6, gen_length=64, steps=64, cfg_scale=0.0, remasking="low_confidence"),
]

# Prepare prompts
prompts_raw = [
    "What is the Dream model?",
    "Explain diffusion models.",
    "Write a haiku about AI.",
]

prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": p}],
        tokenize=False,
        add_generation_prompt=True,
    )
    for p in prompts_raw
]

# Generate with per-prompt sampling params
outputs = llm.generate(prompts, list_of_sampling_params, None)

# Display results
for i, output in enumerate(outputs):
    print(f"\nPrompt: {prompts_raw[i]}")
    print(f"Params: temp={list_of_sampling_params[i].temperature}, "
          f"remasking={list_of_sampling_params[i].remasking}")
    print(f"Output: {output['text']}\n")
```

## Troubleshooting

### Poor Quality Output

1. **Increase steps**: Try `steps=gen_length` or higher
2. **Adjust temperature**: Lower for factual, higher for creative
3. **Enable CFG**: Set `cfg_scale=0.5-1.0`
4. **Change remasking**: Try different `remasking` strategy

### Slow Generation

1. **Reduce steps**: Fewer diffusion steps
2. **Enable sparse attention**: Use `SparseConfig`
3. **Reduce gen_length**: Generate fewer tokens

### Out of Memory

1. **Enable sparse attention**: Lower `retention_ratio`
2. **Reduce gen_length**: Shorter sequences
3. **Use tensor parallelism**: `tensor_parallel_size=2`
4. **Reduce max_model_len**: Smaller context window

## Best Practices

1. **Match steps to gen_length**: `steps=gen_length` typically works best
2. **Use chat templates**: Always apply chat template for instruct models
3. **Start with low temperature**: `temperature=0.0` for factual responses
4. **Experiment with remasking**: "low_confidence" for factual, "random" for creative
5. **Enable CFG for quality**: `cfg_scale=0.5-1.0` can improve output quality

## Next Steps

- See [Quick Start Tutorial](quickstart.md) for more examples
- See [Sparse Attention Guide](sparse_attention.md) for memory optimization
- See [API Reference](api_reference.md) for complete parameter documentation
