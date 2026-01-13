# Architecture Overview

This document provides an architectural overview of dLLM-Serve, a lightweight diffusion LLM serving framework.

## System Design

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         User Layer                           │
├─────────────────────────────────────────────────────────────┤
│  Python API (LLM class)  │  REST API (FastAPI Server)      │
└──────────────┬───────────────────────────┬─────────────────┘
               │                           │
               ▼                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      LLMEngine Layer                         │
├─────────────────────────────────────────────────────────────┤
│  - Request Management      │  - Tokenization               │
│  - Batch Orchestration     │  - Progress Tracking          │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│                      Scheduler Layer                        │
├─────────────────────────────────────────────────────────────┤
│  - Request Scheduling       │  - Batching Logic             │
│  - Sequence Tracking        │  - Prefill/Decode States      │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│                    ModelRunner Layer                        │
├─────────────────────────────────────────────────────────────┤
│  - Model Execution          │  - CUDA Kernels               │
│  - Tensor Parallelism       │  - Memory Management          │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│                      Model Layer                             │
├─────────────────────────────────────────────────────────────┤
│  - Attention Layers         │  - Sparse Attention           │
│  - Linear Layers            │  - Layer Normalization        │
│  - Positional Embeddings    │  - Transformer Blocks         │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### LLM (dllmserve/llm.py)

Thin wrapper around LLMEngine for user-facing API.

```python
class LLM(LLMEngine):
    pass
```

**Responsibilities:**
- Provide clean user interface
- Inherit all functionality from LLMEngine

### LLMEngine (dllmserve/engine/llm_engine.py)

Main inference engine orchestrator.

**Key Responsibilities:**
- Model initialization and configuration
- Request lifecycle management
- Batch generation coordination
- Multi-GPU coordination

**Key Methods:**
- `__init__()`: Initialize model, tokenizer, scheduler
- `add_request()`: Add a single generation request
- `generate()`: Main entry point for batch generation
- `step()`: Execute one generation step

**Architecture Flow:**

```
generate()
    │
    ├──> add_request() [for each prompt]
    │     │
    │     ├──> Create Sequence
    │     └──> scheduler.add()
    │
    └──> while not finished:
          ├──> scheduler.schedule() → [seqs, is_prefill]
          ├──> model_runner.call("run", seqs, is_prefill)
          ├──> scheduler.postprocess(seqs, token_ids)
          └──> Return finished sequences
```

### Scheduler (dllmserve/engine/scheduler.py)

Manages request scheduling and batching.

**Responsibilities:**
- Maintain request queue
- Schedule requests for execution
- Manage prefill vs decode batches
- Track sequence states

**Batching Strategy:**
- Separate prefill and decode batches
- Maximize throughput via batching
- Prioritize prefill for faster first-token latency

**Sequence States:**
1. **Waiting**: Queued but not yet scheduled
2. **Prefill**: Processing prompt (initial forward pass)
3. **Decode**: Generating tokens (iterative forward passes)
4. **Finished**: Generation complete

### ModelRunner (dllmserve/engine/model_runner.py)

Executes model forward passes and CUDA kernels.

**Responsibilities:**
- Load and execute model
- Manage CUDA memory
- Handle tensor parallelism
- Execute custom CUDA kernels

**Execution Modes:**
- **Prefill**: Process full prompt in parallel
- **Decode**: Generate tokens iteratively

**Tensor Parallelism:**
- Split model across multiple GPUs
- Synchronize attention outputs
- Reduce communication overhead

### BlockManager (dllmserve/engine/block_manager.py)

Manages KV cache memory allocation.

**Responsibilities:**
- Allocate memory blocks for KV cache
- Track block usage across sequences
- Support prefix caching
- Handle memory defragmentation

**Block Allocation:**
- Fixed-size blocks (e.g., 32 tokens per block)
- Linked list structure for sequences
- Reference counting for shared blocks

**Prefix Caching:**
- Share common prefix blocks
- Avoid redundant computation
- Reduce memory usage

### Sequence (dllmserve/engine/sequence.py)

Data structures for generation sequences.

**Key Classes:**
- `Sequence`: Single generation request
- `SequenceStatus`: State machine for sequence lifecycle
- `ModelType`: Enum for model types (LLaDA, Dream)

**Sequence Lifecycle:**

```
Created → Waiting → Prefill → Decode → Finished
                        │         │
                        └────┬────┘
                             │
                    (Can be preempted)
```

## Sparse Attention

### Overview

Sparse attention reduces memory and computation by using only a subset of attention heads.

### SparseConfig (dllmserve/sparse/state.py)

Configuration for sparse attention.

**Key Parameters:**
- `enabled`: Enable/disable sparsity
- `retention_ratio`: Fraction of heads to retain
- `delay_step`: Step before sparsity activates
- `default_block_len`: Block size for caching
- `head_select`: Enable head-level selection

### Sparse Attention Flow

```
1. Initialize SparseConfig with desired parameters
2. Pass to LLM.generate() via sparse_configs
3. Scheduler attaches config to each Sequence
4. ModelRunner applies sparsity during execution:
   - Select subset of attention heads
   - Cache sparse attention patterns
   - Execute sparse matrix operations
```

### Memory Efficiency

Sparse attention provides approximate memory reduction:

```
Memory Reduction ≈ (1 - retention_ratio) × Attention_Memory
```

Example:
- At `retention_ratio=0.5`: ~50% attention memory reduction
- At `retention_ratio=0.3`: ~70% attention memory reduction

## Model Types

### ModelType Detection

Model type is auto-detected from config during initialization:

```python
# In config.py
if hasattr(config, "uses_shifted_prediction"):
    model_type = ModelType.DREAM
elif hasattr(config, "d_model"):
    model_type = ModelType.LLADA
else:
    model_type = ModelType.AUTOREGRESSIVE
```

### LLaDA Model

- **Prediction**: Direct (h[i] → token[i])
- **Config Format**: `d_model`, `n_heads`
- **Special Tokens**: mask_token_id=126336, pad_token_id=126081

### Dream Model

- **Prediction**: Shifted (h[i] → token[i+1])
- **Config Format**: `hidden_size`, `num_attention_heads`
- **Special Tokens**: mask_token_id=151666, pad_token_id=151643

## Key Design Decisions

### 1. Offline-First Design

- Optimized for batch processing
- Minimize online serving overhead
- Maximize throughput over latency

### 2. Clean Codebase

- ~1,200 lines of core Python code
- Readable and maintainable
- Well-documented and tested

### 3. Modular Architecture

- Clear separation of concerns
- Easy to extend and modify
- Pluggable components

### 4. GPU Optimization

- Custom CUDA kernels
- Tensor parallelism
- Efficient memory management
- CUDA graphs support

### 5. Sparse Attention

- Head-level sparsity
- Configurable retention ratios
- Block-based caching
- Prefix caching support

## Performance Characteristics

### Throughput Optimization

- **Batching**: Process multiple requests simultaneously
- **Tensor Parallelism**: Distribute model across GPUs
- **Sparse Attention**: Reduce memory and computation
- **CUDA Graphs**: Reduce kernel launch overhead

### Latency Optimization

- **Prefill Priority**: Process prompts quickly
- **Efficient Batching**: Minimize batch formation time
- **CUDA Graphs**: Reduce kernel launch overhead
- **Fast Attention**: GPU-optimized attention kernels

### Memory Optimization

- **Sparse Attention**: Reduce attention memory
- **Prefix Caching**: Share common prompt KV cache
- **Block Management**: Efficient KV cache allocation
- **Tensor Parallelism**: Distribute memory across GPUs

## Extension Points

### Adding New Models

1. Add model type to `ModelType` enum
2. Implement model-specific logic in `config.py`
3. Add model detection logic
4. Implement any model-specific layers

### Adding New Sampling Strategies

1. Extend `SamplingParams` dataclass
2. Implement strategy in `ModelRunner`
3. Update scheduler logic if needed

### Adding New Sparsity Methods

1. Extend `SparseConfig` dataclass
2. Implement sparsity logic in attention layers
3. Update `ModelRunner` to use new sparsity

### Adding New Server Endpoints

1. Add endpoint to `server/app.py`
2. Implement request/response models
3. Update `EngineWorker` if needed

## Performance Benchmarks

See [benchmarks/](../benchmarks/) for comprehensive performance data.

### Typical Performance

LLaDA-8B-Instruct on A100 GPU:
- **Throughput**: ~50-100 tokens/second
- **Latency**: ~500ms first token, ~20ms subsequent tokens
- **Memory**: ~16GB for full model, ~8GB with sparse attention (50% ratio)

Dream-v0-Instruct-7B on A100 GPU:
- **Throughput**: ~60-120 tokens/second
- **Latency**: ~400ms first token, ~15ms subsequent tokens
- **Memory**: ~14GB for full model, ~7GB with sparse attention (50% ratio)

## Code Organization

```
dllmserve/
├── llm.py                  # Main LLM interface
├── config.py               # Configuration for model types
├── sampling_params.py      # Sampling parameters
├── engine/
│   ├── llm_engine.py       # Main inference engine
│   ├── model_runner.py     # Model execution
│   ├── scheduler.py        # Request scheduling
│   ├── block_manager.py    # KV cache management
│   └── sequence.py         # Sequence data structures
├── layers/
│   ├── attention.py        # Attention mechanisms
│   ├── linear.py           # Linear layers
│   ├── layernorm.py        # Layer normalization
│   └── rotary_embedding.py # RoPE embeddings
├── sparse/
│   └── state.py            # SparseConfig
└── utils/
    └── ...                 # Utility functions
```

## Next Steps

- See [API Reference](api_reference.md) for usage details
- See [Sparse Attention Guide](sparse_attention.md) for optimization
- See [Diffusion Models Guide](diffusion_models.md) for model details
