# dLLM-Serve Test Suite

Comprehensive test suite for dllm-serve, covering unit tests, integration tests, and performance tests.

## Quick Start

### Run All Tests

```bash
python tests/run_all_tests.py
```

### Run Individual Test Suites

```bash
# Unit tests - Model-specific tests
python tests/unit/test_llada_sparse.py
python tests/unit/test_dream_sparse.py

# Integration tests - Inference tests
python tests/integration/test_inference_dream.py
python tests/integration/test_inference_llada.py
python tests/integration/test_server.py

# Performance tests
python tests/performance/test_log_optimize.py
```

## Test Organization

### Unit Tests (`tests/unit/`)

**test_llada_sparse.py** - LLaDA model sparse attention
- Head-level sparsity with LLaDA
- Multiple retention ratios
- Performance comparison vs dense
- Output validation with assertions
- Memory cleanup between runs

**test_dream_sparse.py** - Dream model sparse attention
- Head-level sparsity with Dream
- Multiple retention ratios
- Dense vs sparse comparison
- Output quality validation
- Memory cleanup between runs

### Integration Tests (`tests/integration/`)

**test_inference_dream.py** - Dream model inference
- End-to-end Dream model testing
- Model property detection
- Sampling parameters validation
- Sparse configuration handling
- Edge cases and error handling

**test_inference_llada.py** - LLaDA model inference
- End-to-end LLaDA model testing
- Batch generation handling
- Different remasking strategies
- Variable generation lengths
- Edge cases and error handling

**test_server.py** - Server integration
- FastAPI server startup
- Request submission
- Async result polling
- Health checks

### Performance Tests (`tests/performance/`)

**test_log_optimize.py** - Log optimization levels
- Different optimization strategies (level 0, 1, 2)
- Performance impact measurement
- Memory usage comparison

## Environment Variables

Set these environment variables before running tests:

```bash
# LLaDA model path (defaults to "llada-instruct")
export LLADA_MODEL_PATH="./llada-instruct/"

# Dream model path (defaults to "Dream-v0-Instruct-7B")
export DREAM_MODEL_PATH="$HOME/huggingface/Dream-v0-Instruct-7B/"
```

## Test Requirements

### Required
- Python 3.10+
- PyTorch 2.4+ with CUDA support
- dllm-serve installed (`pip install -e .`)
- At least one diffusion model (LLaDA or Dream)

### Optional
- pytest (for advanced test features)
- Multiple GPUs (for tensor parallelism tests)

## Memory Management

To avoid OOM errors when running multiple tests sequentially, each test includes proper cleanup:

```python
# Cleanup: explicitly delete LLM to free GPU memory
# Note: exit() is automatically called by atexit handler
del llm
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
```

**Important:** Run LLaDA and Dream tests in separate processes/sessions to avoid OOM:

```bash
# Terminal 1 - Test LLaDA
python tests/unit/test_llada_sparse.py

# Terminal 2 - Test Dream (separate session)
python tests/unit/test_dream_sparse.py
```

## Writing New Tests

### Test File Structure

```python
#!/usr/bin/env python3
"""
Brief description of what this test suite covers.
"""
import gc
import os
import time
import torch
from dllmserve import LLM, SamplingParams
from dllmserve.sparse.state import SparseConfig


def test_feature_name():
    """Test description."""
    print("\n" + "=" * 70)
    print("Test Suite Name")
    print("=" * 70)

    MODEL_PATH = os.environ.get("MODEL_PATH", "default-path")

    # Ensure GPU memory is clean
    torch.cuda.empty_cache()

    llm = LLM(MODEL_PATH, enforce_eager=True, tensor_parallel_size=1)
    print(f"✓ Model initialized: {MODEL_PATH}")

    # Test code here
    params = SamplingParams(temperature=0.0, gen_length=64, steps=64)

    torch.cuda.synchronize()
    start = time.time()
    outputs = llm.generate(prompts, [params], use_tqdm=False)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    # Validation
    assert len(outputs) == len(prompts)
    for i, out in enumerate(outputs):
        assert len(out["text"]) > 10, f"Output {i} too short"
        print(f"  Output {i+1}: {out['text'][:100]}...")

    throughput = len(prompts) * params.gen_length / elapsed
    print(f"✓ Throughput: {throughput:.1f} tokens/s, Time: {elapsed:.2f}s")
    print("✓ All outputs valid")

    # Cleanup
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print("✓ Model cleaned up from GPU memory")


def main():
    print("=" * 70)
    print("Test Suite Name")
    print("=" * 70)

    test_feature_name()

    print("\n" + "=" * 70)
    print("All tests PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

### Best Practices

1. **Always validate outputs**: Check for non-empty, meaningful text
2. **Use assertions**: Add explicit checks for expected behavior
3. **Provide clear output**: Use descriptive print statements with ✓ markers
4. **Clean up resources**: Always include cleanup code with `gc.collect()` and `torch.cuda.empty_cache()`
5. **Measure performance**: Include throughput and timing metrics
6. **Handle memory**: Use `torch.cuda.reset_peak_memory_stats()` for accurate tracking
7. **Document intent**: Add docstrings and comments
8. **Separate models**: Test LLaDA and Dream in separate files to avoid OOM

## Test Coverage

Current test coverage includes:

- ✓ LLaDA model sparse attention
- ✓ Dream model sparse attention
- ✓ LLaDA model inference
- ✓ Dream model inference
- ✓ Server integration
- ✓ Log optimization levels
- ✓ Memory management and cleanup
- ✓ Performance benchmarking
- ✓ Multiple remasking strategies
- ✓ Edge cases and error handling

## Troubleshooting

### Out of Memory Errors

**Run tests in separate sessions:**
```bash
# Session 1 - Test LLaDA only
export LLADA_MODEL_PATH=/path/to/llada
python tests/unit/test_llada_sparse.py

# Session 2 - Test Dream only (after LLaDA completes)
export DREAM_MODEL_PATH=/path/to/dream
python tests/unit/test_dream_sparse.py
```

**Reduce generation parameters:**
```python
params = SamplingParams(gen_length=32, steps=32)  # Smaller
```

### Model Not Found

Check environment variables:
```bash
echo $LLADA_MODEL_PATH
echo $DREAM_MODEL_PATH
```

Ensure models are downloaded:
```bash
huggingface-cli download Dream-org/Dream-v0-Instruct-7B
```

### CUDA Errors

Clear cache between tests:
```python
import torch
torch.cuda.empty_cache()
```

### Slow Tests

Tests may take several minutes depending on:
- GPU speed
- Model size
- Generation length
- Batch size

Use smaller models or reduce test parameters for faster iteration.

## CI/CD Integration

To integrate with CI/CD pipelines:

```bash
# Run LLaDA tests
export LLADA_MODEL_PATH=/path/to/llada
python tests/unit/test_llada_sparse.py

# Run Dream tests in separate stage/job
export DREAM_MODEL_PATH=/path/to/dream
python tests/unit/test_dream_sparse.py
```

## Contributing

When adding new tests:

1. Follow the existing structure
2. Use separate files for LLaDA and Dream tests
3. Include proper memory cleanup
4. Add performance metrics (throughput, time)
5. Add appropriate assertions
6. Document in this README
7. Test on clean environment

## License

Same as dllm-serve main project.
