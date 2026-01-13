#!/usr/bin/env python3
"""
Memory management tests for dllm-serve.
Tests for memory leaks, GPU memory management, and resource cleanup.
"""
import os
import gc
import warnings
import torch
from dllmserve import LLM, SamplingParams
from transformers import AutoTokenizer

# Suppress deprecation warnings from transformers
warnings.filterwarnings("ignore", message="`torch_dtype` is deprecated")


def get_gpu_memory_usage():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def test_memory_management():
    """Test memory management and leak detection."""
    print("=" * 80)
    print("Memory Management Test Suite")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU memory tests")
        return True

    MODEL_PATH = os.environ.get("LLADA_MODEL_PATH", "llada-instruct")

    test_results = []

    # Test 1: Memory usage after initialization
    print("-" * 60)
    print("Test 1: Model Initialization Memory")
    print("-" * 60)
    try:
        torch.cuda.empty_cache()
        gc.collect()
        initial_memory = get_gpu_memory_usage()
        print(f"Initial GPU memory: {initial_memory:.2f} MB")

        llm = LLM(MODEL_PATH, enforce_eager=True, tensor_parallel_size=1)
        after_init_memory = get_gpu_memory_usage()
        print(f"After init GPU memory: {after_init_memory:.2f} MB")
        print(f"Model memory footprint: {after_init_memory - initial_memory:.2f} MB")

        assert after_init_memory > initial_memory, "Model should use GPU memory"
        test_results.append(("Model initialization memory", True))
    except Exception as e:
        print(f"✗ Failed: {e}")
        test_results.append(("Model initialization memory", False))
        return False

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Test 2: Memory stability across multiple generations
    print("\n" + "-" * 60)
    print("Test 2: Memory Stability (10 generations)")
    print("-" * 60)
    try:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Test"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        params = SamplingParams(
            temperature=0.0, gen_length=16, steps=16, cfg_scale=0.0
        )

        memory_samples = []
        for i in range(10):
            before = get_gpu_memory_usage()
            _ = llm.generate([prompt], params, None)
            after = get_gpu_memory_usage()
            memory_samples.append(after - before)

            if i % 3 == 0:
                print(f"  Generation {i+1}: {after - before:+.2f} MB delta")

        # Check memory doesn't grow significantly
        avg_delta = sum(memory_samples) / len(memory_samples)
        max_delta = max(memory_samples)

        print(f"Average memory delta: {avg_delta:+.2f} MB")
        print(f"Max memory delta: {max_delta:+.2f} MB")

        # Allow some variation but not continuous growth
        assert max_delta < 100, f"Memory delta too large: {max_delta} MB"

        test_results.append(("Memory stability", True))
    except Exception as e:
        print(f"✗ Failed: {e}")
        test_results.append(("Memory stability", False))

    # Test 3: Batch processing memory
    print("\n" + "-" * 60)
    print("Test 3: Batch Processing Memory")
    print("-" * 60)
    try:
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": f"Test {i}"}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for i in range(5)
        ]
        params = SamplingParams(
            temperature=0.0, gen_length=16, steps=16, cfg_scale=0.0
        )

        before_batch = get_gpu_memory_usage()
        _ = llm.generate(prompts, [params] * len(prompts), None)
        after_batch = get_gpu_memory_usage()

        batch_memory = after_batch - before_batch
        print(f"Batch memory delta: {batch_memory:+.2f} MB")

        # Batch should not use excessive memory
        assert abs(batch_memory) < 500, f"Batch memory usage too high: {batch_memory} MB"

        test_results.append(("Batch processing memory", True))
    except Exception as e:
        print(f"✗ Failed: {e}")
        test_results.append(("Batch processing memory", False))

    # Test 4: Memory cleanup after cache clear
    print("\n" + "-" * 60)
    print("Test 4: Cache Cleanup")
    print("-" * 60)
    try:
        before_clear = get_gpu_memory_usage()
        torch.cuda.empty_cache()
        gc.collect()
        after_clear = get_gpu_memory_usage()

        freed_memory = before_clear - after_clear
        print(f"Memory before clear: {before_clear:.2f} MB")
        print(f"Memory after clear: {after_clear:.2f} MB")
        print(f"Freed memory: {freed_memory:.2f} MB")

        test_results.append(("Cache cleanup", True))
    except Exception as e:
        print(f"✗ Failed: {e}")
        test_results.append(("Cache cleanup", False))

    # Test 5: Peak memory usage
    print("\n" + "-" * 60)
    print("Test 5: Peak Memory Usage")
    print("-" * 60)
    try:
        torch.cuda.reset_peak_memory_stats()
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Generate a long response"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        params = SamplingParams(
            temperature=0.0, gen_length=128, steps=128, cfg_scale=0.0
        )

        _ = llm.generate([prompt], params, None)

        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        current_memory = get_gpu_memory_usage()

        print(f"Peak memory: {peak_memory:.2f} MB")
        print(f"Current memory: {current_memory:.2f} MB")
        print(f"Peak overhead: {peak_memory - current_memory:.2f} MB")

        test_results.append(("Peak memory tracking", True))
    except Exception as e:
        print(f"✗ Failed: {e}")
        test_results.append(("Peak memory tracking", False))

    # Test 6: Memory with sparse attention
    print("\n" + "-" * 60)
    print("Test 6: Sparse Attention Memory")
    print("-" * 60)
    try:
        from dllmserve.sparse.state import SparseConfig

        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Test"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        params = SamplingParams(
            temperature=0.0, gen_length=32, steps=32, cfg_scale=0.0
        )

        # Dense
        torch.cuda.empty_cache()
        before_dense = get_gpu_memory_usage()
        _ = llm.generate([prompt], params, None)
        after_dense = get_gpu_memory_usage()
        dense_delta = after_dense - before_dense

        # Sparse
        torch.cuda.empty_cache()
        sparse_config = SparseConfig(
            enabled=True, retention_ratio=0.5, default_block_len=16
        )
        before_sparse = get_gpu_memory_usage()
        _ = llm.generate([prompt], params, sparse_config)
        after_sparse = get_gpu_memory_usage()
        sparse_delta = after_sparse - before_sparse

        print(f"Dense memory delta: {dense_delta:+.2f} MB")
        print(f"Sparse memory delta: {sparse_delta:+.2f} MB")
        print(f"Memory saved with sparse: {dense_delta - sparse_delta:+.2f} MB")

        test_results.append(("Sparse attention memory", True))
    except Exception as e:
        print(f"✗ Failed: {e}")
        test_results.append(("Sparse attention memory", False))

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    passed = sum(1 for _, success in test_results if success)
    total = len(test_results)
    print(f"Passed: {passed}/{total}")

    for test_name, success in test_results:
        status = "✓" if success else "✗"
        print(f"  {status} {test_name}")

    # Final memory stats
    print("\nFinal GPU Memory Stats:")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
    print(f"  Reserved: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB")
    print(f"  Max allocated: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB")

    print("=" * 80)

    return all(success for _, success in test_results)


if __name__ == "__main__":
    success = test_memory_management()
    exit(0 if success else 1)
