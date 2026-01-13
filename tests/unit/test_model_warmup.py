#!/usr/bin/env python3
"""
Model warmup and initialization tests for dllm-serve.
Tests proper model initialization, warmup, and first-run behavior.
"""
import os
import time
import warnings
import torch
from dllmserve import LLM, SamplingParams
from transformers import AutoTokenizer

# Suppress deprecation warnings from transformers
warnings.filterwarnings("ignore", message="`torch_dtype` is deprecated")


def test_model_warmup():
    """Test model warmup and initialization."""
    print("=" * 80)
    print("Model Warmup and Initialization Test Suite")
    print("=" * 80)

    MODEL_PATH = os.environ.get("LLADA_MODEL_PATH", "llada-instruct")

    test_results = []

    # Test 1: Model initialization time
    print("-" * 60)
    print("Test 1: Model Initialization Time")
    print("-" * 60)
    try:
        start_time = time.time()
        llm = LLM(MODEL_PATH, enforce_eager=True, tensor_parallel_size=1)
        init_time = time.time() - start_time

        print(f"✓ Model initialized in {init_time:.2f}s")
        test_results.append(("Model initialization", True, init_time))
    except Exception as e:
        print(f"✗ Failed: {e}")
        test_results.append(("Model initialization", False, 0))
        return False

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Test 2: First generation (cold start)
    print("\n" + "-" * 60)
    print("Test 2: First Generation (Cold Start)")
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

        start_time = time.time()
        output = llm.generate([prompt], params, None)[0]
        first_gen_time = time.time() - start_time

        print(f"✓ First generation completed in {first_gen_time:.2f}s")
        print(f"  Output: {output['text'][:50]!r}...")
        test_results.append(("First generation", True, first_gen_time))
    except Exception as e:
        print(f"✗ Failed: {e}")
        test_results.append(("First generation", False, 0))

    # Test 3: Second generation (warm)
    print("\n" + "-" * 60)
    print("Test 3: Second Generation (Warm)")
    print("-" * 60)
    try:
        start_time = time.time()
        output = llm.generate([prompt], params, None)[0]
        second_gen_time = time.time() - start_time

        print(f"✓ Second generation completed in {second_gen_time:.2f}s")
        print(f"  Speedup: {first_gen_time / second_gen_time:.2f}x")
        test_results.append(("Second generation", True, second_gen_time))
    except Exception as e:
        print(f"✗ Failed: {e}")
        test_results.append(("Second generation", False, 0))

    # Test 4: Batch warmup
    print("\n" + "-" * 60)
    print("Test 4: Batch Generation Warmup")
    print("-" * 60)
    try:
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": f"Test {i}"}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for i in range(3)
        ]
        params_list = [params] * len(prompts)

        # First batch
        start_time = time.time()
        _ = llm.generate(prompts, params_list, None)
        first_batch_time = time.time() - start_time

        # Second batch
        start_time = time.time()
        _ = llm.generate(prompts, params_list, None)
        second_batch_time = time.time() - start_time

        print(f"✓ First batch: {first_batch_time:.2f}s")
        print(f"✓ Second batch: {second_batch_time:.2f}s")
        print(f"  Speedup: {first_batch_time / second_batch_time:.2f}x")
        test_results.append(("Batch warmup", True, first_batch_time))
    except Exception as e:
        print(f"✗ Failed: {e}")
        test_results.append(("Batch warmup", False, 0))

    # Test 5: Sparse attention warmup
    print("\n" + "-" * 60)
    print("Test 5: Sparse Attention Warmup")
    print("-" * 60)
    try:
        from dllmserve.sparse.state import SparseConfig

        sparse_config = SparseConfig(
            enabled=True, retention_ratio=0.5, default_block_len=16
        )

        # First sparse generation
        start_time = time.time()
        _ = llm.generate([prompt], params, sparse_config)
        first_sparse_time = time.time() - start_time

        # Second sparse generation
        start_time = time.time()
        _ = llm.generate([prompt], params, sparse_config)
        second_sparse_time = time.time() - start_time

        print(f"✓ First sparse generation: {first_sparse_time:.2f}s")
        print(f"✓ Second sparse generation: {second_sparse_time:.2f}s")
        print(f"  Speedup: {first_sparse_time / second_sparse_time:.2f}x")
        test_results.append(("Sparse warmup", True, first_sparse_time))
    except Exception as e:
        print(f"✗ Failed: {e}")
        test_results.append(("Sparse warmup", False, 0))

    # Test 6: Different generation lengths
    print("\n" + "-" * 60)
    print("Test 6: Different Generation Lengths")
    print("-" * 60)
    try:
        lengths = [16, 32, 64, 128]
        for length in lengths:
            params_len = SamplingParams(
                temperature=0.0, gen_length=length, steps=length, cfg_scale=0.0
            )

            start_time = time.time()
            _ = llm.generate([prompt], params_len, None)
            elapsed = time.time() - start_time

            tokens_per_sec = length / elapsed
            print(f"  gen_length={length}: {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)")

        test_results.append(("Different lengths", True, 0))
    except Exception as e:
        print(f"✗ Failed: {e}")
        test_results.append(("Different lengths", False, 0))

    # Test 7: Model state consistency
    print("\n" + "-" * 60)
    print("Test 7: Model State Consistency")
    print("-" * 60)
    try:
        # Generate same prompt multiple times
        outputs = []
        for _ in range(3):
            output = llm.generate([prompt], params, None)[0]["text"]
            outputs.append(output)

        # All outputs should be non-empty
        assert all(len(out) > 0 for out in outputs), "Empty outputs detected"

        print(f"✓ All 3 generations produced non-empty outputs")
        print(f"  Output lengths: {[len(out) for out in outputs]}")
        test_results.append(("State consistency", True, 0))
    except Exception as e:
        print(f"✗ Failed: {e}")
        test_results.append(("State consistency", False, 0))

    # Test 8: GPU memory warmup
    if torch.cuda.is_available():
        print("\n" + "-" * 60)
        print("Test 8: GPU Memory Warmup")
        print("-" * 60)
        try:
            memory_samples = []
            for i in range(5):
                torch.cuda.reset_peak_memory_stats()
                _ = llm.generate([prompt], params, None)
                peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
                memory_samples.append(peak_mem)

            print(f"✓ Peak memory samples: {[f'{m:.0f}MB' for m in memory_samples]}")

            # Check memory is stable (not growing)
            if max(memory_samples) - min(memory_samples) < 100:  # Less than 100MB variation
                print(f"  Memory is stable")
            else:
                print(f"  Warning: Memory variation > 100MB")

            test_results.append(("GPU memory warmup", True, 0))
        except Exception as e:
            print(f"✗ Failed: {e}")
            test_results.append(("GPU memory warmup", False, 0))

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    passed = sum(1 for _, success, _ in test_results if success)
    total = len(test_results)
    print(f"Passed: {passed}/{total}")

    for test_name, success, elapsed in test_results:
        status = "✓" if success else "✗"
        if elapsed > 0:
            print(f"  {status} {test_name} ({elapsed:.2f}s)")
        else:
            print(f"  {status} {test_name}")

    # Performance summary
    print("\nPerformance Insights:")
    init_result = next((r for r in test_results if r[0] == "Model initialization"), None)
    first_gen_result = next((r for r in test_results if r[0] == "First generation"), None)
    second_gen_result = next((r for r in test_results if r[0] == "Second generation"), None)

    if init_result and first_gen_result:
        total_cold_start = init_result[2] + first_gen_result[2]
        print(f"  Total cold start time: {total_cold_start:.2f}s")

    if first_gen_result and second_gen_result:
        speedup = first_gen_result[2] / second_gen_result[2] if second_gen_result[2] > 0 else 0
        print(f"  Warmup speedup: {speedup:.2f}x")

    print("=" * 80)

    return all(success for _, success, _ in test_results)


if __name__ == "__main__":
    success = test_model_warmup()
    exit(0 if success else 1)
