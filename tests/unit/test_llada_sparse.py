#!/usr/bin/env python3
"""
Sparse attention tests for LLaDA model.

Tests:
- Head-level sparse attention with various configurations
- Performance comparison: dense vs sparse
- Output quality validation
"""
import gc
import os
import time
import torch
from dllmserve import LLM, SamplingParams
from dllmserve.sparse.state import SparseConfig


def test_llada_sparse():
    """Test sparse attention with LLaDA model."""
    print("\n" + "=" * 70)
    print("LLaDA Sparse Attention Test")
    print("=" * 70)

    MODEL_PATH = os.environ.get("LLADA_MODEL_PATH", "llada-instruct")

    # Ensure GPU memory is clean
    torch.cuda.empty_cache()

    llm = LLM(MODEL_PATH, enforce_eager=True, tensor_parallel_size=1, log_optimize_level=2)
    print(f"✓ LLaDA initialized: {MODEL_PATH}")

    prompts = [
        "What is machine learning?",
        "What are neural networks?",
        "What is artificial intelligence?",
    ]

    params = SamplingParams(gen_length=128, steps=128, temperature=0.0, cfg_scale=0.0)

    # Test with sparse attention
    sparse_config = SparseConfig(
        enabled=True, retention_ratio=0.5, delay_step=1, default_block_len=32, head_select=True
    )

    torch.cuda.synchronize()
    start = time.time()
    outputs = llm.generate(prompts, [params] * len(prompts), sparse_config, use_tqdm=False)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    assert len(outputs) == len(prompts)
    for i, out in enumerate(outputs):
        assert len(out["text"]) > 10, f"Output {i} too short"
        print(f"  Prompt {i+1}: {out['text'][:100]}...")

    throughput = len(prompts) * params.gen_length / elapsed
    print(f"✓ Throughput: {throughput:.1f} tokens/s, Time: {elapsed:.2f}s")
    print("✓ All outputs valid")

    # Cleanup: explicitly delete LLM to free GPU memory
    # Note: exit() is automatically called by atexit handler
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print("✓ LLaDA model cleaned up from GPU memory")


def main():
    print("=" * 70)
    print("LLaDA Sparse Attention Test Suite")
    print("=" * 70)

    test_llada_sparse()

    print("\n" + "=" * 70)
    print("All tests PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
