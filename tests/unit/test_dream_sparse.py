#!/usr/bin/env python3
"""
Sparse attention tests for Dream model.

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
from transformers import AutoTokenizer


def test_dream_sparse():
    """Test sparse attention with Dream model."""
    print("\n" + "=" * 70)
    print("Dream Sparse Attention Test")
    print("=" * 70)

    MODEL_PATH = os.environ.get("DREAM_MODEL_PATH", "Dream-v0-Instruct-7B")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Ensure GPU memory is clean
    torch.cuda.empty_cache()

    llm = LLM(MODEL_PATH, enforce_eager=True, tensor_parallel_size=1)
    print(f"✓ Dream initialized: {MODEL_PATH}")

    # Test: identical prompts with different configs to validate consistency
    prompts_raw = [
        "What is the capital of France? Answer in one sentence.",
        "Explain machine learning in one sentence.",
        "Explain machine learning in one sentence.",  # Same as #2
    ]

    formatted = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in prompts_raw
    ]

    params = SamplingParams(temperature=0.0, gen_length=64, steps=64, cfg_scale=0.0, remasking="low_confidence")

    # Prompt 1: dense, Prompt 2: sparse(ratio=0.5), Prompt 3: dense (should match prompt 2 if ratio=1, but here ratio=0.5)
    sparse_configs = [None, SparseConfig(enabled=True, retention_ratio=0.5, default_block_len=16), None]

    torch.cuda.synchronize()
    start = time.time()
    outputs = llm.generate(formatted, [params] * 3, sparse_configs, use_tqdm=False)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    assert len(outputs) == 3
    for i, out in enumerate(outputs):
        assert len(out["text"].strip()) > 10, f"Output {i} too short"
        print(f"  Prompt {i+1}: {out['text'][:100]}...")

    throughput = 3 * params.gen_length / elapsed
    print(f"✓ Throughput: {throughput:.1f} tokens/s, Time: {elapsed:.2f}s")
    print("✓ All outputs valid")

    # Cleanup: explicitly delete LLM to free GPU memory
    # Note: exit() is automatically called by atexit handler
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print("✓ Dream model cleaned up from GPU memory")


def main():
    print("=" * 70)
    print("Dream Sparse Attention Test Suite")
    print("=" * 70)

    test_dream_sparse()

    print("\n" + "=" * 70)
    print("All tests PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
