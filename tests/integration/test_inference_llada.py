#!/usr/bin/env python3
"""LLaDA model integration tests."""
import gc
import os
import torch
from dllmserve import LLM, SamplingParams
from transformers import AutoTokenizer


def test_llada_batch_generation(llm, tokenizer):
    """Test batch generation with LLaDA."""
    print("\n" + "=" * 70)
    print("LLaDA Batch Generation Test")
    print("=" * 70)

    prompts = [
        "What is 2+2?",
        "What is the capital of France?",
        "Write a haiku.",
    ]

    params = SamplingParams(temperature=0.0, gen_length=32, steps=32, cfg_scale=0.0, remasking="low_confidence")

    formatted = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]

    outputs = llm.generate(formatted, [params] * len(prompts), None)
    assert len(outputs) == len(prompts)
    for i, out in enumerate(outputs):
        assert len(out["text"]) > 5, f"Output {i} too short"
        print(f"  {i+1}. {out['text'][:80]}...")
    print("✓ Batch generation test passed")


def test_llada_remasking_strategies(llm, tokenizer):
    """Test different remasking strategies."""
    print("\n" + "=" * 70)
    print("LLaDA Remasking Strategies Test")
    print("=" * 70)

    prompt = "Tell me a short story."
    formatted = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)

    for remasking in ["low_confidence", "random"]:
        params = SamplingParams(temperature=0.0, gen_length=32, steps=32, cfg_scale=0.0, remasking=remasking)
        outputs = llm.generate([formatted], [params], None)
        assert len(outputs[0]["text"]) > 5
        print(f"  ✓ {remasking}: {outputs[0]['text'][:50]}...")


def test_llada_variable_lengths(llm, tokenizer):
    """Test variable generation lengths."""
    print("\n" + "=" * 70)
    print("LLaDA Variable Generation Lengths Test")
    print("=" * 70)

    prompt = "Tell me a short story."
    formatted = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)

    params = [
        SamplingParams(temperature=0.0, gen_length=16, steps=16, cfg_scale=0.0, remasking="low_confidence"),
        SamplingParams(temperature=0.0, gen_length=64, steps=64, cfg_scale=0.0, remasking="low_confidence"),
    ]

    outputs = llm.generate([formatted, formatted], params, None)
    assert len(outputs) == 2
    print(f"  ✓ Length 16: {len(outputs[0]['text'])} chars")
    print(f"  ✓ Length 64: {len(outputs[1]['text'])} chars")


def main():
    print("=" * 70)
    print("LLaDA Integration Tests")
    print("=" * 70)

    MODEL_PATH = os.environ.get("LLADA_MODEL_PATH", "llada-instruct")
    llm = LLM(MODEL_PATH, enforce_eager=True, tensor_parallel_size=1)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Run all tests with the same LLM instance to avoid memory issues
    test_llada_batch_generation(llm, tokenizer)
    test_llada_remasking_strategies(llm, tokenizer)
    test_llada_variable_lengths(llm, tokenizer)

    # Final cleanup
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("\n" + "=" * 70)
    print("All LLaDA tests PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
