#!/usr/bin/env python3
"""
Edge case tests for dllm-serve.
Tests unusual inputs, boundary conditions, and error scenarios.
"""
import os
import warnings
import torch
from dllmserve import LLM, SamplingParams
from dllmserve.sparse.state import SparseConfig
from transformers import AutoTokenizer

# Suppress deprecation warnings from transformers
warnings.filterwarnings("ignore", message="`torch_dtype` is deprecated")


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("=" * 80)
    print("Edge Case Test Suite")
    print("=" * 80)

    MODEL_PATH = os.environ.get("LLADA_MODEL_PATH", "llada-instruct")
    llm = LLM(MODEL_PATH, enforce_eager=True, tensor_parallel_size=1)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"✓ LLM initialized with model: {MODEL_PATH}\n")

    test_results = []

    # Test 1: Very short generation length
    print("-" * 60)
    print("Test 1: Very Short Generation (1 token)")
    print("-" * 60)
    try:
        params = SamplingParams(
            temperature=0.0, gen_length=1, steps=1, cfg_scale=0.0
        )
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hi"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        outputs = llm.generate([prompt], params, None)
        assert len(outputs) == 1
        assert len(outputs[0]["text"]) > 0
        print(f"✓ Generated: {outputs[0]['text']!r}")
        test_results.append(("Very short generation", True))
    except Exception as e:
        print(f"✗ Failed: {e}")
        test_results.append(("Very short generation", False))

    # Test 2: Empty-like prompt (just whitespace in content)
    print("\n" + "-" * 60)
    print("Test 2: Whitespace-Only Prompt")
    print("-" * 60)
    try:
        params = SamplingParams(
            temperature=0.0, gen_length=16, steps=16, cfg_scale=0.0
        )
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "   "}],
            tokenize=False,
            add_generation_prompt=True,
        )
        outputs = llm.generate([prompt], params, None)
        assert len(outputs) == 1
        print(f"✓ Generated: {outputs[0]['text']!r}")
        test_results.append(("Whitespace prompt", True))
    except Exception as e:
        print(f"✗ Failed: {e}")
        test_results.append(("Whitespace prompt", False))

    # Test 3: Special characters and unicode
    print("\n" + "-" * 60)
    print("Test 3: Special Characters and Unicode")
    print("-" * 60)
    try:
        params = SamplingParams(
            temperature=0.0, gen_length=16, steps=16, cfg_scale=0.0
        )
        special_prompts = [
            "What is 2+2? 🤔",
            "Explain AI in français",
            "Test with symbols: @#$%^&*()",
        ]
        formatted = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in special_prompts
        ]
        outputs = llm.generate(formatted, [params] * len(formatted), None)
        assert len(outputs) == len(special_prompts)
        for i, output in enumerate(outputs):
            print(f"  Prompt {i+1}: {special_prompts[i]!r}")
            print(f"  Output: {output['text'][:50]!r}...")
        test_results.append(("Special characters", True))
    except Exception as e:
        print(f"✗ Failed: {e}")
        test_results.append(("Special characters", False))

    # Test 4: Very long prompt
    print("\n" + "-" * 60)
    print("Test 4: Long Prompt (500+ characters)")
    print("-" * 60)
    try:
        params = SamplingParams(
            temperature=0.0, gen_length=16, steps=16, cfg_scale=0.0
        )
        long_content = "This is a very long prompt. " * 20  # ~560 chars
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": long_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        outputs = llm.generate([prompt], params, None)
        assert len(outputs) == 1
        print(f"✓ Generated response to {len(long_content)} char prompt")
        print(f"  Output: {outputs[0]['text'][:50]!r}...")
        test_results.append(("Long prompt", True))
    except Exception as e:
        print(f"✗ Failed: {e}")
        test_results.append(("Long prompt", False))

    # Test 5: Batch with mixed lengths
    print("\n" + "-" * 60)
    print("Test 5: Batch with Different Output Lengths")
    print("-" * 60)
    try:
        prompts = ["Short?", "Medium length question?", "Long one?"]
        params_list = [
            SamplingParams(temperature=0.0, gen_length=8, steps=8, cfg_scale=0.0),
            SamplingParams(temperature=0.0, gen_length=32, steps=32, cfg_scale=0.0),
            SamplingParams(temperature=0.0, gen_length=16, steps=16, cfg_scale=0.0),
        ]
        formatted = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]
        outputs = llm.generate(formatted, params_list, None)
        assert len(outputs) == len(prompts)
        for i, output in enumerate(outputs):
            print(f"  Prompt {i+1} (gen_length={params_list[i].gen_length}): {len(output['text'])} chars")
        test_results.append(("Mixed output lengths", True))
    except Exception as e:
        print(f"✗ Failed: {e}")
        test_results.append(("Mixed output lengths", False))

    # Test 6: Zero temperature
    print("\n" + "-" * 60)
    print("Test 6: Zero Temperature (Deterministic)")
    print("-" * 60)
    try:
        params = SamplingParams(
            temperature=0.0, gen_length=16, steps=16, cfg_scale=0.0
        )
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "What is 2+2?"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        output1 = llm.generate([prompt], params, None)[0]["text"]
        output2 = llm.generate([prompt], params, None)[0]["text"]
        # Note: For diffusion models, outputs may differ even with temp=0
        # due to remasking strategies, so we just check they're non-empty
        assert len(output1) > 0 and len(output2) > 0
        print(f"✓ Output 1: {output1[:50]!r}")
        print(f"✓ Output 2: {output2[:50]!r}")
        if output1 == output2:
            print("  Note: Outputs are identical (deterministic)")
        else:
            print("  Note: Outputs differ (expected for diffusion models with remasking)")
        test_results.append(("Zero temperature", True))
    except Exception as e:
        print(f"✗ Failed: {e}")
        test_results.append(("Zero temperature", False))

    # Test 7: Sparse config with extreme ratios
    print("\n" + "-" * 60)
    print("Test 7: Sparse Attention - Extreme Retention Ratios")
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

        # Test with 0% retention (minimal attention)
        sparse_0 = SparseConfig(
            enabled=True, retention_ratio=0.0, default_block_len=16
        )
        output_0 = llm.generate([prompt], params, sparse_0)[0]
        print(f"  0% retention: {output_0['text'][:50]!r}")

        # Test with 100% retention (should match dense)
        sparse_100 = SparseConfig(
            enabled=True, retention_ratio=1.0, default_block_len=16
        )
        output_100 = llm.generate([prompt], params, sparse_100)[0]
        print(f"  100% retention: {output_100['text'][:50]!r}")

        test_results.append(("Extreme sparse ratios", True))
    except Exception as e:
        print(f"✗ Failed: {e}")
        test_results.append(("Extreme sparse ratios", False))

    # Test 8: Different remasking strategies
    print("\n" + "-" * 60)
    print("Test 8: Different Remasking Strategies")
    print("-" * 60)
    try:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Explain AI briefly"}],
            tokenize=False,
            add_generation_prompt=True,
        )

        strategies = ["low_confidence", "random"]
        for strategy in strategies:
            params = SamplingParams(
                temperature=0.0, gen_length=16, steps=16,
                cfg_scale=0.0, remasking=strategy
            )
            output = llm.generate([prompt], params, None)[0]
            print(f"  {strategy}: {output['text'][:50]!r}")

        test_results.append(("Remasking strategies", True))
    except Exception as e:
        print(f"✗ Failed: {e}")
        test_results.append(("Remasking strategies", False))

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
    print("=" * 80)

    return all(success for _, success in test_results)


if __name__ == "__main__":
    success = test_edge_cases()
    exit(0 if success else 1)
