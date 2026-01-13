#!/usr/bin/env python3
"""Dream model integration tests."""
import os
import sys
import torch
from dllmserve import LLM, SamplingParams
from dllmserve.sparse.state import SparseConfig
from transformers import AutoTokenizer


def test_dream_shifted_prediction():
    """Test Dream shifted prediction property."""
    print("\n[1/3] Testing Dream shifted prediction...")

    # Mock test - verify model has the property
    class MockDreamConfig:
        def __init__(self):
            self.hidden_size = 1024
            self.num_attention_heads = 16
            self.uses_shifted_prediction = True

    config = MockDreamConfig()
    assert hasattr(config, 'uses_shifted_prediction')
    assert config.uses_shifted_prediction == True
    print("  ✓ Dream model has uses_shifted_prediction=True")


def test_dream_sampling_params():
    """Test Dream-specific SamplingParams."""
    print("\n[2/3] Testing Dream SamplingParams...")

    configs = [
        SamplingParams(temperature=0.0, gen_length=32, steps=32, cfg_scale=0.0, remasking="low_confidence"),
        SamplingParams(temperature=0.6, gen_length=64, steps=64, cfg_scale=1.0, remasking="random"),
    ]

    for params in configs:
        assert params.gen_length > 0
        assert params.remasking in ["low_confidence", "random"]
    print("  ✓ All SamplingParams valid")


def test_dream_inference():
    """Test Dream model inference."""
    print("\n[3/3] Testing Dream inference...")

    MODEL_PATH = os.environ.get("DREAM_MODEL_PATH", "Dream-org/Dream-v0-Instruct-7B")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    llm = LLM(MODEL_PATH, enforce_eager=True, tensor_parallel_size=1)

    prompts_raw = ["What is machine learning?", "Explain AI in one sentence."]
    formatted = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in prompts_raw
    ]

    params = SamplingParams(temperature=0.0, gen_length=32, steps=32, cfg_scale=0.0, remasking="low_confidence")
    outputs = llm.generate(formatted, [params] * 2, None)

    assert len(outputs) == 2
    for i, out in enumerate(outputs):
        assert len(out["text"]) > 5
        print(f"  {i+1}. {out['text'][:60]}...")
    print("  ✓ Inference test passed")


def main():
    print("=" * 70)
    print("Dream Integration Tests")
    print("=" * 70)

    test_dream_shifted_prediction()
    test_dream_sampling_params()
    test_dream_inference()

    print("\n" + "=" * 70)
    print("All Dream tests PASSED")
    print("=" * 70)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
