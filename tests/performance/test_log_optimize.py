#!/usr/bin/env python3
"""
Log optimization level tests for dllm-serve.

Tests that different log_optimize_level values produce consistent outputs:
- Level 0: No optimization
- Level 1: Normal optimization
- Level 2: Aggressive optimization

Note: Each level runs in a separate subprocess to ensure clean GPU memory,
as PyTorch doesn't reliably release memory for large models.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Test script that runs in each subprocess
_TEST_SCRIPT = '''
import os
import sys
import time
import torch
from dllmserve import LLM, SamplingParams

def run_level(level):
    """Run a single log_optimize_level test."""
    MODEL_PATH = os.environ.get("LLADA_MODEL_PATH", "llada-instruct")
    prompt = "What is machine learning?"
    params = SamplingParams(gen_length=32, steps=32, temperature=0.0, cfg_scale=0.0)

    llm = LLM(
        MODEL_PATH,
        enforce_eager=True,
        tensor_parallel_size=1,
        log_optimize_level=level,
        max_model_len=512,
        gpu_memory_utilization=0.8,
    )

    torch.cuda.synchronize()
    start = time.time()
    outputs = llm.generate([prompt], [params], None, use_tqdm=False)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    result = outputs[0]["text"]
    print(f"OUTPUT:{result}")
    print(f"TIME:{elapsed:.2f}")
    return result

if __name__ == "__main__":
    level = int(sys.argv[1])
    try:
        output = run_level(level)
        # Write output to temp file for parent process
        temp_file = sys.argv[2]
        with open(temp_file, 'w') as f:
            f.write(output)
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
'''


def test_log_optimize_levels():
    """Test consistency across different log_optimize_level values."""
    print("=" * 70)
    print("Log Optimization Level Consistency Test")
    print("=" * 70)
    print("Note: Running each level in separate subprocess for clean GPU memory\n")

    results = {}
    times = {}

    # Run each level in a separate subprocess
    for level in [0, 1, 2]:
        print(f"--- Testing log_optimize_level={level} ---")

        # Create temp file for output
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            temp_file = f.name

        try:
            # Run test in subprocess
            result = subprocess.run(
                [sys.executable, '-c', _TEST_SCRIPT, str(level), temp_file],
                env=os.environ.copy(),
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                print(f"  ✗ FAILED: {result.stderr[:200]}")
                return False

            # Parse output
            output = None
            elapsed = None
            for line in result.stdout.strip().split('\n'):
                if line.startswith('OUTPUT:'):
                    output = line[7:]
                elif line.startswith('TIME:'):
                    elapsed = float(line[5:])

            if output is None:
                print(f"  ✗ FAILED: Could not parse output")
                return False

            results[level] = output
            times[level] = elapsed
            print(f"  Output: {output[:80]}...")
            print(f"  Time: {elapsed:.2f}s")

        except subprocess.TimeoutExpired:
            print(f"  ✗ FAILED: Timeout")
            return False
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass

    # Check consistency
    print("\n" + "=" * 70)
    print("Consistency Check")
    print("=" * 70)

    if results[0] == results[1] == results[2]:
        print("✓ All outputs are CONSISTENT across log_optimize_level 0, 1, 2")
        return True
    else:
        print("✗ Outputs DIFFER across log_optimize_level values:")
        for level, text in results.items():
            print(f"  Level {level}: {text[:80]}...")
        return False


def main():
    success = test_log_optimize_levels()
    print("\n" + "=" * 70)
    print("✓ PASSED" if success else "✗ FAILED")
    print("=" * 70)
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
