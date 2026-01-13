#!/usr/bin/env python3
"""
Comprehensive test runner for dllm-serve.
Runs all test suites and generates a summary report.
"""
import os
import sys
import time
import subprocess
from pathlib import Path


# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    """Print a formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}")
    print(f"{text:^80}")
    print(f"{'=' * 80}{Colors.ENDC}\n")


def run_test(test_path, test_name, env_vars=None):
    """
    Run a single test file and return results.

    Args:
        test_path: Path to test file
        test_name: Display name for the test
        env_vars: Optional environment variables

    Returns:
        dict: Test results
    """
    print(f"{Colors.BOLD}Running: {test_name}{Colors.ENDC}")
    print(f"File: {test_path}")

    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, test_path],
            env=env,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        elapsed = time.time() - start_time

        success = result.returncode == 0

        return {
            "name": test_name,
            "path": test_path,
            "success": success,
            "elapsed": elapsed,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return {
            "name": test_name,
            "path": test_path,
            "success": False,
            "elapsed": elapsed,
            "stdout": "",
            "stderr": "Test timed out after 10 minutes",
            "returncode": -1,
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "name": test_name,
            "path": test_path,
            "success": False,
            "elapsed": elapsed,
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
        }


def main():
    """Run all tests and generate report."""
    print_header("DLLM-Serve Comprehensive Test Suite")

    # Setup environment variables
    base_dir = Path(__file__).parent.parent
    llada_model = os.environ.get("LLADA_MODEL_PATH", "./llada-instruct")
    dream_model = os.environ.get("DREAM_MODEL_PATH", f"./Dream-v0-Instruct-7B")


    env_vars = {
        "LLADA_MODEL_PATH": llada_model,
        "DREAM_MODEL_PATH": dream_model,
    }

    print(f"Environment:")
    print(f"  LLADA_MODEL_PATH: {llada_model}")
    print(f"  DREAM_MODEL_PATH: {dream_model}")
    print()

    # Define test suites - only including tests that exist
    test_suites = {
        "Unit Tests": [
            ("tests/unit/test_llada_sparse.py", "Sparse Attention (LLaDA)"),
            ("tests/unit/test_dream_sparse.py", "Sparse Attention (Dream)"),
            ("tests/unit/test_model_warmup.py", "Model Warmup"),
            ("tests/unit/test_edge_cases.py", "Edge Cases"),
        ],
        "Integration Tests": [
            ("tests/integration/test_inference_dream.py", "Dream Inference"),
            ("tests/integration/test_inference_llada.py", "LLaDA Inference"),
        ],
        "Performance Tests": [
            ("tests/performance/test_memory_management.py", "Memory Management"),
            ("tests/performance/test_log_optimize.py", "Log Optimization"),
        ],
    }

    # Optional tests (may require specific setup)
    optional_tests = {
        "Optional Tests": [
            ("tests/integration/test_server.py", "Server Integration"),
        ],
    }

    # Benchmark tests removed - files no longer exist
    benchmark_tests = {}

    all_results = []
    suite_summaries = []

    # Run main test suites
    for suite_name, tests in test_suites.items():
        print_header(suite_name)

        suite_results = []
        for test_file, test_name in tests:
            test_path = base_dir / test_file

            if not test_path.exists():
                print(f"{Colors.WARNING}⚠ Test file not found: {test_path}{Colors.ENDC}")
                continue

            result = run_test(str(test_path), test_name, env_vars)
            suite_results.append(result)
            all_results.append(result)

            # Print result
            if result["success"]:
                print(f"{Colors.OKGREEN}✓ PASSED{Colors.ENDC} ({result['elapsed']:.1f}s)\n")
            else:
                print(f"{Colors.FAIL}✗ FAILED{Colors.ENDC} ({result['elapsed']:.1f}s)")
                if result["stderr"]:
                    print(f"Error: {result['stderr'][:200]}")
                print()

        # Suite summary
        passed = sum(1 for r in suite_results if r["success"])
        total = len(suite_results)
        suite_summaries.append((suite_name, passed, total))

        print(f"{suite_name}: {passed}/{total} passed\n")

    # Run optional tests
    print_header("Optional Tests (may be skipped)")
    for suite_name, tests in optional_tests.items():
        for test_file, test_name in tests:
            test_path = base_dir / test_file

            if not test_path.exists():
                print(f"{Colors.WARNING}⚠ Skipping: {test_name} (file not found){Colors.ENDC}")
                continue

            print(f"Running optional test: {test_name}")
            result = run_test(str(test_path), test_name, env_vars)

            if result["success"]:
                print(f"{Colors.OKGREEN}✓ PASSED{Colors.ENDC} ({result['elapsed']:.1f}s)\n")
            else:
                print(f"{Colors.WARNING}⚠ FAILED{Colors.ENDC} (optional, {result['elapsed']:.1f}s)\n")

    # Final Summary
    print_header("Test Summary")

    for suite_name, passed, total in suite_summaries:
        status_color = Colors.OKGREEN if passed == total else Colors.FAIL
        print(f"{status_color}{suite_name}: {passed}/{total}{Colors.ENDC}")

    total_passed = sum(1 for r in all_results if r["success"])
    total_tests = len(all_results)

    print(f"\n{Colors.BOLD}Overall: {total_passed}/{total_tests} tests passed{Colors.ENDC}")

    # Detailed failures
    failures = [r for r in all_results if not r["success"]]
    if failures:
        print(f"\n{Colors.FAIL}{Colors.BOLD}Failed Tests:{Colors.ENDC}")
        for f in failures:
            print(f"  - {f['name']}")
            if f["stderr"]:
                print(f"    {f['stderr'][:100]}...")

    # Performance summary
    total_time = sum(r["elapsed"] for r in all_results)
    print(f"\n{Colors.BOLD}Total execution time: {total_time:.1f}s{Colors.ENDC}")

    print("\n" + "=" * 80)

    # Exit with appropriate code
    if total_passed == total_tests:
        print(f"{Colors.OKGREEN}All tests passed!{Colors.ENDC}")
        return 0
    else:
        print(f"{Colors.FAIL}Some tests failed.{Colors.ENDC}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
