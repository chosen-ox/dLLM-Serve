#!/usr/bin/env python3
"""
Simple test script for the DLLM-Serve server.
Starts the server and sends requests to test the complete workflow.
"""

import os
import time
import requests
import subprocess
import signal
import sys
import json
from threading import Thread


class ServerTester:
    def __init__(self, host="127.0.0.1", port=8000):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.server_process = None
        self.log_file = "server_test_log.txt"

    def start_server(self):
        """Start the FastAPI server"""
        print("Starting server...")
        # Adjust the command based on your server file name
        cmd = [
            "uvicorn",
            "server.app:app",
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--workers",
            "1",
        ]
        # Set MODEL_PATH to use the llada-instruct model
        env = os.environ.copy()
        env["MODEL_PATH"] = "llada-instruct"
        # self.server_process = subprocess.Popen(
        #     cmd,
        #     stdout=subprocess.PIPE,
        #     stderr=subprocess.PIPE,
        #     text=True,
        #     env=env,
        # )
        self.server_process = subprocess.Popen(
            cmd, stdout=open(self.log_file, "w"), stderr=open(self.log_file, "w"), text=True, env=env
        )

        # Wait for server to start
        max_retries = 30
        for i in range(max_retries):
            try:
                response = requests.get(f"{self.base_url}/v1/health", timeout=2)
                if response.status_code == 200:
                    print(f"✅ Server started successfully at {self.base_url}")
                    return True
            except requests.exceptions.RequestException:
                pass

            time.sleep(1)
            print(f"Waiting for server to start... ({i+1}/{max_retries})")

        print("❌ Failed to start server")
        return False

    def stop_server(self):
        """Stop the server"""
        if self.server_process:
            print("Stopping server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            print("✅ Server stopped")
            # Clean up log file
            if os.path.exists(self.log_file):
                os.remove(self.log_file)
                print("✅ Log file cleaned up")

    def test_health(self):
        """Test health endpoint"""
        print("\n🔍 Testing health endpoint...")
        try:
            response = requests.get(f"{self.base_url}/v1/health")
            if response.status_code == 200:
                print("✅ Health check passed")
                print(f"Response: {response.json()}")
                return True
            else:
                print(f"❌ Health check failed with status: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False

    def test_single_generate(self):
        """Test single generation request"""
        print("\n🔍 Testing single generation...")

        # Submit request
        payload = {
            "prompt": "What is machine learning?",
            "temperature": 0.7,
            "gen_length": 32,
        }

        try:
            response = requests.post(f"{self.base_url}/v1/generate", json=payload)
            if response.status_code != 200:
                print(f"❌ Generate request failed with status: {response.status_code}")
                print(f"Response: {response.text}")
                return False

            result = response.json()
            print(f"✅ Generation submitted successfully")
            print(f"Request IDs: {result['request_ids']}")

            # Get the request ID
            request_id = result["request_ids"][0]

            # Poll for results
            return self.poll_for_result(request_id)

        except Exception as e:
            print(f"❌ Single generation test failed: {e}")
            return False

    def test_batch_generate(self):
        """Test batch generation request"""
        print("\n🔍 Testing batch generation...")

        # Submit batch request
        payload = {
            "prompts": [
                "What is artificial intelligence?",
                "What are neural networks?",
                "What is quantum computer?",
            ],
            "temperature": 0.5,
            "gen_length": 32,
        }

        try:
            response = requests.post(f"{self.base_url}/v1/generate_batch", json=payload)
            if response.status_code != 200:
                print(
                    f"❌ Batch generate request failed with status: {response.status_code}"
                )
                print(f"Response: {response.text}")
                return False

            result = response.json()
            print(f"✅ Batch generation submitted successfully")
            print(f"Request IDs: {result['request_ids']}")

            # Poll for all results
            all_success = True
            for request_id in result["request_ids"]:
                success = self.poll_for_result(request_id)
                all_success = all_success and success

            return all_success

        except Exception as e:
            print(f"❌ Batch generation test failed: {e}")
            return False

    def poll_for_result(self, request_id, max_wait=60):
        """Poll for result until completion or timeout"""
        print(f"📋 Polling for result of request {request_id}...")

        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                response = requests.get(f"{self.base_url}/v1/result/{request_id}")

                if response.status_code == 404:
                    print(f"❌ Request ID {request_id} not found")
                    return False

                if response.status_code != 200:
                    print(
                        f"❌ Result request failed with status: {response.status_code}"
                    )
                    return False

                result = response.json()
                status = result["status"]

                print(f"Status: {status}")

                if status == "finished":
                    print(f"✅ Request {request_id} completed!")
                    print(f"Generated text: {result.get('text', 'No text returned')}")
                    return True
                elif status in ["processing", "submitted", "running"]:
                    time.sleep(2)  # Wait before polling again
                else:
                    print(f"❌ Unexpected status: {status}")
                    return False

            except Exception as e:
                print(f"❌ Error polling result: {e}")
                return False

        print(f"❌ Timeout waiting for request {request_id}")
        return False

    def test_invalid_request_id(self):
        """Test requesting result for invalid ID"""
        print("\n🔍 Testing invalid request ID...")

        try:
            response = requests.get(f"{self.base_url}/v1/result/99999")
            if response.status_code == 404:
                print("✅ Invalid request ID correctly returned 404")
                return True
            else:
                print(f"❌ Expected 404, got {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Invalid request ID test failed: {e}")
            return False

    def run_all_tests(self):
        """Run all tests"""
        print("🚀 Starting DLLM-Serve Server Test Suite")
        print("=" * 50)

        if not self.start_server():
            return False

        try:
            # Run tests
            tests = [
                ("Health Check", self.test_health),
                ("Single Generation", self.test_single_generate),
                ("Batch Generation", self.test_batch_generate),
                ("Invalid Request ID", self.test_invalid_request_id),
            ]

            results = []
            for test_name, test_func in tests:
                print(f"\n{'='*20} {test_name} {'='*20}")
                success = test_func()
                results.append((test_name, success))

            # Print summary
            print("\n" + "=" * 50)
            print("📊 TEST SUMMARY")
            print("=" * 50)

            passed = 0
            for test_name, success in results:
                status = "✅ PASS" if success else "❌ FAIL"
                print(f"{test_name}: {status}")
                if success:
                    passed += 1

            print(f"\nTotal: {passed}/{len(results)} tests passed")

            if passed == len(results):
                print("🎉 All tests passed!")
                return True
            else:
                print("❌ Some tests failed!")
                return False

        finally:
            self.stop_server()


def main():
    """Main function"""

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n\nReceived interrupt signal. Stopping...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Run tests
    tester = ServerTester()
    success = tester.run_all_tests()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
