#!/usr/bin/env python3
"""
Test script to verify the vLLM classifier /classify endpoint is working correctly.
Usage: python test_classify_endpoint.py [--port 8000] [--hostname localhost]
"""
import argparse
import httpx
import json
import sys

def test_classify_endpoint(hostname="localhost", port=8000):
    """Test the /classify endpoint with a sample request."""
    base_url = f"http://{hostname}:{port}"

    print(f"Testing vLLM classifier at {base_url}")
    print("=" * 80)

    # Test 1: Health check
    print("\n[1] Testing /health endpoint...")
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{base_url}/health")
            resp.raise_for_status()
            print(f"✓ Health check passed (status: {resp.status_code})")
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False

    # Test 2: Models endpoint
    print("\n[2] Testing /v1/models endpoint...")
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{base_url}/v1/models")
            resp.raise_for_status()
            models = resp.json()
            print(f"✓ Models endpoint responded")
            print(f"  Available models: {json.dumps(models, indent=2)}")
    except Exception as e:
        print(f"✗ Models endpoint failed: {e}")
        return False

    # Test 3: Classify endpoint
    print("\n[3] Testing /classify endpoint...")

    # Build a sample input similar to what partition_reward_vllm_serve.py sends
    # Format: CLS + sentence1 + SEP + sentence2 + SEP
    # For simplicity, we'll just send space-separated text (the server should handle tokenization)
    test_input = "[CLS] This is a test sentence. [SEP] This is another test sentence. [SEP]"

    payload = {
        "model": "similarity_gpu_0",  # Model name pattern from partition_reward_vllm_serve.py
        "input": [test_input],
    }

    print(f"  Request payload: {json.dumps(payload, indent=2)}")

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(f"{base_url}/classify", json=payload)
            resp.raise_for_status()
            result = resp.json()
            print(f"✓ Classify endpoint responded (status: {resp.status_code})")
            print(f"  Response: {json.dumps(result, indent=2)}")

            # Check if we got the expected structure
            if "data" in result and len(result["data"]) > 0 and "probs" in result["data"][0]:
                probs = result["data"][0]["probs"]
                print(f"  ✓ Got probability scores: {probs}")
                print(f"  Similarity probability (last element): {probs[-1]:.4f}")
                return True
            else:
                print(f"  ✗ Unexpected response structure")
                return False

    except httpx.HTTPError as e:
        print(f"✗ Classify endpoint HTTP error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"  Response content: {e.response.text}")
        return False
    except Exception as e:
        print(f"✗ Classify endpoint failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

def main():
    parser = argparse.ArgumentParser(description="Test vLLM classifier /classify endpoint")
    parser.add_argument("--hostname", default="localhost", help="Server hostname (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    args = parser.parse_args()

    success = test_classify_endpoint(args.hostname, args.port)

    print("\n" + "=" * 80)
    if success:
        print("✓ All tests PASSED - classifier endpoint is ready!")
        sys.exit(0)
    else:
        print("✗ Tests FAILED - classifier endpoint has issues")
        sys.exit(1)

if __name__ == "__main__":
    main()
