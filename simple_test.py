#!/usr/bin/env python3
"""
Simple test script to verify the API is working correctly.
"""

import json
import requests

def test_health():
    """Test the health endpoint."""
    try:
        response = requests.get('http://localhost:8080/health')
        print(f"Health check: {response.status_code} - {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_create_bucket():
    """Test creating a vector bucket."""
    try:
        payload = {"vectorBucketName": "test-bucket"}
        response = requests.post(
            'http://localhost:8080/CreateVectorBucket',
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        print(f"Create bucket: {response.status_code} - {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Create bucket failed: {e}")
        return False

def test_list_buckets():
    """Test listing vector buckets."""
    try:
        response = requests.post(
            'http://localhost:8080/ListVectorBuckets',
            json={},
            headers={'Content-Type': 'application/json'}
        )
        print(f"List buckets: {response.status_code} - {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"List buckets failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing S3 Vectors API")
    
    tests = [
        ("Health Check", test_health),
        ("Create Vector Bucket", test_create_bucket),
        ("List Vector Buckets", test_list_buckets)
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\n🧪 Testing {name}...")
        if test_func():
            print(f"✅ {name}: PASSED")
            passed += 1
        else:
            print(f"❌ {name}: FAILED")
    
    print(f"\n📊 Results: {passed}/{total} tests passed ({100*passed/total:.1f}%)")
