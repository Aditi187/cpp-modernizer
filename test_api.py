#!/usr/bin/env python3
"""Test script for cpp-modernizer API"""

import requests
import json
import sys

# Configuration
API_URL = "http://localhost:8000"
API_TOKEN = "cpp-modernizer-dev-token"

def test_health():
    """Test health endpoint"""
    print("\n🧪 Test 1: Health Check")
    print("=" * 60)
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        response.raise_for_status()
        data = response.json()
        print("✅ Health Check PASSED")
        print(json.dumps(data, indent=2))
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_modernize():
    """Test modernize endpoint"""
    print("\n🧪 Test 2: Modernize Single File")
    print("=" * 60)
    
    # Create test file
    legacy_code = """#include <stdio.h>

int main() {
    int *ptr = (int*)malloc(sizeof(int) * 10);
    printf("Hello\\n");
    free(ptr);
    return 0;
}"""
    
    with open("test_legacy.cpp", "w") as f:
        f.write(legacy_code)
    
    print("📝 Created test file: test_legacy.cpp")
    print("Content:")
    print("─" * 60)
    print(legacy_code)
    print("─" * 60)
    
    try:
        headers = {
            "Authorization": f"Bearer {API_TOKEN}"
        }
        
        with open("test_legacy.cpp", "rb") as f:
            files = {
                "file": f
            }
            params = {
                "skip_verify": "true"
            }
            
            print("\n📤 Sending to modernizer API endpoint: /modernize/file...")
            response = requests.post(
                f"{API_URL}/modernize/file",
                headers=headers,
                files=files,
                params=params,
                timeout=120
            )
            
            if response.status_code != 200:
                print(f"⚠️  API returned {response.status_code}")
                print(f"Response: {response.text}")
                return False
            
            result = response.json()
            
            print("✅ Modernization SUCCESSFUL")
            print("\n📊 Results:")
            print(f"  Success:             {result.get('success', 'N/A')}")
            print(f"  Modernization Score: {result.get('score', 'N/A')}")
            print(f"  Safety Rating:       {result.get('safety_rating', 'N/A')}")
            print(f"  Compiler Status:     {result.get('compiler_status', 'N/A')}")
            print(f"  Attribution:         {result.get('attribution', 'N/A')}")
            print(f"  Processing Time:     {result.get('processing_time_ms', 'N/A')} ms")
            print(f"  Legacy Patterns:     {result.get('legacy_patterns_found', 0)}")
            print(f"  Tokens Used:         {result.get('tokens_used', 'N/A')}")
            
            if result.get('modernized_code'):
                print("\n📝 Modernized Code:")
                print("─" * 60)
                modernized = result['modernized_code']
                print(modernized[:500])
                if len(modernized) > 500:
                    print("... (truncated)")
                print("─" * 60)
            
            if result.get('diff'):
                print("\n📋 Transformation Diff:")
                diff_info = result['diff']
                print(f"  Added Lines:    {diff_info.get('added_lines', 0)}")
                print(f"  Removed Lines:  {diff_info.get('removed_lines', 0)}")
                print("  Preview:")
                print("  " + "\n  ".join(diff_info.get('diff_preview', '').split('\n')[:10]))
            
            return True
            
    except Exception as e:
        print(f"❌ Modernization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("  CPP-MODERNIZER API TEST SUITE")
    print("=" * 60)
    
    # Test health first
    if not test_health():
        print("\n⚠️  API is not responding. Make sure:")
        print("   1. Ollama is running: ollama serve")
        print("   2. API server is running: python -m uvicorn api:app --port 8000")
        sys.exit(1)
    
    # Test modernize
    test_modernize()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
