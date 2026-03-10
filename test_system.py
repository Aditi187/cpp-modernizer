"""
System Integration Test Harness.

This script tests the three core pillars of the engine independently:
1. Environment: Is g++ accessible and working?
2. Parsing: Can Tree-sitter actually 'see' C++ functions?
3. Intelligence: Is Ollama responding to modernization prompts?
"""

import os
import subprocess
import json
import logging
from core.parser import extract_functions_from_cpp_file
from core.differential_tester import _compile_and_run_cpp
import requests

# Setup logging to see detailed output
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_compiler():
    """Test 1: Compiler Path and Version."""
    print("\n--- [TEST 1] COMPILER CHECK ---")
    gpp_exe = r"C:\msys64\mingw64\bin\g++.exe"
    if not os.path.exists(gpp_exe):
        print(f"❌ FAIL: g++ not found at {gpp_exe}")
        return False
    
    try:
        result = subprocess.run([gpp_exe, "--version"], capture_output=True, text=True)
        print(f"✅ SUCCESS: {result.stdout.splitlines()[0]}")
        return True
    except Exception as e:
        print(f"❌ FAIL: Compiler execution failed: {e}")
        return False

def test_parser():
    """Test 2: Tree-sitter Extraction."""
    print("\n--- [TEST 2] PARSER CHECK ---")
    dummy_code = """
    int add(int a, int b) { return a + b; }
    void greet() { return; }
    """
    with open("smoke_test.cpp", "w") as f:
        f.write(dummy_code)
    
    try:
        functions = extract_functions_from_cpp_file("smoke_test.cpp")
        names = [fn['name'] for fn in functions]
        if "add" in names and "greet" in names:
            print(f"✅ SUCCESS: Extracted {len(names)} functions: {names}")
            return True
        else:
            print(f"❌ FAIL: Expected ['add', 'greet'], got {names}")
            return False
    except Exception as e:
        print(f"❌ FAIL: Parser crashed: {e}")
        return False
    finally:
        if os.path.exists("smoke_test.cpp"):
            os.remove("smoke_test.cpp")

def test_ollama():
    """Test 3: Ollama Connectivity and Model Availability."""
    print("\n--- [TEST 3] OLLAMA CHECK ---")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = [m['name'] for m in response.json().get('models', [])]
            target = "deepseek-coder:6.7b"
            if target in models:
                print(f"✅ SUCCESS: Ollama is running and '{target}' is loaded.")
                return True
            else:
                print(f"❌ FAIL: Ollama is running, but '{target}' is missing. Run 'ollama pull {target}'")
                return False
        else:
            print(f"❌ FAIL: Ollama returned status {response.status_code}")
            return False
    except Exception as e:
        print("❌ FAIL: Ollama not reachable. Ensure the Ollama app is running.")
        return False

if __name__ == "__main__":
    results = {
        "Compiler": test_compiler(),
        "Parser": test_parser(),
        "Ollama": test_ollama()
    }
    
    print("\n" + "="*30)
    print("FINAL TEST SUMMARY")
    print("="*30)
    for test, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test.ljust(15)}: {status}")
    
    if all(results.values()):
        print("\n🚀 ALL SYSTEMS GO: You are ready to run the full workflow.")
    else:
        print("\n⚠️  FIX ERRORS: Resolve the 'FAIL' items before running the engine.")