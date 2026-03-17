import requests
import os
import sys

def test_ollama():
    base = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434').rstrip('/')
    url = f"{base}/api/chat"
    model = os.environ.get('OLLAMA_MODEL', 'deepseek-coder:6.7b').strip() or 'deepseek-coder:6.7b'
    timeout_seconds = int(os.environ.get('OLLAMA_TIMEOUT_SECONDS', '30'))
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False
    }
    print(f"Connecting to {url} with model {model} (timeout={timeout_seconds}s)")
    try:
        response = requests.post(url, json=payload, timeout=timeout_seconds)
        print(f"Status code: {response.status_code}")
        print("Response JSON:")
        print(response.json())
    except requests.exceptions.Timeout:
        print(f"Request timed out after {timeout_seconds} seconds")
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    test_ollama()