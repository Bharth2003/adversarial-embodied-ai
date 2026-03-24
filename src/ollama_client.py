# src/ollama_client.py
# Optimized for CPU i5 — reduced timeouts, faster retries, smaller token budget
import requests
import time
import json

def ollama_generate(model: str, prompt: str, temperature: float = 0.7, top_p: float = 0.9, num_predict: int = 200):
    """
    Optimized Ollama API client for CPU-constrained environments.
    - Reduced num_predict default from 400 → 200
    - Faster retry with shorter waits
    - Timeout tuned for smaller models
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": num_predict,
            # CPU optimization flags
            "num_thread": 4,          # Match i5 core count
            "num_ctx": 2048,          # Reduced context window (faster)
            "repeat_penalty": 1.1,    # Slight penalty to avoid loops
        }
    }

    # Try up to 3 times with shorter waits (was 5 attempts)
    for attempt in range(3):
        try:
            response = requests.post(url, json=payload, timeout=180)

            if response.status_code == 200:
                return response.json().get('response', '')
            elif response.status_code == 500:
                wait_time = (attempt + 1) * 3  # 3s, 6s, 9s (was 5, 10, 15)
                print(f"\n[Ollama Busy 500] Model {model} is loading. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"\n[Ollama Error {response.status_code}] {response.text}")
                return ""
        except requests.exceptions.RequestException as e:
            wait_time = (attempt + 1) * 3
            print(f"\n[Connection Error] Retrying in {wait_time}s...")
            time.sleep(wait_time)

    return ""